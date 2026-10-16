"""Experimental original-column wrapper search; no application state is written."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
import threading
from numbers import Real
from pathlib import Path
import platform
import re
from time import perf_counter
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ResamplingPipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from Helpers import prepare_estimator_input
from JBGCVReporting import partition_signature


def finite_number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


@dataclass(frozen=True)
class SearchOptions:
    trials: int = 100
    sizes: tuple = (5, 10, 20, 40, 60)
    inner_folds: int = 5
    outer_folds: int = 3
    top_sets: int = 2
    refine_rounds: int = 2
    max_evaluations: int = 300
    min_gain: float = 0.005
    seed: int = 42
    scaler: str = 'STA'
    oversampling: str = 'NOG'
    reduction: str = 'NOR'
    C: float = 1.0
    max_iterations: int = 2000
    model_type: str = 'Linear'
    non_linear_max_iterations: int = 2000
    non_linear_alpha: float = 0.0001
    non_linear_hidden_layers: tuple = (100,)
    non_linear_learning_rate: float = 0.001
    tree_estimator: str = 'Extra Trees'
    tree_n_estimators: int = 100
    tree_max_depth: int | None = None
    tree_min_samples_leaf: int = 1
    tree_class_weight: str = 'None'
    tree_positive_threshold: float = 0.5
    parallel_workers: int = 1
    catboost_iterations: int = 400
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05
    catboost_class_weight: str = 'None'

    @property
    def planned_fit_upper_bound(self):
        return (self.outer_folds + 1) * self.max_evaluations * self.inner_folds + 2 * self.outer_folds

    def validate(self):
        for name in ('trials', 'inner_folds', 'outer_folds', 'top_sets', 'max_evaluations',
                     'max_iterations', 'non_linear_max_iterations', 'tree_n_estimators',
                     'tree_min_samples_leaf', 'parallel_workers', 'catboost_iterations', 'catboost_depth'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.inner_folds < 2 or self.outer_folds < 2:
            raise ValueError('Inner and outer CV each require at least two folds')
        if type(self.refine_rounds) is not int or self.refine_rounds < 0:
            raise ValueError('refine_rounds must be a nonnegative integer')
        if self.max_evaluations < self.trials + 1:
            raise ValueError('max_evaluations must allow the full-column baseline plus all random trials')
        if not self.sizes or any(type(n) is not int or n < 1 for n in self.sizes):
            raise ValueError('Subset sizes must be positive integers')
        if self.reduction not in ('NOR', 'PCA'):
            raise ValueError('reduction must be NOR or PCA')
        if self.oversampling not in ('NOG', 'RND', 'SME'):
            raise ValueError('oversampling must be NOG, RND or SME')
        if self.scaler not in ('STA', 'MAX') or not finite_number(self.C) or self.C <= 0:
            raise ValueError('Use scaler STA/MAX and positive finite C')
        if not finite_number(self.min_gain) or self.min_gain < 0:
            raise ValueError('min_gain must be finite and nonnegative')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if self.tree_class_weight not in ('None', 'balanced', 'balanced_subsample'):
            raise ValueError('tree_class_weight must be None, balanced or balanced_subsample')
        if not finite_number(self.tree_positive_threshold) or not 0 < self.tree_positive_threshold < 1:
            raise ValueError('tree_positive_threshold must be in (0, 1)')
        if self.tree_estimator not in ('Extra Trees', 'Random Forest'):
            raise ValueError('tree_estimator must be Extra Trees or Random Forest')
        if self.model_type not in ('Linear', 'Non-linear', 'Tree', 'Combination', 'CatBoost'):
            raise ValueError('model_type must be Linear, Non-linear, Tree, Combination or CatBoost')
        if self.catboost_depth > 16 or not finite_number(self.catboost_learning_rate) or not 0 < self.catboost_learning_rate <= 1:
            raise ValueError('CatBoost depth must be <= 16 and learning rate in (0, 1]')
        if self.catboost_class_weight not in ('None', 'Balanced'):
            raise ValueError('CatBoost class weight must be None or Balanced')
        if self.model_type == 'CatBoost' and (self.oversampling != 'NOG' or self.reduction != 'NOR'):
            raise ValueError('CatBoost native categoricals require No Oversampling and No Reduction; SMOTE/PCA are incompatible')
        if (not isinstance(self.non_linear_hidden_layers, tuple) or not self.non_linear_hidden_layers
                or any(type(n) is not int or n < 1 for n in self.non_linear_hidden_layers)):
            raise ValueError('non_linear_hidden_layers must be a tuple of positive integers')
        if not finite_number(self.non_linear_alpha) or self.non_linear_alpha < 0:
            raise ValueError('non_linear_alpha must be finite and nonnegative')
        if not finite_number(self.non_linear_learning_rate) or self.non_linear_learning_rate <= 0:
            raise ValueError('non_linear_learning_rate must be finite and positive')
        if self.tree_max_depth is not None and (type(self.tree_max_depth) is not int or self.tree_max_depth < 1):
            raise ValueError('tree_max_depth must be None or a positive integer')


class NativeCatBoostEncoder(TransformerMixin, BaseEstimator):
    """Preserve original categorical values (not one-hot/text converted)."""
    def __init__(self, columns, text_options=None):
        self.columns = columns
        self.text_options = text_options

    def fit(self, X, y=None):
        settings = self.text_options or {}
        configured = set(settings.get('text_columns', ())) | set(settings.get('category_columns', ()))
        self.categorical_columns_ = tuple(c for c in self.columns
            if c in configured or not pd.api.types.is_numeric_dtype(X[c]))
        return self

    def transform(self, X):
        frame = X.loc[:, list(self.columns)].copy()
        for name in self.columns:
            if name in self.categorical_columns_:
                # CatBoost requires categorical missing values to be strings.
                frame[name] = frame[name].astype('string').fillna('__MISSING__').astype(str)
            else:
                frame[name] = pd.to_numeric(frame[name], errors='coerce').astype(float)
        return frame


class NativeCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable estimator detecting native categorical columns after fold-local encoding."""
    def __init__(self, iterations=400, depth=6, learning_rate=0.05,
                 class_weight='None', random_seed=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.random_seed = random_seed

    def fit(self, X, y):
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError('CatBoost requires the optional catboost package: pip install catboost') from exc
        categorical = [name for name in X.columns
                       if not pd.api.types.is_numeric_dtype(X[name])]
        self.estimator_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, random_seed=self.random_seed,
            auto_class_weights=('Balanced' if self.class_weight == 'Balanced' else None),
            cat_features=categorical, thread_count=1, verbose=False,
            allow_writing_files=False)
        self.estimator_.fit(X, y)
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).ravel()

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def make_estimator(options):
    """One fixed investigative estimator; independent of ordinary model grids."""
    options.validate()
    if options.model_type == 'Linear':
        return 'logistic', LogisticRegression(C=options.C, solver='lbfgs',
            max_iter=options.max_iterations, random_state=options.seed)
    if options.model_type == 'Non-linear':
        return 'mlp', MLPClassifier(hidden_layer_sizes=options.non_linear_hidden_layers,
            activation='relu', solver='adam', alpha=options.non_linear_alpha,
            learning_rate_init=options.non_linear_learning_rate,
            max_iter=options.non_linear_max_iterations, random_state=options.seed,
            early_stopping=False, warm_start=False)
    if options.model_type == 'CatBoost':
        return 'catboost', NativeCatBoostClassifier(iterations=options.catboost_iterations,
            depth=options.catboost_depth, learning_rate=options.catboost_learning_rate,
            class_weight=options.catboost_class_weight, random_seed=options.seed)
    if options.model_type == 'Combination':
        # Mirror JBG's FUTV: soft voting over MLP, RandomForest and AdaBoost.
        # The original grid searches voting weights; investigative weights are
        # fixed at the historically successful (1, 2, 1) configuration.
        return 'futv', VotingClassifier(estimators=[
            ('mlpc', MLPClassifier(max_iter=20000, random_state=options.seed)),
            ('rfcl', RandomForestClassifier(random_state=options.seed, n_jobs=1)),
            ('abc', AdaBoostClassifier(random_state=options.seed)),
        ], voting='soft', weights=(1, 2, 1), n_jobs=1)
    estimator_class = (RandomForestClassifier if options.tree_estimator == 'Random Forest'
                       else ExtraTreesClassifier)
    estimator = estimator_class(n_estimators=options.tree_n_estimators,
        max_depth=options.tree_max_depth, min_samples_leaf=options.tree_min_samples_leaf,
        criterion='gini', max_features='sqrt', class_weight=(None if options.tree_class_weight == 'None' else options.tree_class_weight),
        random_state=options.seed, n_jobs=1, warm_start=False)
    return ('random_forest' if options.tree_estimator == 'Random Forest' else 'extra_trees'), estimator


def _json_model_parameter(value):
    """Record nested sklearn estimator settings without storing live objects."""
    if isinstance(value, BaseEstimator):
        return dict(estimator=f'{type(value).__module__}.{type(value).__name__}',
                    parameters={key: _json_model_parameter(item)
                                for key, item in value.get_params(deep=False).items()})
    if isinstance(value, (list, tuple)):
        return [_json_model_parameter(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_model_parameter(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def model_identity(options):
    _, estimator = make_estimator(options)
    return dict(model_type=options.model_type,
                estimator=f'{type(estimator).__module__}.{type(estimator).__name__}',
                parameters={key: _json_model_parameter(value)
                            for key, value in estimator.get_params(deep=False).items()})


def study_provenance(X, context=None):
    """Accept explicit source identifiers, never arbitrary application configs."""
    context = dict(context or {})
    fields = {'project', 'source_kind', 'source_tables', 'class_column', 'id_column',
              'excluded_columns', 'group_column'}
    if set(context) - fields:
        raise ValueError('Unsupported study context fields; pass source identifiers only')
    for key, value in context.items():
        if key in ('source_tables', 'excluded_columns'):
            if not isinstance(value, (tuple, list)) or not all(isinstance(v, str) for v in value):
                raise ValueError(f'{key} must be a list of strings')
        elif not isinstance(value, str):
            raise ValueError(f'{key} must be a string')
    schema = [{'name': c, 'dtype': str(X[c].dtype)} for c in X.columns]
    digest = hashlib.sha256(json.dumps(schema, ensure_ascii=False).encode('utf-8'))
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy(dtype='<u8').tobytes())
    source_dir = Path(__file__).resolve().parent
    code = hashlib.sha256()
    for name in ('JBGFeatureSearch.py', 'JBGCVReporting.py', 'Helpers.py', 'JBGTransformers.py'):
        code.update(name.encode('utf-8'))
        code.update((source_dir / name).read_bytes())
    changelog = source_dir.parent / 'CHANGELOG.md'
    revision = re.search(r'^## (\d+)\b', changelog.read_text(encoding='utf-8-sig'), re.MULTILINE) if changelog.exists() else None
    return dict(context=context, original_column_schema=schema,
                development_feature_sha256=digest.hexdigest(),
                feature_search_revision='130', jbg_revision=revision.group(1) if revision else None,
                backend_source_sha256=code.hexdigest(), python_version=platform.python_version(),
                sklearn_version=sklearn.__version__,
                validation_status='candidate',
                limitations='Source dtypes are loaded pandas types, not verified SQL schema or feature meanings. '
                            'Nested CV validates the procedure, not the exact final subset or another table.')


class FoldColumnEncoder(TransformerMixin, BaseEstimator):
    """Construct a fresh JBG converter using only selected training-fold columns.

    The existing converter mutates its fitted column lists and is not directly
    sklearn-cloneable. This adapter keeps constructor arguments untouched.
    """
    def __init__(self, columns, text_options=None):
        self.columns = columns
        self.text_options = text_options

    def fit(self, X, y=None):
        self.converter_ = None
        settings = dict(self.text_options or {})
        text = [c for c in settings.pop('text_columns', ()) if c in self.columns]
        category = [c for c in settings.pop('category_columns', ()) if c in self.columns]
        if text or category:
            from JBGTransformers import TextDataToNumbersConverter
            self.converter_ = TextDataToNumbersConverter(
                text_columns=text, category_columns=category, **settings)
            self.converter_.fit(X.loc[:, list(self.columns)])
        self.n_features_in_ = len(self.columns)
        return self

    def transform(self, X):
        selected = X.loc[:, list(self.columns)]
        converted = self.converter_.transform(selected) if self.converter_ is not None else selected
        prepared = prepare_estimator_input(converted)
        if isinstance(prepared, pd.DataFrame):
            return prepared.to_numpy(dtype=float, na_value=np.nan)
        return prepared.astype(float)


def make_pipeline(columns, options, text_options):
    name, estimator = make_estimator(options)
    if options.model_type == 'CatBoost':
        return Pipeline([('columns', NativeCatBoostEncoder(tuple(columns), text_options)),
                         (name, estimator)])
    scaler = MaxAbsScaler() if options.scaler == 'MAX' else StandardScaler(with_mean=False)
    steps = [
        ('columns', FoldColumnEncoder(tuple(columns), text_options)),
        ('impute', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)),
        ('scale', scaler),
    ]
    if options.oversampling != 'NOG':
        # Reuse the ordinary JBG oversampler definitions, but place resampling
        # strictly inside the training fold of every inner/outer CV fit.
        from JBGMeta import Oversampling
        method = Oversampling[options.oversampling]
        sampler = method.get_callable_oversampler()
        sampler.set_params(random_state=options.seed)
        steps.append(('oversampling', sampler))
    if options.reduction == 'PCA':
        # Match ordinary JBG's 95%-variance PCA; applied AFTER SMOTE, inside CV.
        steps.append(('pca', PCA(n_components=0.95, svd_solver='covariance_eigh')))
    steps.append((name, estimator))
    return ResamplingPipeline(steps) if options.oversampling != 'NOG' else Pipeline(steps)


def investigative_predictions(model, X, target, options):
    """Apply a fixed threshold only to binary tree models; never tune on validation rows."""
    if options.model_type != 'Tree' or options.tree_positive_threshold == 0.5:
        return model.predict(X)
    labels = model.classes_
    if len(labels) != 2:
        raise ValueError('Custom positive threshold is supported for binary Tree investigations only')
    probability = model.predict_proba(X)[:, list(labels).index(target)]
    negative = next(label for label in labels if label != target)
    return np.where(probability >= options.tree_positive_threshold, target, negative)

def classification_counts(y, predicted, target):
    actual = np.asarray(y) == target
    positive = np.asarray(predicted) == target
    return dict(tp=int(np.sum(actual & positive)), fp=int(np.sum(~actual & positive)),
                fn=int(np.sum(actual & ~positive)), tn=int(np.sum(~actual & ~positive)))


def metrics(counts):
    tp, fp, fn, tn = (counts[name] for name in ('tp', 'fp', 'fn', 'tn'))
    return dict(precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0.,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                predicted_positives=tp + fp, target_support=tp + fn, rows=tp + fp + fn + tn,
                **counts)



def multiclass_diagnostics(actual, predicted, labels):
    """Pooled out-of-fold multiclass reports, never per-fold metric averaging."""
    matrix = confusion_matrix(actual, predicted, labels=labels).astype(int)
    per_class = {}
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        per_class[str(label)] = dict(precision=precision, recall=recall, f1=f1,
                                     support=int(matrix[i, :].sum()), predicted=int(matrix[:, i].sum()),
                                     tp=tp, fp=fp, fn=fn)
    return dict(macro_precision=float(np.mean([v['precision'] for v in per_class.values()])),
                macro_recall=float(np.mean([v['recall'] for v in per_class.values()])),
                macro_f1=float(np.mean([v['f1'] for v in per_class.values()])),
                class_labels=[str(label) for label in labels], confusion_matrix=matrix.tolist(),
                per_class=per_class)


def add_matrices(matrices):
    return np.sum(np.asarray(matrices, dtype=int), axis=0).tolist()


def diagnostics_from_matrix(matrix, labels):
    # Expand count matrix only to reuse the exact pooled diagnostic definition.
    actual, predicted = [], []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            actual.extend([labels[i]] * int(count))
            predicted.extend([labels[j]] * int(count))
    return multiclass_diagnostics(actual, predicted, labels)

def folds_for(X, y, count, seed, groups=None):
    if min(Counter(y).values()) < count:
        raise ValueError(f'{count} CV folds require at least {count} rows of each class')
    if groups is None:
        splitter = StratifiedKFold(count, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y))
    else:
        if (groups.isna() | groups.astype(str).str.strip().isin(['', 'nan', 'None', 'NaT', '<NA>', '?'])).any() or groups.nunique() < count:
            raise ValueError('Grouping needs nonmissing groups and at least one group per fold')
        splitter = StratifiedGroupKFold(count, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y, groups))
    for train, test in folds:
        if set(y.iloc[train]) != set(y) or set(y.iloc[test]) != set(y):
            raise ValueError('Every training/validation fold must contain every class; reduce folds or reconsider groups')
        if groups is not None and set(groups.iloc[train]) & set(groups.iloc[test]):
            raise ValueError('A group crosses a CV boundary')
    return folds


def random_subsets(columns, options):
    """Deterministic, size-balanced proposals with broad subset-size coverage.

    Explicit sizes remain authoritative; seed controls reproducibility. The
    full-column baseline is evaluated separately and never duplicated.
    """
    columns = tuple(columns)
    n = len(columns)
    sizes = sorted(set(min(size, n) for size in options.sizes))
    rng = np.random.default_rng(options.seed)
    seen = {columns}
    result = []
    # Cycle through sizes rather than exhausting a single size. Prioritize
    # unexplored sizes if small search spaces contain many duplicates.
    for i in range(max(100, options.trials * 100)):
        size = sizes[i % len(sizes)]
        indices = set(rng.choice(n, size=size, replace=False).tolist())
        subset = tuple(c for j, c in enumerate(columns) if j in indices)
        if subset not in seen:
            seen.add(subset)
            result.append(subset)
        if len(result) == options.trials:
            return result
    if n <= 12:
        for size in sizes:
            for subset in combinations(columns, size):
                if subset not in seen:
                    seen.add(subset)
                    result.append(subset)
                if len(result) == options.trials:
                    return result
    return result


def ranking(row):
    # A zero-MCC tie is not evidence for removing columns. Protect the full baseline.
    baseline_tie = row.get('stage') == 'full_baseline' and row['mean_mcc'] == 0.0
    return (-row['mean_mcc'], 0 if baseline_tie else 1, row['n_columns'], row['std_mcc'], row['columns'])


class InvestigationPaused(Exception):
    """Cooperative pause at a completed candidate or outer-model boundary."""


def _checkpoint_fingerprint(X, y, target, options, text_options, groups, holdout_signature):
    """Bind checkpoints to feature VALUES, row ordering, labels and settings."""
    digest = hashlib.sha256()
    for frame in (X, y, groups):
        if frame is None:
            digest.update(b'NO_GROUPS')
            continue
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype='<u8').tobytes())
        if isinstance(frame, pd.DataFrame):
            digest.update(json.dumps([(str(k), str(v)) for k, v in frame.dtypes.items()],
                                     ensure_ascii=False).encode('utf-8'))
    payload = dict(columns=list(X.columns), target=str(target), options=asdict(options),
                   text_options=text_options or {}, reserved_holdout_signature=holdout_signature)
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8'))
    return digest.hexdigest()


def _atomic_json(path, record):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _restore_rows(path):
    if not Path(path).exists():
        return []
    result = []
    for record in pd.read_csv(path, sep=';', encoding='utf-8-sig').to_dict('records'):
        for key in ('columns', 'fold_mcc', 'seed_columns', 'before_columns', 'after_columns'):
            if key in record and isinstance(record[key], str):
                record[key] = tuple(json.loads(record[key])) if key != 'fold_mcc' else json.loads(record[key])
        if pd.isna(record.get('error')):
            record['error'] = ''
        result.append(record)
    return result


def _load_committed_records(directory, prefix):
    records = []
    for file in sorted((Path(directory) / '.checkpoints').glob(prefix + '*.json')):
        record = json.loads(file.read_text(encoding='utf-8'))
        for key in ('columns', 'seed_columns', 'before_columns', 'after_columns'):
            if key in record:
                record[key] = tuple(record[key])
        records.append(record)
    return records


def _commit_record(directory, identifier, record):
    destination = Path(directory) / '.checkpoints'
    destination.mkdir(exist_ok=True)
    _atomic_json(destination / (identifier + '.json'), record)



def _parallel_candidate_job(X, y, target, options, text_options, groups, columns, stage):
    """Evaluate a single subset in an isolated process; workers never write checkpoints."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        local = Search(X, y, target, options, text_options, 'worker', lambda row: None, groups=groups)
        row = local.evaluate(columns, stage)
    return row, local.fit_count


class Search:
    def __init__(self, X, y, target, options, text_options, scope, sink, groups=None, progress=None,
                 on_event=None, previous=None, pause_event=None, checkpoint_directory=None):
        self.X, self.y, self.target = X, y, target
        self.options, self.text_options = options, text_options
        self.scope, self.sink, self.progress = scope, sink, progress
        self.groups = groups
        self.folds = folds_for(X, y, options.inner_folds, options.seed, groups)
        self.cache = {tuple(r["columns"]): r for r in (previous or [])}
        self.moves = []
        self.pause_event = pause_event
        self.checkpoint_directory = checkpoint_directory
        self.fit_count = 0
        self.on_event = on_event
        self.executor = None

    def emit(self, event, **values):
        if self.on_event:
            self.on_event(dict(event=event, scope=self.scope, model_type=self.options.model_type, **values))

    def evaluate(self, columns, stage):
        if self.pause_event is not None and self.pause_event.is_set():
            raise InvestigationPaused('Paused at candidate checkpoint')
        columns = tuple(c for c in self.X.columns if c in columns)
        if columns in self.cache:
            return self.cache[columns]
        if len(self.cache) >= self.options.max_evaluations:
            return None
        started = perf_counter()
        scores, counts, warning_count = [], Counter(), 0
        actual_all, predicted_all = [], []
        error = ''
        try:
            for fold, (train, test) in enumerate(self.folds, 1):
                model = make_pipeline(columns, self.options, self.text_options)
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter('always', ConvergenceWarning)
                    self.fit_count += 1
                    self.emit('fit_started', phase='inner', candidate=len(self.cache) + 1,
                              fold=fold, folds=len(self.folds), n_columns=len(columns), stage=stage)
                    model.fit(self.X.iloc[train], self.y.iloc[train])
                warning_count += sum(issubclass(w.category, ConvergenceWarning) for w in recorded)
                predicted = investigative_predictions(model, self.X.iloc[test], self.target, self.options)
                scores.append(float(matthews_corrcoef(self.y.iloc[test], predicted)))
                counts.update(classification_counts(self.y.iloc[test], predicted, self.target))
                if self.y.nunique() > 2:
                    actual_all.extend(self.y.iloc[test].tolist())
                    predicted_all.extend(predicted.tolist())
                self.emit('fit_completed', phase='inner', candidate=len(self.cache) + 1,
                          fold=fold, folds=len(self.folds), n_columns=len(columns), stage=stage)
            if not np.isfinite(scores).all():
                raise ValueError('CV produced a nonfinite MCC')
        except Exception as ex:
            error = f'{type(ex).__name__}: {ex}'
        row = dict(scope=self.scope, candidate=len(self.cache) + 1, stage=stage,
                   model_type=self.options.model_type,
                   columns=columns, n_columns=len(columns), mean_mcc=float(np.mean(scores)) if not error else np.nan,
                   std_mcc=float(np.std(scores)) if not error else np.nan,
                   fold_mcc=scores, convergence_warnings=warning_count,
                   seconds=perf_counter() - started, error=error,
                   **metrics(counts),
                   **(multiclass_diagnostics(actual_all, predicted_all, sorted(set(self.y), key=str))
                      if self.y.nunique() > 2 and not error else {}))
        self.cache[columns] = row
        self.sink(row)
        self.emit('candidate_completed', candidate=len(self.cache),
                  max_evaluations=self.options.max_evaluations, n_columns=len(columns),
                  mean_mcc=row['mean_mcc'] if not error else None, error=error)
        if self.progress:
            self.progress(f'{self.scope}: candidate {len(self.cache)}/{self.options.max_evaluations}, '
                          f'{len(columns)} columns, MCC={row["mean_mcc"]:.4f}' + (' FAILED' if error else ''))
        return row

    def evaluate_batch(self, requests):
        """Commit in input order, even when worker completion order differs.

        Batches never exceed worker count: on termination, at most one small
        batch of calculated but uncommitted scores can need recomputation.
        """
        if self.options.parallel_workers == 1:
            return [self.evaluate(columns, stage) for columns, stage in requests]
        results = [None] * len(requests)
        pending = []
        reserved = set(self.cache)
        for index, (requested, stage) in enumerate(requests):
            columns = tuple(c for c in self.X.columns if c in requested)
            if columns in self.cache:
                results[index] = self.cache[columns]
            elif columns in reserved or len(reserved) >= self.options.max_evaluations:
                results[index] = None
            else:
                reserved.add(columns)
                pending.append((index, columns, stage))
        if self.pause_event is not None and self.pause_event.is_set():
            raise InvestigationPaused('Paused at parallel batch boundary')
        # A single persistent executor for the current batch. Never share sink,
        # progress callback, threads, lock or GUI state across processes.
        for offset in range(0, len(pending), self.options.parallel_workers):
            batch = pending[offset:offset + self.options.parallel_workers]
            if self.pause_event is not None and self.pause_event.is_set():
                raise InvestigationPaused('Paused at parallel batch boundary')
            futures = [self.executor.submit(_parallel_candidate_job, self.X, self.y, self.target,
                       self.options, self.text_options, self.groups, columns, stage)
                       for _, columns, stage in batch]
            for (index, columns, stage), future in zip(batch, futures):
                row, n_fits = future.result()
                row['scope'] = self.scope
                row['candidate'] = len(self.cache) + 1
                self.cache[columns] = row
                self.fit_count += n_fits
                # Process workers don't send GUI callbacks. Report committed
                # fit units on the parent thread after the result is durable.
                for fold_number in range(n_fits):
                    self.emit('fit_started', phase='inner', candidate=row['candidate'],
                              fold=fold_number + 1, folds=len(self.folds),
                              n_columns=len(columns), stage=stage)
                self.sink(row)
                self.emit('candidate_completed', candidate=len(self.cache),
                          max_evaluations=self.options.max_evaluations,
                          n_columns=len(columns), mean_mcc=row['mean_mcc'] if not row['error'] else None,
                          error=row['error'])
                if self.progress:
                    self.progress(f'{self.scope}: candidate {len(self.cache)}/{self.options.max_evaluations}, '
                                  f'{len(columns)} columns, MCC={row["mean_mcc"]:.4f}')
                results[index] = row
        return results

    def run(self):
        if self.options.parallel_workers == 1:
            return self._run_impl()
        # 'spawn' is safe from the Jupyter/Voilà background callback thread,
        # unlike forking an already multithreaded kernel.
        with ProcessPoolExecutor(max_workers=self.options.parallel_workers,
                                 mp_context=get_context('spawn')) as self.executor:
            try:
                return self._run_impl()
            finally:
                self.executor = None

    def _run_impl(self):
        self.emit('search_started' , max_evaluations=self.options.max_evaluations)
        columns = tuple(self.X.columns)
        self.evaluate(columns, 'full_baseline')
        self.evaluate_batch([(subset, 'random') for subset in random_subsets(columns, self.options)])
        # Exclude refinement results when reconstructing seed frontier on resume.
        successful = sorted((r for r in self.cache.values()
                             if not r['error'] and r.get('stage') != 'refine'), key=ranking)
        if not successful:
            raise ValueError(f'{self.scope}: no subset completed all CV folds; inspect subsets.csv')
        for seed_number, seed_row in enumerate(successful[:self.options.top_sets], 1):
            current = seed_row
            for round_number in range(self.options.refine_rounds):
                base = current['columns']
                neighbours = []
                for feature in columns:
                    if feature in base and len(base) > 1:
                        proposal = tuple(c for c in base if c != feature)
                        operation = 'remove'
                    elif feature not in base:
                        proposal = tuple(c for c in columns if c in base or c == feature)
                        operation = 'add'
                    else:
                        continue
                    neighbours.append((proposal, operation, feature))
                remaining = self.options.max_evaluations - len(self.cache)
                pending = [(p, op, feature) for p, op, feature in neighbours if p not in self.cache]
                plan_path = (Path(self.checkpoint_directory) / '.checkpoints' /
                             f'refinement_plan_{self.scope}_seed{seed_number}_round{round_number + 1}.json'
                             if self.checkpoint_directory is not None else None)
                if plan_path is not None and plan_path.is_file():
                    plan = json.loads(plan_path.read_text(encoding='utf-8'))
                    if tuple(plan['base_columns']) != tuple(base):
                        raise ValueError('Checkpoint refinement plan does not match the reconstructed search')
                    allowed = {tuple(value) for value in plan['allowed_columns']}
                    neighbours = [entry for entry in neighbours if entry[0] in allowed]
                else:
                    if remaining <= 0:
                        break
                    # Persist this immutable neighbourhood decision BEFORE fitting
                    # any proposals. Recomputing it after a pause could spend the
                    # reduced remaining budget on different candidates.
                    if len(pending) > remaining:
                        rng = np.random.default_rng(self.options.seed + 1009 * (round_number + 1) + 7919 * len(base))
                        indices = sorted(rng.choice(len(pending), size=remaining, replace=False).tolist())
                        chosen = [pending[i] for i in indices]
                        allowed = {p for p, _, _ in chosen} | set(self.cache)
                        neighbours = [entry for entry in neighbours if entry[0] in allowed]
                        self.emit('refinement_budget_limited', round=round_number + 1,
                                  proposed=len(pending), evaluated_new=len(chosen),
                                  remaining_before=remaining)
                    if plan_path is not None:
                        plan_path.parent.mkdir(exist_ok=True)
                        _atomic_json(plan_path, dict(base_columns=list(base),
                                                     allowed_columns=[list(p) for p, _, _ in neighbours]))
                rows = self.evaluate_batch([(p, 'refine') for p, _, _ in neighbours])
                evaluated = [(row, operation, feature) for row, (_, operation, feature) in zip(rows, neighbours)]
                usable = [(r, op, feature) for r, op, feature in evaluated if r is not None and not r['error']]
                if not usable:
                    break
                best, operation, feature = min(usable, key=lambda item: ranking(item[0]))
                gain = best['mean_mcc'] - current['mean_mcc']
                if gain <= self.options.min_gain:
                    break
                self.moves.append(dict(scope=self.scope, seed_columns=seed_row['columns'], round=round_number + 1,
                    operation=operation, feature=feature, before_columns=current['columns'],
                    after_columns=best['columns'], before_mcc=current['mean_mcc'],
                    after_mcc=best['mean_mcc'], gain=gain))
                current = best
        # Budget-limited neighbourhoods are exploratory, not proof of a local optimum.
        # Every evaluated proposal remains eligible for the final ranking.
        best = min((r for r in self.cache.values() if not r['error']), key=ranking)
        return best


def export_frame(rows, path, append=False, columns=None):
    frame = pd.DataFrame(rows, columns=columns)
    for name in ('columns', 'fold_mcc', 'seed_columns', 'before_columns', 'after_columns', 'class_labels', 'confusion_matrix', 'per_class'):
        if name in frame:
            frame[name] = frame[name].map(lambda value: json.dumps(value, ensure_ascii=False))
    path = Path(path)
    exists = append and path.exists()
    frame.to_csv(path, index=False, sep=';', mode='a' if append else 'w',
                 header=not exists, encoding='utf-8' if exists else 'utf-8-sig')


def probability_diagnostics(y, scores, target):
    """Pooled out-of-fold ranking and descriptive, NOT optimized, thresholds."""
    actual = np.asarray(y) == target
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(actual) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid out-of-fold probabilities')
    if len(np.unique(actual)) != 2:
        raise ValueError('Diagnostics require positive and negative observations')
    thresholds = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        tn = int(np.sum(~actual & ~predicted))
        rows.append(dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=tp / (tp + fp) if tp + fp else 0.,
                         recall=tp / (tp + fn) if tp + fn else 0.,
                         specificity=tn / (tn + fp) if tn + fp else 0.,
                         mcc=float(matthews_corrcoef(actual, predicted))))
    bins = np.linspace(0., 1., 11)
    calibration = []
    for index in range(10):
        mask = ((scores >= bins[index]) & (scores < bins[index+1])) if index < 9 else ((scores >= bins[index]) & (scores <= bins[index+1]))
        if mask.any():
            calibration.append(dict(lower=float(bins[index]), upper=float(bins[index+1]),
                count=int(mask.sum()), mean_probability=float(scores[mask].mean()),
                observed_rate=float(actual[mask].mean())))
    return dict(roc_auc=float(roc_auc_score(actual, scores)),
        average_precision=float(average_precision_score(actual, scores)),
        prevalence=float(actual.mean()), brier_score=float(brier_score_loss(actual, scores)),
        threshold_analysis=rows, calibration_bins=calibration,
        interpretation='Descriptive thresholds on pooled out-of-fold predictions; no threshold selected or independently validated.')


def subgroup_probability_diagnostics(y, scores, target, values, minimum_positives=30):
    """Compare pooled OOF scores within each stratum; never refit on stratum data."""
    values = pd.Series(values, index=y.index)
    if not values.index.equals(y.index):
        raise ValueError('Subgroup values must align with development labels')
    result = []
    names = values.astype('string').fillna('<missing>')
    if names.nunique() > 100:
        raise ValueError('Subgroup column has over 100 values; bin continuous columns first')
    for name in sorted(names.unique()):
        mask = (names == name).to_numpy(dtype=bool)
        positive = int((np.asarray(y)[mask] == target).sum())
        negative = int(mask.sum() - positive)
        row = dict(value=str(name), rows=int(mask.sum()), positives=positive,
                   negatives=negative, prevalence=positive / int(mask.sum()))
        if positive == 0 or negative == 0:
            row.update(status='undefined_single_class', roc_auc=None, average_precision=None,
                       brier_score=None, threshold_analysis=[], calibration_bins=[])
        elif positive < minimum_positives:
            row.update(status='insufficient_positives', roc_auc=None, average_precision=None,
                       brier_score=None, threshold_analysis=[], calibration_bins=[])
        else:
            row.update(status='ok', **probability_diagnostics(y.iloc[np.flatnonzero(mask)],
                                                               np.asarray(scores)[mask], target))
        result.append(row)
    return result


def evaluate_probability_diagnostics(X, y, target, options, text_options, groups=None,
                                     columns=None, pause_event=None, on_event=None,
                                     subgroup_column=None):
    """Fold-local training; each development record is predicted exactly once."""
    if len(set(y)) != 2:
        raise ValueError('Probability diagnostics currently support binary classification')
    columns = tuple(columns if columns is not None else X.columns)
    scores = np.full(len(y), np.nan, dtype=float)
    for fold, (train, test) in enumerate(folds_for(X, y, options.outer_folds, options.seed, groups), 1):
        if pause_event is not None and pause_event.is_set():
            raise InvestigationPaused('Paused before diagnostics fold')
        model = make_pipeline(columns, options, text_options)
        if on_event is not None:
            on_event(dict(event='fit_started', phase='diagnostics', model_type=options.model_type, fold=fold, folds=options.outer_folds))
        model.fit(X.iloc[train], y.iloc[train])
        if not hasattr(model, 'predict_proba'):
            raise ValueError('Model does not expose predict_proba for diagnostics')
        classes = list(model.classes_)
        scores[test] = model.predict_proba(X.iloc[test])[:, classes.index(target)]
        if on_event is not None:
            on_event(dict(event='diagnostics_fold_completed', fold=fold, folds=options.outer_folds))
    report = probability_diagnostics(y, scores, target)
    if subgroup_column:
        if subgroup_column not in X.columns:
            raise ValueError(f'Unknown subgroup column: {subgroup_column}')
        report['subgroups'] = subgroup_probability_diagnostics(y, scores, target,
                                                                 X[subgroup_column])
        report['subgroup_column'] = subgroup_column
    return report


def run_experiment(X, y, target, output, options=None, text_options=None, groups=None,
                   reserved_holdout_signature=None, progress=print, on_event=None, study_context=None,
                   resume=False, pause_event=None, investigation_mode="selection_plus_diagnostics", subgroup_column=None):
    """Nested CV evaluates selection on training data; external holdout never enters.

    A final search on all development rows supplies a recommendation, not an
    independently validated score for that exact final subset.
    """
    options = options or SearchOptions()
    options.validate()
    if investigation_mode not in ('diagnostics_only', 'selection_plus_diagnostics'):
        raise ValueError('Unknown investigation mode')
    if not isinstance(X, pd.DataFrame) or X.empty or X.shape[1] == 0 or X.columns.has_duplicates or not all(isinstance(c, str) for c in X.columns):
        raise ValueError('Need a nonempty DataFrame of uniquely named original columns')
    if not isinstance(y, pd.Series) or not X.index.equals(y.index) or X.index.has_duplicates:
        raise ValueError('Original columns and labels need aligned unique row IDs')
    if y.isna().any() or len(set(y)) < 2 or target not in set(y):
        raise ValueError('Investigation requires at least two classes and a target present in y')
    labels = sorted(set(y), key=str)
    is_multiclass = len(labels) > 2
    if is_multiclass and options.model_type == 'Tree' and options.tree_positive_threshold != 0.5:
        raise ValueError('Custom positive threshold requires binary classification; use 0.5 for multiclass')
    if groups is not None and not groups.index.equals(X.index):
        raise ValueError('Group labels and original rows must align')
    if subgroup_column is not None and subgroup_column not in X.columns:
        raise ValueError(f'Unknown subgroup column: {subgroup_column}')
    provenance = study_provenance(X, study_context)
    identity = model_identity(options)
    # Validate every inner partition before starting an expensive study.
    outer = folds_for(X, y, options.outer_folds, options.seed, groups)
    for train, _ in outer:
        folds_for(X.iloc[train], y.iloc[train], options.inner_folds, options.seed,
                  groups.iloc[train] if groups is not None else None)
    output = Path(output)
    summary_path = output / 'summary.json'
    fingerprint = _checkpoint_fingerprint(X, y, target, options, text_options, groups,
                                          reserved_holdout_signature)
    if resume:
        if not summary_path.is_file():
            raise ValueError('No checkpoint summary.json to resume')
        existing = json.loads(summary_path.read_text(encoding='utf-8'))
        if existing.get('subgroup_column') != subgroup_column:
            raise ValueError('Checkpoint subgroup variable mismatch')
        if existing.get('checkpoint_fingerprint') != fingerprint:
            raise ValueError('Checkpoint does not match current feature data, labels, settings or holdout')
        old_provenance = existing.get('provenance', {})
        for key in ('backend_source_sha256', 'sklearn_version'):
            if old_provenance.get(key) != provenance.get(key):
                raise ValueError('Checkpoint implementation or scikit-learn version changed; safe resume is unavailable')
        if existing.get('status') == 'completed':
            raise ValueError('This investigation is already completed')
    elif output.exists() and any(p.name not in ('development_snapshot.pkl', 'development_snapshot.sha256', 'investigation.lock') for p in output.iterdir()):
        raise ValueError('Output directory must be new or empty (except a new development snapshot); previous results are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    all_rows = _load_committed_records(output, 'candidate_') if resume else []
    moves, selected_sets = [], []
    outer_rows = _load_committed_records(output, 'outer_') if resume else []
    if resume:
        # Rebuild human-readable exports from atomic commits; a forced shutdown
        # may have torn the trailing line of a CSV append.
        if all_rows:
            export_frame(all_rows, output / 'subsets.csv')
        if outer_rows:
            export_frame(outer_rows, output / 'outer_validation.csv')
    signatures = dict(development=partition_signature(X.index, y), reserved_holdout=reserved_holdout_signature)
    manifest = dict(schema_version=3, study_id=str(uuid4()),
                    started_at=datetime.now(timezone.utc).isoformat(),
                    status='running', experimental=True, options=asdict(options), target=str(target),
                    investigative_model=identity, provenance=provenance,
                    original_columns=list(X.columns), text_options=text_options or {}, signatures=signatures,
                    development_rows=len(X), development_classes=dict(Counter(y)), grouped_cv=groups is not None,
                    evaluation=dict(type='multiclass' if is_multiclass else 'binary',
                                    ranking='mean fold multiclass MCC' if is_multiclass else 'mean fold MCC',
                                    target=str(target), class_labels=[str(v) for v in labels]),
                    planned_fit_upper_bound=(options.outer_folds if investigation_mode == 'diagnostics_only' else options.planned_fit_upper_bound + 2 * options.outer_folds), attempted_fits=0,
                    checkpoint_fingerprint=fingerprint, investigation_mode=investigation_mode,
                    subgroup_column=subgroup_column)
    if groups is not None:
        manifest['signatures']['development_groups'] = hashlib.sha256(
            pd.util.hash_pandas_object(groups, index=True).to_numpy(dtype='<u8').tobytes()).hexdigest()
    if resume:
        manifest = existing
        manifest['status'] = 'running'
        manifest.pop('error', None)
        manifest['resumed_at'] = datetime.now(timezone.utc).isoformat()
        investigation_mode = existing.get('investigation_mode', 'selection_plus_diagnostics')
    _atomic_json(summary_path, manifest)
    def sink(row):
        # Atomic candidate commit is authoritative; CSV is an expendable export.
        _commit_record(output, f"candidate_{row['scope']}_{row['candidate']:04d}", row)
        all_rows.append(row)
        export_frame([row], output / 'subsets.csv', append=True)
    event_callback = on_event
    run_started = perf_counter()
    def emit(event):
        nonlocal event_callback
        if event['event'] == 'fit_started':
            manifest['attempted_fits'] += 1
        if event_callback:
            try:
                event_callback(dict(schema_version=1, study_id=manifest['study_id'],
                                    attempted_fits=manifest['attempted_fits'],
                                    completed_candidates=len(all_rows),
                                    elapsed_seconds=perf_counter() - run_started,
                                    planned_fit_upper_bound=manifest['planned_fit_upper_bound'], **event))
            except Exception as ex:
                manifest['progress_reporting_error'] = type(ex).__name__
                event_callback = None
                warnings.warn('Feature-search progress callback failed; investigation continues.', RuntimeWarning)
    try:
        if investigation_mode == 'diagnostics_only':
            manifest['diagnostics'] = {'full_baseline': evaluate_probability_diagnostics(
                X, y, target, options, text_options, groups, pause_event=pause_event, on_event=emit,
                subgroup_column=subgroup_column)}
            if subgroup_column:
                export_frame(manifest['diagnostics']['full_baseline']['subgroups'],
                             output / 'subgroup_diagnostics_full_baseline.csv')
            manifest['status'] = 'completed'
            manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
            _atomic_json(summary_path, manifest)
            return manifest
        emit(dict(event='study_resumed' if resume else 'study_started', model_type=options.model_type))
        for fold, (train, test) in enumerate(outer, 1):
            search = Search(X.iloc[train], y.iloc[train], target, options, text_options,
                f'outer_{fold}', sink, groups.iloc[train] if groups is not None else None, progress, emit,
                previous=[r for r in all_rows if r['scope'] == f'outer_{fold}'], pause_event=pause_event, checkpoint_directory=output)
            best = search.run()
            moves.extend(search.moves)
            selected_sets.append(best['columns'])
            for name, subset in [('selected', best['columns']), ('full_baseline', tuple(X.columns))]:
                if any(r['fold'] == fold and r['model'] == name for r in outer_rows):
                    continue
                if pause_event is not None and pause_event.is_set():
                    raise InvestigationPaused('Paused at outer-validation checkpoint')
                model = make_pipeline(subset, options, text_options)
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter('always', ConvergenceWarning)
                    emit(dict(event='fit_started', phase='outer', scope=f'outer_{fold}',
                              model_type=options.model_type, model=name, n_columns=len(subset)))
                    model.fit(X.iloc[train], y.iloc[train])
                predicted = investigative_predictions(model, X.iloc[test], target, options)
                emit(dict(event='fit_completed', phase='outer', scope=f'outer_{fold}',
                          model_type=options.model_type, model=name, n_columns=len(subset)))
                outer_rows.append(dict(fold=fold, model=name, model_type=options.model_type, columns=subset,
                    mcc=float(matthews_corrcoef(y.iloc[test], predicted)),
                    convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in recorded),
                    validation_signature=partition_signature(X.iloc[test].index, y.iloc[test]),
                    **metrics(classification_counts(y.iloc[test], predicted, target)),
                    **(multiclass_diagnostics(y.iloc[test], predicted, labels) if is_multiclass else {})))
                _commit_record(output, f'outer_{fold}_{name}', outer_rows[-1])
                export_frame(outer_rows, output / 'outer_validation.csv')
            export_frame(outer_rows, output / 'outer_validation.csv')
            export_frame(moves, output / 'refinement.csv', columns=['scope', 'seed_columns', 'round', 'operation', 'feature', 'before_columns', 'after_columns', 'before_mcc', 'after_mcc', 'gain'])
            emit(dict(event='outer_validation_completed', scope=f'outer_{fold}',
                      model_type=options.model_type, fold=fold, folds=len(outer)))
        search = Search(X, y, target, options, text_options, 'development', sink, groups, progress, emit,
                        previous=[r for r in all_rows if r['scope'] == 'development'], pause_event=pause_event, checkpoint_directory=output)
        best = search.run()
        moves.extend(search.moves)
        export_frame(moves, output / 'refinement.csv', columns=['scope', 'seed_columns', 'round', 'operation', 'feature', 'before_columns', 'after_columns', 'before_mcc', 'after_mcc', 'gain'])
        summaries = {}
        for name in ('selected', 'full_baseline'):
            rows = [r for r in outer_rows if r['model'] == name]
            counts = {key: sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn', 'tn')}
            summaries[name] = dict(mean_mcc=float(np.mean([r['mcc'] for r in rows])),
                std_mcc=float(np.std([r['mcc'] for r in rows])), **metrics(counts),
                **(diagnostics_from_matrix(add_matrices([r['confusion_matrix'] for r in rows]), labels)
                   if is_multiclass else {}))
        paired = [next(r['mcc'] for r in outer_rows if r['fold'] == f and r['model'] == 'selected') -
                  next(r['mcc'] for r in outer_rows if r['fold'] == f and r['model'] == 'full_baseline')
                  for f in range(1, options.outer_folds + 1)]
        stability = [dict(column=c, selected_outer_folds=sum(c in s for s in selected_sets),
                          selection_fraction=sum(c in s for s in selected_sets) / len(selected_sets)) for c in X.columns]
        export_frame(stability, output / 'selection_stability.csv')
        comparisons = []
        for size in sorted({r['n_columns'] for r in search.cache.values()}):
            rows = sorted((r for r in search.cache.values() if r['n_columns'] == size and not r['error']), key=ranking)
            if len(rows) >= 2:
                good, bad = rows[0], rows[-1]
                comparisons.append(dict(n_columns=size, best_mcc=good['mean_mcc'], worst_mcc=bad['mean_mcc'],
                    best_only=json.dumps([c for c in good['columns'] if c not in bad['columns']], ensure_ascii=False),
                    worst_only=json.dumps([c for c in bad['columns'] if c not in good['columns']], ensure_ascii=False)))
        export_frame(comparisons, output / 'subset_comparisons.csv', columns=['n_columns', 'best_mcc', 'worst_mcc', 'best_only', 'worst_only'])
        recommendation = dict(selected_columns=list(best['columns']),
            model_type=options.model_type, validation_status='candidate',
            excluded_columns=[c for c in X.columns if c not in best['columns']],
            search_mean_mcc=best['mean_mcc'], search_std_mcc=best['std_mcc'],
            score_status='Exploratory search-CV score; not independent validation of this exact subset')
        (output / 'selected_features.json').write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding='utf-8')
        if summaries['selected']['mean_mcc'] < summaries['full_baseline']['mean_mcc']:
            manifest['baseline_warning'] = ('Nested selected-subset procedure underperforms the full-column baseline; '
                                            'do not apply the reduced subset without independent review.')
        diagnostics = {}
        for name, columns in (('selected', recommendation['selected_columns']),
                              ('full_baseline', X.columns)):
            diagnostics[name] = evaluate_probability_diagnostics(
                X, y, target, options, text_options, groups, columns=columns,
                pause_event=pause_event, on_event=emit, subgroup_column=subgroup_column)
        manifest['diagnostics'] = diagnostics
        if subgroup_column:
            for report_name, report in diagnostics.items():
                export_frame(report['subgroups'], output / f'subgroup_diagnostics_{report_name}.csv')
        manifest.update(status='completed', evaluated_subsets=len(all_rows),
            nested_validation=summaries, paired_outer_mcc_gains=paired, recommendation=recommendation,
            interpretation='Outer CV evaluates the entire selection procedure, not the exact final development subset. No external holdout evaluation or application configuration/model update.',
            limitations='Best/worst contrasts and selection frequency are model-dependent clues, not intrinsic feature importance. Search scores are optimistic. Group CV measures new-group generalization only when the supplied groups represent it.')
    except BaseException as ex:
        manifest.update(status='paused' if isinstance(ex, InvestigationPaused) else ('interrupted' if isinstance(ex, KeyboardInterrupt) else 'failed'),
                        error=f'{type(ex).__name__}: {ex}', evaluated_subsets=len(all_rows))
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        _atomic_json(summary_path, manifest)
        # Reporting must not replace the original failure/interruption.
        try:
            emit(dict(event='study_' + manifest['status'], model_type=options.model_type,
                      error=manifest['error']))
        except BaseException:
            pass
        raise
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    emit(dict(event='study_completed', model_type=options.model_type))
    _atomic_json(summary_path, manifest)
    return manifest
