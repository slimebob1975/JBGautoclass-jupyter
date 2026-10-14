"""Experimental original-column wrapper search; no application state is written."""
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
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
    C: float = 1.0
    max_iterations: int = 2000
    model_type: str = 'Linear'
    non_linear_max_iterations: int = 2000
    non_linear_alpha: float = 0.0001
    non_linear_hidden_layers: tuple = (100,)
    non_linear_learning_rate: float = 0.001
    tree_n_estimators: int = 100
    tree_max_depth: int | None = None
    tree_min_samples_leaf: int = 1

    @property
    def planned_fit_upper_bound(self):
        return (self.outer_folds + 1) * self.max_evaluations * self.inner_folds + 2 * self.outer_folds

    def validate(self):
        for name in ('trials', 'inner_folds', 'outer_folds', 'top_sets', 'max_evaluations',
                     'max_iterations', 'non_linear_max_iterations', 'tree_n_estimators',
                     'tree_min_samples_leaf'):
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
        if self.scaler not in ('STA', 'MAX') or not finite_number(self.C) or self.C <= 0:
            raise ValueError('Use scaler STA/MAX and positive finite C')
        if not finite_number(self.min_gain) or self.min_gain < 0:
            raise ValueError('min_gain must be finite and nonnegative')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if self.model_type not in ('Linear', 'Non-linear', 'Tree'):
            raise ValueError('model_type must be Linear, Non-linear or Tree')
        if (not isinstance(self.non_linear_hidden_layers, tuple) or not self.non_linear_hidden_layers
                or any(type(n) is not int or n < 1 for n in self.non_linear_hidden_layers)):
            raise ValueError('non_linear_hidden_layers must be a tuple of positive integers')
        if not finite_number(self.non_linear_alpha) or self.non_linear_alpha < 0:
            raise ValueError('non_linear_alpha must be finite and nonnegative')
        if not finite_number(self.non_linear_learning_rate) or self.non_linear_learning_rate <= 0:
            raise ValueError('non_linear_learning_rate must be finite and positive')
        if self.tree_max_depth is not None and (type(self.tree_max_depth) is not int or self.tree_max_depth < 1):
            raise ValueError('tree_max_depth must be None or a positive integer')


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
    return 'extra_trees', ExtraTreesClassifier(n_estimators=options.tree_n_estimators,
        max_depth=options.tree_max_depth, min_samples_leaf=options.tree_min_samples_leaf,
        criterion='gini', max_features='sqrt', class_weight=None,
        random_state=options.seed, n_jobs=1, warm_start=False)


def model_identity(options):
    _, estimator = make_estimator(options)
    return dict(model_type=options.model_type,
                estimator=f'{type(estimator).__module__}.{type(estimator).__name__}',
                parameters=estimator.get_params(deep=False))


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
    scaler = MaxAbsScaler() if options.scaler == 'MAX' else StandardScaler(with_mean=False)
    return Pipeline([
        ('columns', FoldColumnEncoder(tuple(columns), text_options)),
        ('impute', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)),
        ('scale', scaler),
        (name, estimator),
    ])


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
    """Round-robin sizes, unique subsets; exhaustively fill small search spaces."""
    columns = tuple(columns)
    sizes = sorted(set(min(n, len(columns)) for n in options.sizes))
    rng = np.random.default_rng(options.seed)
    seen = {columns}
    result = []
    for i in range(max(100, options.trials * 50)):
        size = sizes[i % len(sizes)]
        chosen = set(rng.choice(len(columns), size=size, replace=False).tolist())
        subset = tuple(c for j, c in enumerate(columns) if j in chosen)
        if subset not in seen:
            seen.add(subset)
            result.append(subset)
        if len(result) == options.trials:
            break
    if len(columns) <= 12 and len(result) < options.trials:
        for size in sizes:
            for subset in combinations(columns, size):
                if subset not in seen:
                    seen.add(subset)
                    result.append(subset)
                if len(result) == options.trials:
                    return result
    return result


def ranking(row):
    # MCC controls selection; exact ties prefer fewer original columns, then lower SD.
    return (-row['mean_mcc'], row['n_columns'], row['std_mcc'], row['columns'])


class Search:
    def __init__(self, X, y, target, options, text_options, scope, sink, groups=None, progress=None,
                 on_event=None):
        self.X, self.y, self.target = X, y, target
        self.options, self.text_options = options, text_options
        self.scope, self.sink, self.progress = scope, sink, progress
        self.folds = folds_for(X, y, options.inner_folds, options.seed, groups)
        self.cache, self.moves = {}, []
        self.fit_count = 0
        self.on_event = on_event

    def emit(self, event, **values):
        if self.on_event:
            self.on_event(dict(event=event, scope=self.scope, model_type=self.options.model_type, **values))

    def evaluate(self, columns, stage):
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
                predicted = model.predict(self.X.iloc[test])
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

    def run(self):
        self.emit('search_started', max_evaluations=self.options.max_evaluations)
        columns = tuple(self.X.columns)
        self.evaluate(columns, 'full_baseline')
        for subset in random_subsets(columns, self.options):
            self.evaluate(subset, 'random')
        successful = sorted((r for r in self.cache.values() if not r['error']), key=ranking)
        if not successful:
            raise ValueError(f'{self.scope}: no subset completed all CV folds; inspect subsets.csv')
        for seed_row in successful[:self.options.top_sets]:
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
                new_count = sum(p not in self.cache for p, _, _ in neighbours)
                # Do not commit a greedy step after screening an arbitrary partial neighbourhood.
                if len(self.cache) + new_count > self.options.max_evaluations:
                    break
                evaluated = [(self.evaluate(p, 'refine'), operation, feature) for p, operation, feature in neighbours]
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
        # All evaluated proposals remain candidates, even when the greedy stopping rule rejects a small gain.
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


def run_experiment(X, y, target, output, options=None, text_options=None, groups=None,
                   reserved_holdout_signature=None, progress=print, on_event=None, study_context=None):
    """Nested CV evaluates selection on training data; external holdout never enters.

    A final search on all development rows supplies a recommendation, not an
    independently validated score for that exact final subset.
    """
    options = options or SearchOptions()
    options.validate()
    if not isinstance(X, pd.DataFrame) or X.empty or X.shape[1] == 0 or X.columns.has_duplicates or not all(isinstance(c, str) for c in X.columns):
        raise ValueError('Need a nonempty DataFrame of uniquely named original columns')
    if not isinstance(y, pd.Series) or not X.index.equals(y.index) or X.index.has_duplicates:
        raise ValueError('Original columns and labels need aligned unique row IDs')
    if y.isna().any() or len(set(y)) < 2 or target not in set(y):
        raise ValueError('Investigation requires at least two classes and a target present in y')
    labels = sorted(set(y), key=str)
    is_multiclass = len(labels) > 2
    if groups is not None and not groups.index.equals(X.index):
        raise ValueError('Group labels and original rows must align')
    provenance = study_provenance(X, study_context)
    identity = model_identity(options)
    # Validate every inner partition before starting an expensive study.
    outer = folds_for(X, y, options.outer_folds, options.seed, groups)
    for train, _ in outer:
        folds_for(X.iloc[train], y.iloc[train], options.inner_folds, options.seed,
                  groups.iloc[train] if groups is not None else None)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be new or empty; previous results are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    all_rows, moves, outer_rows, selected_sets = [], [], [], []
    signatures = dict(development=partition_signature(X.index, y), reserved_holdout=reserved_holdout_signature)
    manifest = dict(schema_version=2, study_id=str(uuid4()),
                    started_at=datetime.now(timezone.utc).isoformat(),
                    status='running', experimental=True, options=asdict(options), target=str(target),
                    investigative_model=identity, provenance=provenance,
                    original_columns=list(X.columns), text_options=text_options or {}, signatures=signatures,
                    development_rows=len(X), development_classes=dict(Counter(y)), grouped_cv=groups is not None,
                    evaluation=dict(type='multiclass' if is_multiclass else 'binary',
                                    ranking='mean fold multiclass MCC' if is_multiclass else 'mean fold MCC',
                                    target=str(target), class_labels=[str(v) for v in labels]),
                    planned_fit_upper_bound=options.planned_fit_upper_bound, attempted_fits=0)
    if groups is not None:
        manifest['signatures']['development_groups'] = hashlib.sha256(
            pd.util.hash_pandas_object(groups, index=True).to_numpy(dtype='<u8').tobytes()).hexdigest()
    (output / 'summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    def sink(row):
        all_rows.append(row)
        # Append each completed candidate so an interrupted run retains its evidence.
        export_frame([row], output / 'subsets.csv', append=True)
    event_callback = on_event
    def emit(event):
        nonlocal event_callback
        if event['event'] == 'fit_started':
            manifest['attempted_fits'] += 1
        if event_callback:
            try:
                event_callback(dict(schema_version=1, study_id=manifest['study_id'],
                                    attempted_fits=manifest['attempted_fits'],
                                    planned_fit_upper_bound=options.planned_fit_upper_bound, **event))
            except Exception as ex:
                manifest['progress_reporting_error'] = type(ex).__name__
                event_callback = None
                warnings.warn('Feature-search progress callback failed; investigation continues.', RuntimeWarning)
    try:
        emit(dict(event='study_started', model_type=options.model_type))
        for fold, (train, test) in enumerate(outer, 1):
            search = Search(X.iloc[train], y.iloc[train], target, options, text_options,
                f'outer_{fold}', sink, groups.iloc[train] if groups is not None else None, progress, emit)
            best = search.run()
            moves.extend(search.moves)
            selected_sets.append(best['columns'])
            for name, subset in [('selected', best['columns']), ('full_baseline', tuple(X.columns))]:
                model = make_pipeline(subset, options, text_options)
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter('always', ConvergenceWarning)
                    emit(dict(event='fit_started', phase='outer', scope=f'outer_{fold}',
                              model_type=options.model_type, model=name, n_columns=len(subset)))
                    model.fit(X.iloc[train], y.iloc[train])
                predicted = model.predict(X.iloc[test])
                emit(dict(event='fit_completed', phase='outer', scope=f'outer_{fold}',
                          model_type=options.model_type, model=name, n_columns=len(subset)))
                outer_rows.append(dict(fold=fold, model=name, model_type=options.model_type, columns=subset,
                    mcc=float(matthews_corrcoef(y.iloc[test], predicted)),
                    convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in recorded),
                    validation_signature=partition_signature(X.iloc[test].index, y.iloc[test]),
                    **metrics(classification_counts(y.iloc[test], predicted, target)),
                    **(multiclass_diagnostics(y.iloc[test], predicted, labels) if is_multiclass else {})))
            export_frame(outer_rows, output / 'outer_validation.csv')
            export_frame(moves, output / 'refinement.csv', columns=['scope', 'seed_columns', 'round', 'operation', 'feature', 'before_columns', 'after_columns', 'before_mcc', 'after_mcc', 'gain'])
            emit(dict(event='outer_validation_completed', scope=f'outer_{fold}',
                      model_type=options.model_type, fold=fold, folds=len(outer)))
        search = Search(X, y, target, options, text_options, 'development', sink, groups, progress, emit)
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
        manifest.update(status='completed', evaluated_subsets=len(all_rows),
            nested_validation=summaries, paired_outer_mcc_gains=paired, recommendation=recommendation,
            interpretation='Outer CV evaluates the entire selection procedure, not the exact final development subset. No external holdout evaluation or application configuration/model update.',
            limitations='Best/worst contrasts and selection frequency are model-dependent clues, not intrinsic feature importance. Search scores are optimistic. Group CV measures new-group generalization only when the supplied groups represent it.')
    except BaseException as ex:
        manifest.update(status='interrupted' if isinstance(ex, KeyboardInterrupt) else 'failed',
                        error=f'{type(ex).__name__}: {ex}', evaluated_subsets=len(all_rows))
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        (output / 'summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        # Reporting must not replace the original failure/interruption.
        try:
            emit(dict(event='study_' + manifest['status'], model_type=options.model_type,
                      error=manifest['error']))
        except BaseException:
            pass
        raise
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    emit(dict(event='study_completed', model_type=options.model_type))
    (output / 'summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest
