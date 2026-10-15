#!/usr/bin/env python
"""Experimental original-column search using JBG settings; run from the repo root."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import getpass
import json
import os
import runpy
from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--last-run', nargs='?', const='.jbg_last_run.json', help='JBG Repeat Last JSON; default source')
    source.add_argument('--config', help='Existing JBG Python configuration file')
    source.add_argument('--demo', action='store_true', help='Synthetic numeric smoke test; no SQL connection')
    p.add_argument('--csv', help='Raw CSV instead of SQL, using the selected JBG configuration schema')
    p.add_argument('--sep', default=';', help='CSV separator (default ;)')
    p.add_argument('--decimal', default=',', help='CSV decimal separator (default ,)')
    p.add_argument('--target', help='Target label; default configured label, then least frequent class')
    p.add_argument('--exclude', nargs='*', default=[], help='Original columns to exclude before searching')
    p.add_argument('--group-column', help='Original column defining groups that must stay together in CV; excluded from features')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--sizes', type=int, nargs='+', default=[5, 10, 20, 40, 60])
    p.add_argument('--inner-folds', type=int, default=5)
    p.add_argument('--outer-folds', type=int, default=3)
    p.add_argument('--top-sets', type=int, default=2)
    p.add_argument('--refine-rounds', type=int, default=2)
    p.add_argument('--max-evaluations', type=int, default=300, help='Distinct subset cap per search, including baseline/refinement')
    p.add_argument('--min-gain', type=float, default=0.005, help='Minimum mean MCC gain to continue greedy refinement')
    p.add_argument('--seed', type=int, default=42, help='Search/nested-CV seed; reserved JBG holdout always uses 42')
    p.add_argument('--scaler', choices=['STA', 'MAX'], default='STA')
    p.add_argument('--oversampling', choices=['NOG', 'RND', 'SME'], default='NOG',
                   help='JBG oversampling: NOG=none, RND=random, SME=SMOTE; training folds only')
    p.add_argument('--reduction', choices=['NOR', 'PCA'], default='NOR')
    p.add_argument('--model-type', choices=['Linear', 'Non-linear', 'Tree', 'Combination'], default='Linear',
                   help='Fixed investigative model, independent of the normal classifier')
    p.add_argument('--C', type=float, default=1., help='Linear inverse regularization strength')
    p.add_argument('--max-iterations', '--linear-max-iterations', type=int, default=2000,
                   help='Linear iteration cap; legacy --max-iterations retains its meaning')
    p.add_argument('--non-linear-max-iterations', type=int, default=2000)
    p.add_argument('--non-linear-alpha', type=float, default=0.0001)
    p.add_argument('--non-linear-hidden-layers', type=int, nargs='+', default=[100])
    p.add_argument('--non-linear-learning-rate', type=float, default=0.001)
    p.add_argument('--tree-estimator', choices=['Extra Trees', 'Random Forest'], default='Extra Trees')
    p.add_argument('--tree-n-estimators', type=int, default=100)
    p.add_argument('--tree-max-depth', type=int, help='Omit for unrestricted depth')
    p.add_argument('--tree-min-samples-leaf', type=int, default=1)
    p.add_argument('--output', help='New/empty output directory; default timestamped feature-search folder')
    return p


def load_config(args):
    from Config import Config
    if args.config:
        namespace = runpy.run_path(str(Path(args.config).resolve()))
        if namespace.get('version', '1.0') == '1.0':
            raise ValueError('V1.0 config is deprecated; use a current JBG configuration')
        config = Config.load_config(SimpleNamespace(**namespace))
    else:
        from GUIHandler import GUIHandler
        path = Path(args.last_run or '.jbg_last_run.json')
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        trusted = snapshot['connection']['trusted_connection']
        user = os.environ.get('JBG_SQL_USERNAME', '')
        password = os.environ.get(Config.SQL_PASSWORD_ENV, '')
        if not trusted and not args.csv:
            user = user or input('SQL username: ').strip()
            password = password or getpass.getpass('SQL password: ')
        config = Config(**GUIHandler._last_run_snapshot_to_config_params(snapshot, user, password))
    if not config.connection.trusted_connection and not args.csv:
        config.connection.sql_username = config.connection.sql_username or input('SQL username: ').strip()
        config.connection.sql_password = config.connection.sql_password or getpass.getpass('SQL password: ')
    # The experiment only reads labelled rows. No task runner, model writes or notifications.
    config.mode.train = True
    config.mode.predict = False
    config.mode.mispredicted = False
    config.mode.calculate_dark_numbers = False
    config.mode.calculate_feature_importance = False
    config.save = False
    return config


def load_development(args):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from JBGCVReporting import partition_signature
    if args.demo:
        if args.csv:
            raise ValueError('--demo cannot be combined with --csv')
        values, labels = make_classification(n_samples=320, n_features=10, n_informative=3,
            n_redundant=2, weights=[0.85, 0.15], flip_y=0, random_state=42)
        X = pd.DataFrame(values, columns=[f'feature_{i+1}' for i in range(10)], index=np.arange(1, 321))
        y = pd.Series(np.where(labels == 1, 'Inkorrekt', 'Korrekt'), index=X.index)
        train, reserved = train_test_split(np.arange(len(X)), stratify=y, test_size=0.2, random_state=42)
        args.study_context = dict(project='demo', source_kind='synthetic')
        return X.iloc[train], y.iloc[train], args.target or min(Counter(y.iloc[train]), key=lambda value: (Counter(y.iloc[train])[value], str(value))), {}, partition_signature(X.iloc[reserved].index, y.iloc[reserved])
    config = load_config(args)
    args.study_context = dict(project=config.name, source_kind='csv' if args.csv else 'sql',
        source_tables=[f'{config.connection.data_catalog}.{config.connection.data_table}'],
        class_column=config.get_class_column_name(), id_column=config.get_id_column_name())
    from JBGHandler import JBGHandler
    from JBGLogger import JBGLogger
    from SQLDataLayer import DataLayer
    logger = JBGLogger(quiet=True, in_terminal=True)
    if args.csv:
        # Preserve code-like text values (e.g. 01) instead of inferring them as integers.
        if len(args.sep) != 1:
            raise ValueError('CSV separator must be one character')
        with Path(args.csv).open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream, delimiter=args.sep), [])
        if len(header) != len(set(header)):
            raise ValueError('Raw CSV has duplicate column names')
        raw = pd.read_csv(args.csv, sep=args.sep, decimal=args.decimal,
                          dtype={c: str for c in config.get_text_column_names()})
        columns = config.get_column_names()
        missing = [c for c in columns if c not in raw.columns]
        if missing:
            raise ValueError(f'Raw CSV is missing configured columns: {missing}')
        if raw.columns.has_duplicates:
            raise ValueError('Raw CSV has duplicate column names')
        # CSV class labels use the same normalization as JBG; unknown rows are excluded.
        label = config.get_class_column_name()
        raw = raw.loc[raw[label].notna() & raw[label].astype(str).str.strip().ne('')]
        rows = raw.loc[:, columns].to_numpy()
        datalayer = None
    else:
        datalayer = DataLayer(config, logger)
        rows = datalayer.get_dataset()
    handler = JBGHandler(datalayer, config, logger)
    dh = handler.add_handler('dataset')
    dh.load_data(rows)
    dh.separate_dataset()
    dh.split_dataset_for_training_and_validation()
    # Only original-column training rows cross the experiment boundary. No globally fitted converter.
    settings = dict(text_columns=tuple(config.get_text_column_names()),
        category_columns=tuple(config.get_categorical_text_column_names()),
        limit_categorize=30, language='sv', stop_words=config.use_stop_words(), df=1.0,
        ngram_range=config.get_ngram_range().ngram_range,
        use_encryption=config.should_hex_encode(), use_categorization=config.use_categorization())
    target = args.target or config.get_dark_number_target()
    if target not in set(dh.Y_train):
        target = min(Counter(dh.Y_train), key=lambda value: (Counter(dh.Y_train)[value], str(value)))
    return dh.X_train, dh.Y_train, target, settings, partition_signature(dh.X_validation.index, dh.Y_validation)


def main(argv=None):
    args = parser().parse_args(argv)
    from JBGFeatureSearch import SearchOptions, run_experiment
    from threadpoolctl import threadpool_limits
    options = SearchOptions(trials=args.trials, sizes=tuple(args.sizes), inner_folds=args.inner_folds,
        outer_folds=args.outer_folds, top_sets=args.top_sets, refine_rounds=args.refine_rounds,
        max_evaluations=args.max_evaluations, min_gain=args.min_gain, seed=args.seed,
        scaler=args.scaler, oversampling=args.oversampling, reduction=args.reduction, C=args.C, max_iterations=args.max_iterations, model_type=args.model_type,
        non_linear_max_iterations=args.non_linear_max_iterations, non_linear_alpha=args.non_linear_alpha,
        non_linear_hidden_layers=tuple(args.non_linear_hidden_layers),
        non_linear_learning_rate=args.non_linear_learning_rate,
        tree_estimator=args.tree_estimator, tree_n_estimators=args.tree_n_estimators, tree_max_depth=args.tree_max_depth,
        tree_min_samples_leaf=args.tree_min_samples_leaf)
    try:
        options.validate()
        X, y, target, text_options, reserved_signature = load_development(args)
        missing = set(args.exclude) - set(X.columns)
        if missing:
            raise ValueError(f'Unknown excluded columns: {sorted(missing)}')
        groups = None
        excluded = list(dict.fromkeys(args.exclude))
        if args.group_column:
            if args.group_column not in X:
                raise ValueError('group-column must be an available original feature column')
            groups = X[args.group_column].copy()
            if args.group_column not in excluded:
                excluded.append(args.group_column)
        X = X.drop(columns=excluded)
        context = dict(getattr(args, 'study_context', {}), excluded_columns=excluded,
                       group_column=args.group_column or '')
        output = Path(args.output or ROOT / 'output' / ('feature_search_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        print(f'Experimental {options.model_type} feature search: {len(X)} training rows, '
              f'{X.shape[1]} original columns, classes={dict(Counter(y))}, target={target!r}.')
        print('JBG holdout reserved and unused. No model, settings, database or Repeat Last updates.')
        from JBGFeatureSearch import model_identity
        print('Investigative model: ' + json.dumps(model_identity(options), ensure_ascii=False))
        print(f'Up to {options.planned_fit_upper_bound} fits; serial execution, one numerical thread. Results: {output}')
        def on_event(event):
            if event['event'] == 'fit_started':
                phase = (f'candidate {event["candidate"]}, fold {event["fold"]}/{event["folds"]}'
                         if event['phase'] == 'inner' else event['model'] + ' outer validation')
                message = f'{event["scope"]}: {phase}; attempted fits {event["attempted_fits"]}/{event["planned_fit_upper_bound"]}'
            elif event['event'] == 'candidate_completed':
                score = 'FAILED' if event['error'] else f'MCC={event["mean_mcc"]:.4f}'
                message = f'{event["scope"]}: candidate {event["candidate"]}/{event["max_evaluations"]}, {event["n_columns"]} columns, {score}'
            else:
                return
            print('\r' + message.ljust(140), end='', flush=True)
        with threadpool_limits(limits=1):
            result = run_experiment(X, y, target, output, options, text_options, groups,
                reserved_holdout_signature=reserved_signature, progress=None,
                on_event=on_event, study_context=context)
        print('\nCompleted. Nested validation:')
        for name, values in result['nested_validation'].items():
            print(f'{name}: MCC={values["mean_mcc"]:.4f}, precision={values["precision"]:.3f}, '
                  f'recall={values["recall"]:.3f}, F1={values["f1"]:.3f}')
        print('Recommended original columns: ' + ', '.join(result['recommendation']['selected_columns']))
        print('The final subset search score is exploratory; compare outer-fold performance and stability.')
        return 0
    except KeyboardInterrupt:
        print('\nInterrupted; completed candidate rows remain in the output directory.', file=sys.stderr)
        return 130
    except Exception as ex:
        print(f'\nFeature search failed: {type(ex).__name__}: {ex}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
