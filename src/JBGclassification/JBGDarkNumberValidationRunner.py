"""Run paired Dark Number fallback validation on the most recent real dataset.

This module is deliberately outside the normal training/prediction flow. It reloads
only the last-run dataset settings and saved model pipeline, then compares the normal
hard/direct baseline against the validation-only ``perturbed_same_model`` candidate
using paired seeds. It never changes production Dark Number behavior or persisted model
artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from Config import Config, DarkNumberAlpha, DarkNumberMethod
from JBGDarkNumberValidation import DarkNumberValidationHarness
from JBGHandler import JBGHandler
from Helpers import prepare_estimator_input
from JBGMeta import (
    AlgorithmTuple,
    NgramRange,
    Oversampling,
    PreprocessTuple,
    ReductionTuple,
    ScoreMetric,
    Undersampling,
)
from JBGModelPersistence import load_model_artifact
from JBGStreamedLogger import JBGLogger
from SQLDataLayer import DataLayer


LAST_RUN_STATE_VERSION = 1
LAST_RUN_STATE_FILENAME = ".jbg_last_run.json"


def load_last_run_snapshot(path: Path) -> dict[str, Any]:
    """Load and minimally validate the persisted Repeat Last state."""
    with path.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if snapshot.get("version") != LAST_RUN_STATE_VERSION:
        raise ValueError(
            f"Unsupported last-run state version {snapshot.get('version')!r}; "
            f"expected {LAST_RUN_STATE_VERSION}."
        )
    for key in ("connection", "mode", "io", "debug", "mail", "name"):
        if key not in snapshot:
            raise ValueError(f"Last-run state is missing required section {key!r}.")
    return snapshot


def resolve_sql_credentials(snapshot: dict[str, Any], sql_username: str | None) -> tuple[str, str]:
    """Resolve runtime-only SQL credentials without persisting them."""
    connection = snapshot["connection"]
    if bool(connection.get("trusted_connection")):
        return "", ""

    username = (sql_username or os.environ.get("JBG_SQL_USERNAME", "")).strip()
    if not username:
        username = input("SQL username: ").strip()
    if not username:
        raise ValueError("SQL username is required for a non-trusted connection.")

    password = os.environ.get(Config.SQL_PASSWORD_ENV, "")
    if not password:
        password = getpass.getpass("SQL password: ")
    if not password:
        raise ValueError(
            f"SQL password is required. Set {Config.SQL_PASSWORD_ENV} or enter it at the prompt."
        )
    return username, password


def config_from_last_run_snapshot(
    snapshot: dict[str, Any],
    sql_username: str,
    sql_password: str,
) -> Config:
    """Recreate the runtime Config represented by the last manual/repeated run."""
    connection = dict(snapshot["connection"])
    mode = snapshot["mode"]

    return Config(
        connection=Config.Connection(
            **connection,
            sql_username=sql_username,
            sql_password=sql_password,
        ),
        mode=Config.Mode(
            train=bool(mode["train"]),
            predict=bool(mode["predict"]),
            mispredicted=bool(mode["mispredicted"]),
            use_metas=bool(mode["use_metas"]),
            use_stop_words=bool(mode["use_stop_words"]),
            ngram_range=NgramRange[mode["ngram_range"]],
            hex_encode=bool(mode["hex_encode"]),
            use_categorization=bool(mode["use_categorization"]),
            category_text_columns=list(mode["category_text_columns"]),
            test_size=float(mode["test_size"]),
            calculate_dark_numbers=bool(mode.get("calculate_dark_numbers", mode["mispredicted"])),
            dark_number_method=DarkNumberMethod.from_config_value(
                mode.get("dark_number_method", "LINEAR")
            ),
            dark_number_alpha=DarkNumberAlpha.from_config_value(
                mode.get("dark_number_alpha", "NONE")
            ),
            dark_number_flip_fraction=float(mode.get("dark_number_flip_fraction", 0.2)),
            experimental_perturbed_dark_number_fallback=bool(
                mode.get("experimental_perturbed_dark_number_fallback", True)
            ),
            oversampler=Oversampling[mode["oversampler"]],
            undersampler=Undersampling[mode["undersampler"]],
            algorithm=AlgorithmTuple(mode["algorithm"]),
            preprocessor=PreprocessTuple(mode["preprocessor"]),
            feature_selection=ReductionTuple(mode["feature_selection"]),
            num_selected_features=mode["num_selected_features"],
            scoring=ScoreMetric[mode["scoring"]],
            max_iterations=mode["max_iterations"],
        ),
        io=Config.IO(**snapshot["io"]),
        debug=Config.Debug(**snapshot["debug"]),
        mail=Config.Mail(**snapshot["mail"]),
        name=str(snapshot["name"]),
        save=False,
    )


def load_validation_inputs(config: Config, logger: JBGLogger, model_path: Path):
    """Fetch the last-run dataset and prepare the feature matrix used by the saved pipeline."""
    datalayer = DataLayer(config, logger)
    handler = JBGHandler(datalayer=datalayer, config=config, logger=logger)
    dataset_handler = handler.add_handler("dataset")

    logger.print_info(
        f"Validation-only data fetch: {config.connection.data_catalog}.{config.connection.data_table}; "
        f"limit={config.get_data_limit() or 'all'}."
    )
    data = handler.get_dataset()
    if data is None or len(data) == 0:
        raise ValueError("The validation runner did not fetch any rows from the configured dataset.")

    dataset_handler.load_data(data)
    dataset_handler.separate_dataset()
    if dataset_handler.X.empty:
        raise ValueError("The validation runner requires labelled rows; no known-class rows were fetched.")

    _, text_converter, _, pipeline, _, n_features_out = load_model_artifact(model_path)
    if pipeline is None:
        raise ValueError(f"Saved model artifact {model_path} does not contain a pipeline.")
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            "The real-dataset validation harness requires predict_proba() for its enrichment diagnostics. "
            "The perturbed_same_model correction candidate itself uses hard predictions only."
        )

    X = dataset_handler.X
    if text_converter is not None:
        X = text_converter.transform(X)
    # sklearn warns and densifies pandas SparseDtype frames. The production code
    # already normalizes those to SciPy CSR; do the same in validation so larger
    # text datasets do not acquire an avoidable dense-memory copy.
    X = prepare_estimator_input(X)
    y = np.asarray(dataset_handler.Y)

    if n_features_out not in (None, 0) and X.shape[1] != int(n_features_out):
        raise ValueError(
            "Saved text/feature converter produced a different number of features than the model artifact: "
            f"expected {n_features_out}, got {X.shape[1]}."
        )

    logger.print_info(
        f"Validation input prepared: rows={X.shape[0]}, features={X.shape[1]}, "
        f"classes={list(pd.unique(y))}."
    )
    return X, y, pipeline


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return slug or "validation"


def _format_rate(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def log_target_summary(logger: JBGLogger, target: Any, paired: pd.DataFrame, summary: dict) -> None:
    """Emit the compact evidence needed to judge the validation-only candidate."""
    logger.print_info(f"Dark Number perturbed same-model validation target: {target}")
    logger.print_info(
        "Source rates: "
        f"baseline unestimable={_format_rate(summary['baseline_unestimable_rate'])}, "
        f"perturbed activation={_format_rate(summary['perturbed_same_model_activation_rate'])}, "
        f"candidate unestimable={_format_rate(summary['candidate_unestimable_rate'])}."
    )
    logger.print_info(
        f"Estimability coverage delta (perturbed - baseline): {summary['estimability_rate_delta']:+.6f}"
    )
    logger.print_info(
        f"Dark Number interval coverage delta (perturbed - baseline): "
        f"{summary['interval_coverage_rate_delta']:+.6f}"
    )
    logger.print_info(
        f"Known injected-noise truth (mean over paired seeds): "
        f"{pd.to_numeric(paired['true_dark_number'], errors='coerce').mean():.6f}"
    )

    logger.print_info(
        "Robustness metrics exclude fallback/unestimable sentinel rows; "
        "D_cv_full is the primary full-data diagnostic for this experiment."
    )
    for column in ("d_cv_test", "d_cv_full", "d_retrained_full"):
        baseline = summary["baseline_mean_absolute_error"][column]
        candidate = summary["perturbed_same_model_mean_absolute_error"][column]
        delta = summary["mean_absolute_error_delta"][column]
        profile = summary["perturbed_estimable_error_profile"][column]
        logger.print_info(
            f"MAE {column} (legacy paired comparison; baseline may use the 1.0 unestimable sentinel): "
            f"baseline={baseline:.6f}, perturbed_same_model={candidate:.6f}, delta={delta:+.6f}."
        )
        logger.print_info(
            f"Robust {column}: estimable={_format_rate(profile['estimable_rate'])} "
            f"({profile['estimable_count']}/{profile['total_count']}), "
            f"MAE={profile['mean_absolute_error']:.6f}, "
            f"median_abs_error={profile['median_absolute_error']:.6f}, "
            f"p90_abs_error={profile['p90_absolute_error']:.6f}, "
            f"bias={profile['mean_signed_error']:+.6f}, "
            f"signed_error_std={profile['signed_error_std']:.6f}."
        )

    for corr_name in ("corr_cv", "corr_retrained"):
        profile = summary["perturbed_correction_stability"][corr_name]
        logger.print_info(
            f"Perturbed correction stability {corr_name}: "
            f"activation={_format_rate(profile['activation_rate'])} "
            f"({profile['accepted_count']}/{profile['total_count']}), "
            f"median={profile['median']:.6f}, mean={profile['mean']:.6f}, "
            f"std={profile['std']:.6f}, cv={profile['coefficient_of_variation']:.6f}, "
            f"iqr={profile['iqr']:.6f}, range=[{profile['min']:.6f}, {profile['max']:.6f}]."
        )


def run_real_dataset_comparison(
    config: Config,
    logger: JBGLogger,
    X,
    y,
    pipeline,
    *,
    targets: list[str] | None,
    n_runs: int,
    random_state: int,
    injected_noise_fraction: float,
    correction_n_splits: int,
    correction_n_repeats: int,
    perturbation_clones: int = 5,
    perturbation_min_valid: int = 3,
    perturbation_max_cv: float = 0.50,
):
    """Run the paired perturbed-clone comparison for selected or all observed class labels."""
    observed_labels = list(pd.unique(y))
    selected_targets = observed_labels if not targets else targets
    missing = [target for target in selected_targets if target not in observed_labels]
    if missing:
        raise ValueError(
            f"Requested validation target(s) are not present in the fetched labels: {missing}. "
            f"Observed labels: {observed_labels}."
        )

    paired_frames = []
    summaries = {}
    for target in selected_targets:
        logger.print_info(
            f"Running paired validation for target={target!r}: runs={n_runs}, "
            f"seed={random_state}, injected_noise={injected_noise_fraction:.0%}, "
            f"correction_noise={config.get_dark_number_flip_fraction():.0%}."
        )
        harness = DarkNumberValidationHarness(
            estimator=pipeline,
            positive_class=target,
            injected_noise_fraction=injected_noise_fraction,
            correction_flip_fraction=config.get_dark_number_flip_fraction(),
            test_size=config.get_test_size(),
            correction_n_splits=correction_n_splits,
            correction_n_repeats=correction_n_repeats,
            random_state=random_state,
            perturbation_clones=perturbation_clones,
            perturbation_min_valid=perturbation_min_valid,
            perturbation_max_cv=perturbation_max_cv,
            logger=logger,
        )
        paired, summary = harness.compare_perturbed_same_model_fallback(X, y, n_runs=n_runs)
        paired.insert(0, "target", target)
        paired_frames.append(paired)
        summaries[str(target)] = summary
        log_target_summary(logger, target, paired, summary)

    return pd.concat(paired_frames, ignore_index=True), summaries


def save_validation_outputs(config: Config, paired: pd.DataFrame, summaries: dict) -> tuple[Path, Path]:
    output_dir = Path(config.script_path) / "output" / "csvs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stem = f"dark_number_perturbed_same_model_robustness_{_safe_slug(config.io.model_name)}_{timestamp}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}_summary.json"

    paired.to_csv(csv_path, sep=";", index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2, default=str)
    return csv_path, json_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only paired comparison of the normal Dark Number correction path "
            "against the perturbed_same_model candidate on the most recent real dataset."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path.cwd() / LAST_RUN_STATE_FILENAME,
        help="Path to .jbg_last_run.json (default: project-local last-run state).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional saved .sav model artifact; defaults to the model named by the last run.",
    )
    parser.add_argument(
        "--sql-username",
        default=None,
        help="SQL username. If omitted, JBG_SQL_USERNAME or an interactive prompt is used.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Class label to validate. Repeat for several labels; default is every observed label.",
    )
    parser.add_argument("--runs", type=int, default=9, help="Paired random seeds per target (default: 9 for robustness validation).")
    parser.add_argument("--seed", type=int, default=42, help="First paired validation seed (default: 42).")
    parser.add_argument(
        "--injected-noise",
        type=float,
        default=None,
        help="Known hidden-positive fraction for validation; defaults to the configured DN flip fraction.",
    )
    parser.add_argument(
        "--correction-splits",
        type=int,
        default=5,
        help="Correction-factor CV splits inside the validation harness (default: 5).",
    )
    parser.add_argument(
        "--correction-repeats",
        type=int,
        default=2,
        help="Correction-factor CV repeats inside the validation harness (default: 2).",
    )
    parser.add_argument(
        "--perturbation-clones",
        type=int,
        default=5,
        help="Shadow clones per correction attempt (default: 5).",
    )
    parser.add_argument(
        "--perturbation-min-valid",
        type=int,
        default=3,
        help="Minimum valid shadow-clone correction factors required (default: 3).",
    )
    parser.add_argument(
        "--perturbation-max-cv",
        type=float,
        default=0.50,
        help="Maximum coefficient of variation for accepted shadow-clone corrections (default: 0.50).",
    )
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.correction_splits < 2:
        raise ValueError("--correction-splits must be at least 2.")
    if args.correction_repeats < 1:
        raise ValueError("--correction-repeats must be at least 1.")
    if args.perturbation_clones < 1:
        raise ValueError("--perturbation-clones must be at least 1.")
    if args.perturbation_min_valid < 1 or args.perturbation_min_valid > args.perturbation_clones:
        raise ValueError("--perturbation-min-valid must be between 1 and --perturbation-clones.")
    if not np.isfinite(args.perturbation_max_cv) or args.perturbation_max_cv < 0:
        raise ValueError("--perturbation-max-cv must be finite and non-negative.")

    snapshot = load_last_run_snapshot(args.state)
    sql_username, sql_password = resolve_sql_credentials(snapshot, args.sql_username)
    config = config_from_last_run_snapshot(snapshot, sql_username, sql_password)
    model_path = args.model or Path(config.get_model_filename())
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model artifact not found: {model_path}")

    injected_noise = (
        config.get_dark_number_flip_fraction()
        if args.injected_noise is None
        else float(args.injected_noise)
    )
    if not 0.0 < injected_noise < 1.0:
        raise ValueError("--injected-noise must be greater than 0 and less than 1.")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    logger = JBGLogger(
        quiet=False,
        in_terminal=True,
        log_filename=f"jbg-dark-number-validation_{timestamp}_pid{os.getpid()}.log",
    )
    logger.print_info("Dark Number perturbed same-model robustness validation runner (validation only; production unchanged).")
    logger.print_info(f"Validation log: {logger.get_log_filename()}")
    logger.print_info(f"Last-run state: {args.state}")
    logger.print_info(f"Saved model artifact: {model_path}")

    with logger.capture_console_output():
        X, y, pipeline = load_validation_inputs(config, logger, model_path)
        paired, summaries = run_real_dataset_comparison(
            config,
            logger,
            X,
            y,
            pipeline,
            targets=args.target,
            n_runs=args.runs,
            random_state=args.seed,
            injected_noise_fraction=injected_noise,
            correction_n_splits=args.correction_splits,
            correction_n_repeats=args.correction_repeats,
            perturbation_clones=args.perturbation_clones,
            perturbation_min_valid=args.perturbation_min_valid,
            perturbation_max_cv=args.perturbation_max_cv,
        )
        csv_path, json_path = save_validation_outputs(config, paired, summaries)
        logger.print_info(f"Paired validation CSV: {csv_path}")
        logger.print_info(f"Validation summary JSON: {json_path}")
        logger.print_info("Validation completed. No production Dark Number settings or model artifacts were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
