"""Bounded, paired CPU study for the Sigmoid+SGD profile (revision 117).

Run with the application's dependencies and src on PYTHONPATH.
This is a parameter-review experiment, not an automatic regression test.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

from JBGTransformers import NNClassifier3PL


PROFILES = (
    ("deep_sgd_002", 2, "sgd", 0.02),
    ("shallow_sgd_002", 0, "sgd", 0.02),
    ("deep_sgd_01", 2, "sgd", 0.1),
    ("shallow_sgd_01", 0, "sgd", 0.1),
    ("deep_adam_0001", 2, "adam", 0.001),
)


def run_study():
    X, y_numeric = load_breast_cancer(return_X_y=True)
    y = np.where(y_numeric == 0, "M", "B")
    # Reserve 20% before any fitting; this experiment never evaluates that set.
    train_indices, _ = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42,
    )
    X, y = X[train_indices], y[train_indices]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
    rows = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with TemporaryDirectory(prefix="jbg-sigmoid-benchmark-") as checkpoint_root:
            for reduction in ("PCA", "NOR"):
                for name, depth, optimizer, rate in PROFILES:
                    for seed in (42, 43, 44):
                        for fold, (fit_indices, validation_indices) in enumerate(folds, 1):
                            # Pair profiles by outer fold and initialization seed.
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            model = NNClassifier3PL(
                                num_hidden_layers=depth, hidden_layer_size=48,
                                activation="sigmoid", optimizer=optimizer,
                                learning_rate=rate, max_epochs=50, dropout_prob=0.1,
                                train_split=True, verbose=False,
                            )
                            model.device = "cpu"
                            model.OUTPUT_DIR = checkpoint_root
                            steps = [("scale", StandardScaler(with_mean=False))]
                            if reduction == "PCA":
                                steps.append(("reduce", PCA(
                                    n_components=0.95, svd_solver="covariance_eigh",
                                )))
                            pipeline = Pipeline([*steps, ("model", model)])
                            pipeline.fit(X[fit_indices], y[fit_indices])
                            validation_X, validation_y = X[validation_indices], y[validation_indices]
                            probability = pipeline.predict_proba(validation_X)[
                                :, list(model.classes_).index("M")
                            ]
                            predictions = pipeline.predict(validation_X)
                            history = model.net.history
                            row = {
                                "reduction": reduction, "profile": name,
                                "seed": seed, "fold": fold,
                                "auc": float(roc_auc_score(validation_y == "M", probability)),
                                "balanced_accuracy": float(balanced_accuracy_score(
                                    validation_y, predictions,
                                )),
                                "recall_M": float(recall_score(
                                    validation_y, predictions, pos_label="M",
                                )),
                                "restored_epoch": int(history[-1]["epoch"]),
                                "first_valid_loss": float(history[0]["valid_loss"]),
                                "restored_valid_loss": float(history[-1]["valid_loss"]),
                            }
                            rows.append(row)
                            print(json.dumps(row), flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "dataset": "sklearn Breast Cancer", "training_rows": len(y),
        "reserved_rows": len(y_numeric) - len(y), "features": X.shape[1],
        "torch_version": torch.__version__, "device": "cpu", "native_threads": 1,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON results file")
    args = parser.parse_args()
    result = run_study()
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
