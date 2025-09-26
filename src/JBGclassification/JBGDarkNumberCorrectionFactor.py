from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from sklearn.base import BaseEstimator, clone
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
import argparse
from sklearn.neural_network import MLPClassifier
from typing import Union, List

DEBUG = True

class DarkNumberCorrectionFactorEstimator(BaseEstimator):
    def __init__(
        self, 
        estimator, 
        flip_fraction: float = 0.1, 
        n_splits: int = 10, 
        n_repeats: int = 3, 
        n_jobs: int = None,
        predict_mode: str = "predict",
        random_state: int = 42,
        positive_class: int = 1,
        sample_size: float = 0.1
    ):
        self.estimator = estimator
        self.flip_fraction = flip_fraction
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.predict_mode = predict_mode
        self.random_state = random_state
        self.positive_class = positive_class
        self.sample_size = sample_size
        self.correction_factor_ = None

    @staticmethod
    def _compute_min_sample_size(
        y,
        flip_fraction: float,
        n_splits: int,
        positive_class: int,
        min_flips: int = 3,
        min_pos_per_fold: int = 2
    ) ->  int:
        """
        Estimate a safe minimum sample size.
        """

        # Count positives in full data
        total_positives = np.sum(y == positive_class)

        # Estimate how many samples are needed to get enough positives after subsampling
        # Want at least min_flips flipped → solve: pos * flip_frac ≥ min_flips
        min_pos_required = int(np.ceil(min_flips / flip_fraction))

        # Also require that each fold gets min_pos_per_fold positives
        min_pos_for_cv = n_splits * min_pos_per_fold

        # Take max of both requirements
        required_positives = max(min_pos_required, min_pos_for_cv)

        # Estimate total dataset size needed to expect that many positives
        pos_ratio = total_positives / len(y) if len(y) > 0 else 0
        if pos_ratio == 0:
            raise ValueError("No positive examples in data.")

        estimated_sample_size = int(np.ceil(required_positives / pos_ratio))
        return estimated_sample_size

    def _evaluate_split(self, X, y, train_idx, repeat_idx):
        rng = np.random.default_rng(self.random_state + repeat_idx if self.random_state is not None else None)
        X_train, y_train = X[train_idx], y[train_idx]
        y_train_flipped = y_train.copy()

        pos_indices = np.where(y_train == self.positive_class)[0]
        n_flip = int(len(pos_indices) * self.flip_fraction)
        if n_flip == 0:
            return 0.0

        flip_indices = rng.choice(pos_indices, size=n_flip, replace=False)

        other_class = [c for c in np.unique(y_train) if c != self.positive_class]
        if not other_class:
            raise ValueError("Cannot find a negative class different from the positive_class.")
        y_train_flipped[flip_indices] = other_class[0]

        model = clone(self.estimator)
        model.fit(X_train, y_train_flipped)

        X_flipped = X_train[flip_indices]
        if self.predict_mode == 'predict':
            y_pred = model.predict(X_flipped)
            reidentified = np.sum(y_pred == self.positive_class)
        elif self.predict_mode == 'predict_proba':
            class_index = list(model.classes_).index(self.positive_class)
            y_proba = model.predict_proba(X_flipped)[:, class_index]
            reidentified = np.sum(y_proba)
        else:
            raise ValueError("predict_mode must be 'predict' or 'predict_proba'")

        return reidentified / n_flip

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

         # Set the smallest possible samples size
        min_sample_needed = self._compute_min_sample_size(
            y,
            flip_fraction=self.flip_fraction,
            n_splits=self.n_splits,
            positive_class=self.positive_class
        )

        n_samples = X.shape[0]

        effective_size = min(n_samples, int(n_samples * self.sample_size))
        if effective_size < min_sample_needed:
            if DEBUG:
                print(f"[Warning] Sample size: {self.sample_size} gives effetive size: {effective_size}, which is less than required: {min_sample_needed}! Skipping.")
            self.correction_factor_ = 1.0
            return self

        if DEBUG:
            print(f"Sample size: {self.sample_size}, minimum sample needed: {min_sample_needed}, effective size: {effective_size}")

        if effective_size < n_samples:
            X_sub, _, y_sub, _ = train_test_split(
                X, y,
                train_size=effective_size,
                stratify=y,
                random_state=self.random_state
            )
            if DEBUG:
                print(f"[DEBUG] Subsampled to {len(X_sub)} rows using stratified sampling (sample_size={self.sample_size}).")
        else:
            X_sub, y_sub = X, y

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        if DEBUG:
            print(f"[DEBUG] Running with estimator with n_jobs={self.n_jobs}")

        results = Parallel(
            n_jobs=self.n_jobs,
            prefer='processes',
            max_nbytes=None
        )(
            delayed(self._evaluate_split)(X_sub, y_sub, train_idx, repeat_idx)
            for repeat_idx in range(self.n_repeats)
            for train_idx, _ in skf.split(X_sub, y_sub)
        )

        mean_r = np.mean(results)
        self.correction_factor_ = 1.0 / mean_r if mean_r > 0 else np.inf
        if DEBUG:
            print(f"[DEBUG] Refound for sample_size={self.sample_size}: {results} leading to correction factor: {self.correction_factor_}")

        return self

    def score(self, X=None, y=None):
        return self.correction_factor_

def main():
    parser = argparse.ArgumentParser(description="Test dark number correction factor estimator.")
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_features', type=int, default=20)
    parser.add_argument('--flip_fraction', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--predict_mode', type=str, default='predict', choices=['predict', 'predict_proba'])
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--positive_class', type=int, default=1)
    parser.add_argument('--sample_size', type=float, default='0.2', help="Sample size as float")

    # Parse in arguments
    args = parser.parse_args()
    
    # Parse single float or comma-separated list
    sample_size = float(args.sample_size)
    
    print("[INFO] Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=8,
        n_redundant=5,
        n_classes=2,
        weights=[0.05, 0.95],
        random_state=args.random_state
    )

    clf = MLPClassifier(random_state=args.random_state, max_iter=2000, early_stopping=True)

    estimator = DarkNumberCorrectionFactorEstimator(
        estimator=clf,
        flip_fraction=args.flip_fraction,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        n_jobs=args.n_jobs,
        predict_mode=args.predict_mode,
        random_state=args.random_state,
        positive_class=args.positive_class,
        sample_size=sample_size
    )

    print(f"[INFO] Estimating correction factor for class {args.positive_class}...")
    estimator.fit(X, y)
    print(f"[RESULT] Estimated correction factor for sample size {args.sample_size}: {estimator.score():.4f}")

if __name__ == "__main__":
    main()
