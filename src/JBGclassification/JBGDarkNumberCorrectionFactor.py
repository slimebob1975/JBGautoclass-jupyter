from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from sklearn.base import BaseEstimator, clone
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
import argparse
from sklearn.neural_network import MLPClassifier

DEBUG = False
MIN_SAMPLE_SIZE = 2000

class DarkNumberCorrectionFactorEstimator(BaseEstimator):
    def __init__(
        self, 
        estimator, 
        flip_fraction=0.1, 
        n_splits=10, 
        n_repeats=3, 
        n_jobs=None,
        predict_mode='predict',
        random_state=None,
        positive_class=1,
        sample_size=0.1 
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

        # Apply stratified sampling if data size if over the limit
        n_samples = X.shape[0]
        effective_size = min(n_samples, max(MIN_SAMPLE_SIZE, int(n_samples * self.sample_size)))

        # Only subsample if needed
        if effective_size < n_samples:
            X, _, y, _ = train_test_split(
                X, y,
                train_size=effective_size,
                stratify=y,
                random_state=self.random_state
            )
            if DEBUG:
                print(f"[DEBUG] Subsampled to {len(X)} rows using stratified sampling.")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        results = Parallel(
            n_jobs=self.n_jobs,
            prefer='processes', # Or threads
            max_nbytes=None
        )(
            delayed(self._evaluate_split)(X, y, train_idx, repeat_idx)
            for repeat_idx in range(self.n_repeats)
            for train_idx, _ in skf.split(X, y)
        )

        if DEBUG:
            print("DarkNumberCorrectionFactorEstimator Results = ", results)

        mean_r = np.mean(results)
        self.correction_factor_ = 1.0 / mean_r if mean_r > 0 else np.inf
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
    parser.add_argument('--sample_size', type=float, default=0.2, help="Sample size as float (fraction). Default: 0.2")

    args = parser.parse_args()
    
    print("[INFO] Generating synthetic classification problem...")
    X, y = make_classification(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=args.random_state,
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
        sample_size=args.sample_size
    )

    print(f"[INFO] Estimating correction factor for class {args.positive_class}...")
    estimator.fit(X, y)
    print(f"[RESULT] Estimated correction factor: {estimator.score():.4f}")

if __name__ == "__main__":
    main()