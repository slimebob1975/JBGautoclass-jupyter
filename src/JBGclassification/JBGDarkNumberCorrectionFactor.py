import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from joblib import Parallel, delayed
from pickle import PicklingError

DEBUG = True
BACKEND_PROCESSES = "processes"
BACKEND_THREADS = "threads"


def evaluate_split(estimator, X, y, train_idx, repeat_idx,
                   flip_fraction, positive_class, predict_mode, random_state):
    """
    Top-level evaluation function for Parallel.
    """
    rng = np.random.default_rng(random_state + repeat_idx if random_state is not None else None)
    X_train, y_train = X[train_idx], y[train_idx]
    y_train_flipped = y_train.copy()

    pos_indices = np.where(y_train == positive_class)[0]
    n_flip = int(len(pos_indices) * flip_fraction)
    if n_flip == 0:
        return 0.0

    flip_indices = rng.choice(pos_indices, size=n_flip, replace=False)

    other_class = [c for c in np.unique(y_train) if c != positive_class]
    if not other_class:
        raise ValueError("Cannot find a negative class different from the positive_class.")
    y_train_flipped[flip_indices] = other_class[0]

    model = clone(estimator)
    model.fit(X_train, y_train_flipped)

    X_flipped = X_train[flip_indices]
    if predict_mode == 'predict':
        y_pred = model.predict(X_flipped)
        reidentified = np.sum(y_pred == positive_class)
    elif predict_mode == 'predict_proba':
        class_index = list(model.classes_).index(positive_class)
        y_proba = model.predict_proba(X_flipped)[:, class_index]
        reidentified = np.sum(y_proba)
    else:
        raise ValueError("predict_mode must be 'predict' or 'predict_proba'")

    return reidentified / n_flip


class DarkNumberCorrectionFactorEstimator(BaseEstimator):
    def __init__(self,
                 estimator,
                 flip_fraction: float = 0.1,
                 n_splits: int = 10,
                 n_repeats: int = 3,
                 n_jobs: int = None,
                 predict_mode: str = "predict",
                 random_state: int = 42,
                 positive_class: int = 1,
                 sample_size: float = 0.1,
                 parallel_backend: str = BACKEND_THREADS):   # default threads
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
        self._parallel_backend = parallel_backend

    @staticmethod
    def _compute_min_sample_size(y,
                                 flip_fraction: float,
                                 n_splits: int,
                                 positive_class: int,
                                 min_flips: int = 3,
                                 min_pos_per_fold: int = 2) -> int:
        total_positives = np.sum(y == positive_class)
        min_pos_required = int(np.ceil(min_flips / flip_fraction))
        min_pos_for_cv = n_splits * min_pos_per_fold
        required_positives = max(min_pos_required, min_pos_for_cv)

        pos_ratio = total_positives / len(y) if len(y) > 0 else 0
        if pos_ratio == 0:
            raise ValueError("No positive examples in data.")

        return int(np.ceil(required_positives / pos_ratio))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Check minimum sample requirement
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
                print(f"[Warning] Sample size {self.sample_size} gives effective size {effective_size}, "
                      f"less than required {min_sample_needed}. Skipping.")
            self.correction_factor_ = 1.0
            return self

        if effective_size < n_samples:
            X_sub, _, y_sub, _ = train_test_split(
                X, y,
                train_size=effective_size,
                stratify=y,
                random_state=self.random_state
            )
            if DEBUG:
                print(f"[DEBUG] Subsampled to {len(X_sub)} rows (sample_size={self.sample_size}).")
        else:
            X_sub, y_sub = X, y

        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True,
                              random_state=self.random_state)

        # Build tasks once
        tasks = [
            delayed(evaluate_split)(
                self.estimator, X_sub, y_sub, train_idx, repeat_idx,
                self.flip_fraction, self.positive_class,
                self.predict_mode, self.random_state
            )
            for repeat_idx in range(self.n_repeats)
            for train_idx, _ in skf.split(X_sub, y_sub)
        ]

        # Try running tasks with retry logic
        results = self._run_parallel(tasks)

        mean_r = np.mean(results)
        self.correction_factor_ = 1.0 / mean_r if mean_r > 0 else np.inf
        if DEBUG:
            print(f"[DEBUG] Correction factor: {self.correction_factor_} from results {results}")

        return self

    def _run_parallel(self, tasks):
        """
        Run tasks with retry logic, reusing the same task list.
        """
        while True:
            try:
                if DEBUG:
                    print(f"[DEBUG] Running Parallel with backend={self._parallel_backend}, n_jobs={self.n_jobs}")
                return Parallel(
                    n_jobs=self.n_jobs,
                    prefer=self._parallel_backend,
                    pre_dispatch="2*n_jobs",
                    batch_size="auto",
                    max_nbytes=None
                )(tasks)

            except (SystemError, MemoryError) as e:
                if DEBUG:
                    print(f"[DEBUG] Parallel failed with {type(e).__name__}: {e}.")
                if self.n_jobs and self.n_jobs > 1:
                    self.n_jobs = max(1, self.n_jobs // 2)
                    if DEBUG:
                        print(f"[DEBUG] Retrying with n_jobs={self.n_jobs}")
                else:
                    raise

            except (PicklingError, AttributeError, TypeError) as e:
                if DEBUG:
                    print(f"[WARNING] Pickling/backend error: {e}")
                if self._parallel_backend == BACKEND_PROCESSES:
                    self._parallel_backend = BACKEND_THREADS
                    if DEBUG:
                        print(f"[DEBUG] Switching backend to threads and retrying")
                else:
                    raise

    def score(self, X=None, y=None):
        return self.correction_factor_
