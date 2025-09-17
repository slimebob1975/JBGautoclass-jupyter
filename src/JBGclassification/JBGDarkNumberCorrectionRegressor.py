import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, clone
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import make_classification, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import argparse
from JBGDarkNumberCorrectionFactor import DarkNumberCorrectionFactorEstimator

class DarkNumberCorrectionFactorRegressor(BaseEstimator):
    def __init__(
        self,
        estimator,
        sample_size_list=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        flip_fraction=0.1,
        n_splits=5,
        n_repeats=2,
        n_jobs=None,
        predict_mode='predict',
        random_state=None,
        positive_class=1,
        type='poly',  # 'poly', 'linear', 'log'
        poly_degree=2
    ):
        self.estimator = estimator
        self.sample_size_list = sample_size_list
        self.flip_fraction = flip_fraction
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.predict_mode = predict_mode
        self.random_state = random_state
        self.positive_class = positive_class
        self.type = type
        self.poly_degree = poly_degree
        self.correction_factor_ = None
        self.model_ = None
        self.sample_results_ = None

    def fit(self, X, y):
        estimator = DarkNumberCorrectionFactorEstimator(
            estimator=clone(self.estimator),
            flip_fraction=self.flip_fraction,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            n_jobs=self.n_jobs,
            predict_mode=self.predict_mode,
            random_state=self.random_state,
            positive_class=self.positive_class,
            sample_size=self.sample_size_list  # List passed directly
        )
        estimator.fit(X, y)

        sample_results = list(zip(estimator.sample_size_used_, estimator.score()))
        self.sample_results_ = sample_results

        X_samples = np.array([r[0] for r in sample_results]).reshape(-1, 1)
        y_corrs = np.array([r[1] for r in sample_results])

        # Fit regression model
        if self.type == 'poly':
            coeffs = Polynomial.fit(X_samples.flatten(), y_corrs, deg=self.poly_degree).convert().coef
            self.model_ = Polynomial(coeffs)
            self.correction_factor_ = float(self.model_(1.0))

        elif self.type == 'linear':
            model = LinearRegression()
            model.fit(X_samples, y_corrs)
            self.model_ = model
            self.correction_factor_ = float(model.predict([[1.0]]))

        elif self.type == 'log':
            log_transformer = FunctionTransformer(np.log1p, validate=True)
            X_log = log_transformer.transform(X_samples)
            model = LinearRegression()
            model.fit(X_log, y_corrs)
            self.model_ = (model, log_transformer)
            self.correction_factor_ = float(model.predict(log_transformer.transform([[1.0]])))
        
        elif self.type == 'logbounded':

            X_flat = X_samples.flatten()

            # Logarithmic function bounded below by 1.0
            def log_func(x, a, b, c):
                return np.maximum(1.0, a * np.log(b * x + 1) + c)

            try:
                popt, _ = curve_fit(
                    log_func,
                    X_flat,
                    y_corrs,
                    p0=(1.0, 1.0, 0.0),
                    bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
                    maxfev=5000
                )
                self.model_ = (log_func, popt)
                self.correction_factor_ = float(log_func(1.0, *popt))
            except RuntimeError as e:
                print(f"[WARNING] Curve fitting failed: {str(e)}. Falling back to max(y_corrs)")
                self.correction_factor_ = max(max(y_corrs), 1.0)
                self.model_ = None

        else:
            raise ValueError("Unsupported type. Choose from 'poly', 'linear', or 'log'.")

        return self

    def score(self, X=None, y=None):
        return self.correction_factor_

    def get_sample_results(self):
        return self.sample_results_

    def plot(self):
        if self.sample_results_ is None:
            raise RuntimeError("Must call fit() before plotting.")

        X_sample = np.array([r[0] for r in self.sample_results_])
        y_sample = np.array([r[1] for r in self.sample_results_])
        xs = np.linspace(min(X_sample), 1.0, 100)

        if self.type == 'poly':
            ys = self.model_(xs)
        elif self.type == 'linear':
            ys = self.model_.predict(xs.reshape(-1, 1))
        elif self.type == 'log':
            model, transformer = self.model_
            ys = model.predict(transformer.transform(xs.reshape(-1, 1)))
        elif self.type == 'logbounded':
            func, popt = self.model_
            ys = func(xs, *popt)
        else:
            raise ValueError("Unsupported type.")

        plt.figure(figsize=(8, 5))
        plt.plot(X_sample, y_sample, 'o', label='Samples')
        plt.plot(xs, ys, '-', label='Regression fit')
        plt.axvline(1.0, color='red', linestyle='--', label='Extrapolation @ 1.0')
        plt.axhline(self.correction_factor_, color='green', linestyle='--', label='Computed corr(1.0)')
        plt.xlabel('Sample Size Fraction')
        plt.ylabel('Correction Factor')
        plt.title('Correction Factor Regression')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Estimate correction factor using regression extrapolation.")
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'adult'])
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--n_features', type=int, default=20)
    parser.add_argument('--flip_fraction', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_repeats', type=int, default=2)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--predict_mode', type=str, default='predict', choices=['predict', 'predict_proba'])
    parser.add_argument('--positive_class', type=int, default=1)
    parser.add_argument('--type', type=str, default='poly', choices=['poly', 'linear', 'log', 'logbounded'])
    parser.add_argument('--poly_degree', type=int, default=2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument(
        '--sample_size_list',
        type=str,
        default='0.05,0.1,0.15,0.2,0.25,0.3,0.35',
        help="Comma-separated list of sample sizes (e.g. '0.1,0.2,0.3')"
    )

    args = parser.parse_args()

    sample_size_list = [float(s.strip()) for s in args.sample_size_list.split(',')]

    if args.dataset == 'synthetic':
        print("[INFO] Generating synthetic dataset...")
        X, y = make_classification(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_informative=int(args.n_features*0.9),
            n_redundant=args.n_features - int(args.n_features*0.9),
            n_classes=2,
            weights=[0.05, 0.95],
            random_state=args.random_state
        )
    elif args.dataset == 'adult':
        X_fetch, y_fetch = fetch_openml(name='adult', version=2, return_X_y=True, as_frame=True)
        n_used_samples = min(args.n_samples, len(y_fetch))
        
        # Stratified sampling       
        X, _, y, _ = train_test_split(X_fetch, y_fetch, train_size=n_used_samples, stratify=y, random_state=args.random_state)

    clf = MLPClassifier(random_state=args.random_state, max_iter=2000, early_stopping=True)

    reg = DarkNumberCorrectionFactorRegressor(
        estimator=clf,
        sample_size_list=sample_size_list,
        flip_fraction=args.flip_fraction,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        n_jobs=args.n_jobs,
        predict_mode=args.predict_mode,
        random_state=args.random_state,
        positive_class=args.positive_class,
        type=args.type,
        poly_degree=args.poly_degree
    )

    print("[INFO] Estimating correction factor using extrapolated regression...")
    reg.fit(X, y)
    extrapolated_score = reg.score()
    print(f"[RESULT] Estimated correction factor at sample_size=1.0: {extrapolated_score:.6f}")

    print("[INFO] Estimating true correction factor using full dataset...")
    true_estimator = DarkNumberCorrectionFactorEstimator(
        estimator=clone(clf),
        flip_fraction=args.flip_fraction,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        n_jobs=args.n_jobs,
        predict_mode=args.predict_mode,
        random_state=args.random_state,
        positive_class=args.positive_class,
        sample_size=1.0
    )
    true_estimator.fit(X, y)
    print(f"[RESULT] Actual correction factor from full data: {true_estimator.score():.6f}")

    if args.plot:
        reg.plot()


if __name__ == "__main__":
    main()