import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from JBGStreamedLogger import JBGLogger
import sys
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

def compute_posneg_rates(TN: int, FP: int, FN: int, TP: int):

    # Compute performance rates
    TNr = TN / (TN + FP) if (TN + FP) > 0 else -1
    TPr = TP / (TP + FN) if (TP + FN) > 0 else -1

    return TPr, TNr

def compute_dark_number(real: pd.Series, predicted: pd.Series):
    
    # Compute confusion matrix elements
    TN, FP, FN, TP = confusion_matrix(real, predicted).ravel()
    
    # Computes rates
    TPr, TNr = compute_posneg_rates(TN, FP, FN, TP)

    # Calculate dark number using no alpha correction
    dark_number = (1 - TNr) * (2 - TPr)
    
    return dark_number

# Compute dark number based on one single alpha value
def compute_dark_number_single_alpha(real: pd.Series, predicted: pd.Series, pred_prob: pd.Series):
    
    # Compute confusion matrix elements
    TN, FP, FN, TP = confusion_matrix(real, predicted).ravel()
    
    # Computes rates
    TPr, TNr = compute_posneg_rates(TN, FP, FN, TP)

    # Compute the mean certainty of the predictions where real and predicted differ
    alpha = pred_prob[real != predicted].mean() if (FP > 0) else 1
    
    # Calculate dark number using single alpha correction
    dark_number = alpha * (1 - TNr) * (2 - TPr)
    
    return alpha, dark_number

# Compute dark number based on two alpha values

def compute_dark_number_separated_alpha(real: pd.Series, predicted: pd.Series, pred_prob: pd.Series):
    
    # Compute confusion matrix elements
    TN, FP, FN, TP = confusion_matrix(real, predicted).ravel()
    
    # Computes rates
    TPr, TNr = compute_posneg_rates(TN, FP, FN, TP)

    # Compute mean certainty for FP and FN
    alpha_FP = pred_prob[(real == 0) & (predicted == 1)].mean() if FP > 0 else 1
    alpha_FN = pred_prob[(real == 1) & (predicted == 0)].mean() if FN > 0 else 1

    # Calculate dark number using separated alpha correction
    dark_number = alpha_FP * (1 - TNr) * alpha_FN * (2 - TPr)
    
    return [alpha_FP, alpha_FN], dark_number

# Compute dark number with non-linear scaling
def compute_dark_number_non_linear(real: pd.Series, predicted: pd.Series, pred_prob: pd.Series,  \
                                   use_alpha=False,  root_degree: int=3):
    
    # Compute confusion matrix elements
    TN, FP, FN, TP = confusion_matrix(real, predicted).ravel()
    
    # Computes rates
    TPr, TNr = compute_posneg_rates(TN, FP, FN, TP)

    # Alpha calculation where real and predicted differs
    if use_alpha:
        alpha = pred_prob[real != predicted].mean() if (FP > 0) else 1
    else:
        alpha = 1.0

    # Calculate dark number using non-linear scaling
    dark_number = alpha * (1 - TNr**(1 / root_degree)) * (2 - TPr**(1 / root_degree))
    
    return alpha if use_alpha else 1.0, dark_number

def estimate_corr_crossfit_serial(
    base_pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    flip_rate: float = 0.20,
    R: int = 10,
    K: int = 5,
    threshold: float = 0.5,
    prior: str = "jeffreys",      # "jeffreys" or "laplace"
    corr_cap: float = 50.0,
    random_state: int | None = 42
) -> tuple[float, tuple[float, float]]:
    """
    Estimate correction factor 'corr' by flipping a share of positives to negative,
    training on the manipulated labels, and measuring how many flipped items the model
    still predicts as positive. Repeats with different random seeds and stratified folds.

    corr = 1 / p_hat, with a small-sample prior and an upper cap for numerical stability.
    Returns (corr_mean, (p5, p95)) where (p5, p95) is a 5–95% interval across R repeats.

    Parameters
    ----------
    base_pipeline : sklearn/imbalanced-learn Pipeline (unfitted or fitted)
        We will CLONE this pipeline and refit inside the routine so your original model
        remains untouched.
    X, y : pd.DataFrame / pd.Series
        Original training data and labels (binary: 1 = positive, 0 = negative).
    flip_rate : float
        Fraction of positive labels to flip to 0 in each training fold.
    R : int
        Number of repetitions with new random flips.
    K : int
        Outer stratified folds; flips are applied within each training fold.
    threshold : float
        Decision threshold for "detected" hidden positives.
    prior : {"jeffreys", "laplace"}
        Small-sample shrinkage prior for p = k/n.
    corr_cap : float
        Upper cap for corr to avoid instability when detections are extremely few.
    random_state : int | None
        Controls reproducibility of flips and folds when provided.

    Notes
    -----
    - For each repetition and fold, we flip a random subset of the positives *in the training split*,
      fit a CLONE of the supplied pipeline on the flipped labels, and then make predictions on the
      training split to count detections among the flipped items.
    - To guard against optimistic bias from in-sample evaluation, we (a) keep flip_rate small,
      (b) aggregate across K-folds and R repeats, and (c) shrink p with a prior.
    - If you prefer stricter out-of-fold counting, you can adapt this template to add an inner
      split that holds out a slice of the flipped training items for counting.
    """

    rng = np.random.default_rng(random_state)
    y = pd.Series(y).astype(int)
    X = pd.DataFrame(X)

    # Sanity: ensure binary {0,1}
    uniq = sorted(pd.unique(y))
    if not (len(uniq) == 2 and uniq[0] in (0, 1) and uniq[1] in (0, 1)):
        raise ValueError("estimate_corr_crossfit expects binary labels encoded as 0/1.")

    # Helper for prior-adjusted p
    def shrink_p(k: int, n: int) -> float:
        if prior == "jeffreys":
            return (k + 0.5) / (n + 1.0)
        elif prior == "laplace":
            return (k + 1.0) / (n + 2.0)
        else:
            raise ValueError("prior must be 'jeffreys' or 'laplace'")

    corr_values = []

    for r in range(R):
        # New fold split each repetition for stability
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=None if random_state is None else (random_state + r))

        total_k = 0  # detections among flipped
        total_n = 0  # number flipped

        for train_idx, _ in skf.split(X, y):
            Xtr = X.iloc[train_idx]
            ytr = y.iloc[train_idx].copy()

            pos_idx = np.flatnonzero(ytr.values == 1)
            if pos_idx.size == 0:
                continue

            n_flip = max(1, int(round(flip_rate * pos_idx.size)))
            flipped_local_idx = rng.choice(pos_idx, size=n_flip, replace=False)

            ytr_flipped = ytr.copy()
            ytr_flipped.iloc[flipped_local_idx] = 0  # hide positives as negatives

            # Fit a fresh clone of the pipeline
            pipe = clone(base_pipeline)
            try:
                pipe.fit(Xtr, ytr_flipped)
            except TypeError:
                pipe.fit(Xtr.to_numpy(), ytr_flipped.to_numpy())

            # Predict on the *training split* and count recovered hidden positives
            try:
                proba = pipe.predict_proba(Xtr.iloc[flipped_local_idx])[:, 1]
            except TypeError:
                proba = pipe.predict_proba(Xtr.iloc[flipped_local_idx].to_numpy())[:, 1]

            detected = int((proba >= threshold).sum())
            total_k += detected
            total_n += n_flip

        if total_n == 0:
            # No flips (degenerate), fall back to corr=1
            corr_values.append(1.0)
            continue

        p_hat = shrink_p(total_k, total_n)
        corr_r = min(1.0 / max(p_hat, 1e-12), corr_cap)
        corr_values.append(float(corr_r))

    corr_values = np.array(corr_values, dtype=float)
    corr_mean = float(np.mean(corr_values))
    lo, hi = np.percentile(corr_values, [5, 95]).tolist()

    return corr_mean, (lo, hi)

def estimate_corr_crossfit(
    base_pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    flip_rate: float = 0.20,
    R: int = 10,
    K: int = 5,
    threshold: float = 0.5,
    prior: str = "jeffreys",
    corr_cap: float = 50.0,
    random_state: int | None = 42,
    n_jobs: int = 1,                  
):
    """
    Som tidigare, men med n_jobs för parallell körning över R on n_jobs > 1.
    Returnerar (corr_mean, (p5, p95)).
    """
    if n_jobs == 1:
        return estimate_corr_crossfit_serial(base_pipeline, X, y, flip_rate, R, K, threshold, prior, corr_cap, random_state)
    else: 
        y = pd.Series(y).astype(int)
        X = pd.DataFrame(X)

        uniq = sorted(pd.unique(y))
        if not (len(uniq) == 2 and set(uniq) <= {0, 1}):
            raise ValueError("estimate_corr_crossfit expects binary labels encoded as 0/1.")

        def shrink_p(k: int, n: int) -> float:
            if prior == "jeffreys":
                return (k + 0.5) / (n + 1.0)
            elif prior == "laplace":
                return (k + 1.0) / (n + 2.0)
            else:
                raise ValueError("prior must be 'jeffreys' or 'laplace'")

        # --- En repetition som fristående jobb (kan köras parallellt) ---
        def one_repeat(r_seed: int) -> float:
            rng = np.random.default_rng(r_seed)
            skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=r_seed)

            total_k = 0
            total_n = 0

            for train_idx, _ in skf.split(X, y):
                Xtr = X.iloc[train_idx]
                ytr = y.iloc[train_idx].copy()

                pos_idx = np.flatnonzero(ytr.values == 1)
                if pos_idx.size == 0:
                    continue

                n_flip = max(1, int(round(flip_rate * pos_idx.size)))
                flipped_local_idx = rng.choice(pos_idx, size=n_flip, replace=False)

                ytr_flipped = ytr.copy()
                ytr_flipped.iloc[flipped_local_idx] = 0

                pipe = clone(base_pipeline)
                try:
                    pipe.fit(Xtr, ytr_flipped)
                except TypeError:
                    pipe.fit(Xtr.to_numpy(), ytr_flipped.to_numpy())

                try:
                    proba = pipe.predict_proba(Xtr.iloc[flipped_local_idx])[:, 1]
                except TypeError:
                    proba = pipe.predict_proba(Xtr.iloc[flipped_local_idx].to_numpy())[:, 1]

                detected = int((proba >= threshold).sum())
                total_k += detected
                total_n += n_flip

            if total_n == 0:
                return 1.0

            p_hat = shrink_p(total_k, total_n)
            corr_r = min(1.0 / max(p_hat, 1e-12), corr_cap)
            return float(corr_r)

        # --- Parallell körning över R upprepningar ---
        # Skapa deterministiska seeds per repetition om random_state är satt
        if random_state is None:
            seeds = [None] * R
        else:
            rng_master = np.random.default_rng(random_state)
            seeds = rng_master.integers(0, 2**32 - 1, size=R).tolist()

        corr_values = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(one_repeat)(s) for s in seeds
        )

        corr_values = np.asarray(corr_values, dtype=float)
        corr_mean = float(np.mean(corr_values))
        lo, hi = np.percentile(corr_values, [5, 95]).tolist()
        return corr_mean, (lo, hi)

# Unified interface for calculating dark numbers
def compute_dark_numbers(real_target: pd.Series, pred_target: pd.Series, prob_pred: pd.Series, \
                         type="base", root_degree=3):

    # This DataFrame will hold the results
    dark_numbers = pd.DataFrame(columns=["type", "target", "alphas", "dark_number"])

    # What types of dark numbers to consider?
    if type == "all":
        types = ["base", "single_alpha", "separated_alpha", "non_linear", "non_linear_alpha"]
    else:
        types = [type]

    # Compute dark number for each type
    for type in types:
    
    # Compute dark number for each target in the classification
        for target in real_target.unique():

            # Convert to binary classification 
            real_target_bin = pd.Series(real_target.apply(func=(lambda x: 1 if x == target else 0)), name="real_target")
            pred_target_bin = pd.Series(pred_target.apply(func=(lambda x: 1 if x == target else 0)), name="pred_target")

            if type == "base":
                dark_number = compute_dark_number(real_target_bin, pred_target_bin)
                alpha = 1.0   
            elif type == "single_alpha":
                alpha, dark_number = compute_dark_number_single_alpha(real_target_bin, pred_target_bin, prob_pred)
            elif type == "separated_alpha":
                alpha, dark_number = compute_dark_number_separated_alpha(real_target_bin, pred_target_bin, prob_pred)
            elif type == "non_linear":
                alpha, dark_number = compute_dark_number_non_linear(real_target_bin, pred_target_bin, prob_pred, \
                                                            use_alpha=False, root_degree=root_degree)
            elif type == "non_linear_alpha":
                alpha, dark_number = compute_dark_number_non_linear(real_target_bin, pred_target_bin, prob_pred, \
                                                            use_alpha=True, root_degree=root_degree)
            else:
                raise ValueError(f"The type of dark number is not supported: {type}")
            
            row = {"type": type, "target": target, "alphas": alpha, "dark_number": dark_number}
            dark_numbers.loc[len(dark_numbers)] = row
        
    return dark_numbers

def main():
    
    print("Dark number computations")

    # Load the Breast Cancer dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    accuracy = model.score(X_train, y_train)
    print(f"Train accuracy: {accuracy}")
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    # Predict probabilities and classes
    y_prob = model.predict_proba(X_train)
    y_prob = [max(row) for row in y_prob]
    y_pred = model.predict(X_train)

    # Create Series with the real, predicted classes and probabilities
    y_train = pd.Series(y_train)
    y_pred = pd.Series(y_pred)
    y_prob = pd.Series(y_prob)

    # Calculate the dark number estimates
    # Convert to binary classification 
    y_train_bin = pd.Series(y_train.apply(func=(lambda x: 1 if x == 1 else 0)))
    y_pred_bin = pd.Series(y_pred.apply(func=(lambda x: 1 if x == 1 else 0)))
    dark_number = compute_dark_number(y_train_bin, y_pred_bin)
    dark_number_single_alpha = compute_dark_number_single_alpha(y_train_bin, y_pred_bin, y_prob)
    dark_number_separated_alpha = compute_dark_number_separated_alpha(y_train_bin, y_pred_bin, y_prob)
    dark_number_non_linear = compute_dark_number_non_linear(y_train_bin, y_pred_bin, y_prob)
    dark_number_non_linear_alpha = compute_dark_number_non_linear(y_train_bin, y_pred_bin, y_prob, use_alpha=True)

    # Print the results
    print(f"Dark Number (No Alpha): {dark_number}")
    print(f"Dark Number (Single Alpha Correction): {dark_number_single_alpha}")
    print(f"Dark Number (Separated Alpha Correction): {dark_number_separated_alpha}")
    print(f"Dark Number (Non-Linear Scaling): {dark_number_non_linear}")
    print(f"Dark Number (Non-Linear Scaling with Alpha correction): {dark_number_non_linear_alpha}")

    for type in ["single_alpha", "separated_alpha", "non_linear", "non_linear_alpha", "all"]:
        dark_numbers = compute_dark_numbers(y_train, y_pred, y_prob, type=type, root_degree=3)
    print(dark_numbers)
    
    
if __name__ == "__main__":
    main()