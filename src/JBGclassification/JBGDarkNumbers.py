import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from JBGLogger import JBGLogger
import sys

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