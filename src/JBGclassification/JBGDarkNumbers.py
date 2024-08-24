import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def compute_dark_number(df: pd.DataFrame, real: str, predicted: str):
    
    # Compute confusion matrix elements
    TP = len(df[(df[real] == 1) & (df[predicted] == 1)])
    FP = len(df[(df[real] == 0) & (df[predicted] == 1)])
    FN = len(df[(df[real] == 1) & (df[predicted] == 0)])
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate dark number using single alpha correction
    dark_number = (precision**(-1) - 1) * (2 - recall)
    
    return dark_number

# Compute dark number based on one single alpha value
def compute_dark_number_single_alpha(df: pd.DataFrame, real: str, predicted: str, probability: str):
    
    # Compute confusion matrix elements
    TP = len(df[(df[real] == 1) & (df[predicted] == 1)])
    FP = len(df[(df[real] == 0) & (df[predicted] == 1)])
    FN = len(df[(df[real] == 1) & (df[predicted] == 0)])
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Compute the mean certainty of the predictions where real and predicted differ
    alpha = df[(df[real] != df[predicted])][probability].mean()
    
    # Calculate dark number using single alpha correction
    dark_number = alpha * (precision**(-1) - 1) * (2 - recall)
    
    return alpha, dark_number

# Compute dark number based on two alpha values

def compute_dark_number_separated_alpha(df: pd.DataFrame, real: str, predicted: str, probability: str):
    
    # Compute confusion matrix elements
    TP = len(df[(df[real] == 1) & (df[predicted] == 1)])
    FP = len(df[(df[real] == 0) & (df[predicted] == 1)])
    FN = len(df[(df[real] == 1) & (df[predicted] == 0)])
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Compute mean certainty for FP and FN
    alpha_FP = df[(df[real] == 0) & (df[predicted] == 1)][probability].mean() if FP > 0 else 1
    alpha_FN = df[(df[real] == 1) & (df[predicted] == 0)][probability].mean() if FN > 0 else 1

    # Calculate dark number using separated alpha correction
    dark_number = alpha_FP * (precision**(-1) - 1) * alpha_FN * (2 - recall)
    
    return [alpha_FP, alpha_FN], dark_number

# Compute dark number with non-linear scaling
def compute_dark_number_non_linear(df: pd.DataFrame, real: str, predicted: str, probability: str,  \
                                   use_alpha=False, log_base: float=np.e, root_degree: int=3):
    
    # Compute confusion matrix elements
    TP = len(df[(df[real] == 1) & (df[predicted] == 1)])
    FP = len(df[(df[real] == 0) & (df[predicted] == 1)])
    FN = len(df[(df[real] == 1) & (df[predicted] == 0)])
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Alpha calculation where real and predicted differs
    if use_alpha:
        alpha = df[(df[real] != df[predicted])][probability].mean()
    else:
        alpha = 1.0

    # Calculate dark number using non-linear scaling
    dark_number = alpha * np.log(precision**(-1)) / np.log(log_base) * (2 - recall**(1/root_degree))
    
    return alpha if use_alpha else None, dark_number

# Unified interface for calculating dark numbers
def compute_dark_numbers(df, real_target, pred_target, prob_pred, type=None, log_base=np.e, root_degree=3):

    df_mod = df.copy(deep=True)
    
    dark_numbers = pd.DataFrame(columns=["type", "target", "alphas", "dark_number"])

    # Compute dark number for each type of dark number
    for target in df[real_target].unique():

        # Convert to binary classification 
        # TODO: check why it produces the same results for each target
        df_mod[real_target] = df_mod[real_target].apply(func=(lambda x: 1 if x == target else 0))
        df_mod[pred_target] = df_mod[pred_target].apply(func=(lambda x: 1 if x == target else 0))

        if type == "single_alpha":
            alpha, dark_number = compute_dark_number_single_alpha(df_mod, real_target, pred_target, prob_pred)
        elif type == "separated_alpha":
            alpha, dark_number = compute_dark_number_separated_alpha(df_mod, real_target, pred_target, prob_pred)
        elif type == "non_linear":
            alpha, dark_number = compute_dark_number_non_linear(df_mod, real_target, pred_target, prob_pred, \
                                                         use_alpha=False, log_base=log_base, root_degree=root_degree)
        elif type == "non_linear_alpha":
            alpha, dark_number = compute_dark_number_non_linear(df_mod, real_target, pred_target, prob_pred, \
                                                         use_alpha=True, log_base=log_base, root_degree=root_degree)
        else:
            dark_number = compute_dark_number(df_mod, real_target, pred_target)
            alpha = None    
        
        row = {"type": type, "target": target, "alphas": alpha, "dark_number": dark_number}
        dark_numbers.loc[len(dark_numbers)] = row
        
    return dark_numbers

def main():
    
    print("Dark number computations")

    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict probabilities and classes
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Create a DataFrame with the real, predicted classes and probabilities
    df = pd.DataFrame({
        'real': y_test,
        'predicted': y_pred,
        'probability': y_prob
    })

    # Calculate the dark number estimates
    dark_number = compute_dark_number(df, 'real', 'predicted')
    dark_number_single_alpha = compute_dark_number_single_alpha(df, 'real', 'predicted', 'probability')
    dark_number_separated_alpha = compute_dark_number_separated_alpha(df, 'real', 'predicted', 'probability')
    dark_number_non_linear = compute_dark_number_non_linear(df, 'real', 'predicted', 'probability')

    # Print the results
    print(f"Dark Number (No Alpha): {dark_number}")
    print(f"Dark Number (Single Alpha Correction): {dark_number_single_alpha}")
    print(f"Dark Number (Separated Alpha Correction): {dark_number_separated_alpha}")
    print(f"Dark Number (Non-Linear Scaling): {dark_number_non_linear}")
    for type in [None, "single_alpha", "separated_alpha", "non_linear", "non_linear_alpha"]:
        dark_numbers = compute_dark_numbers(df, 'real', 'predicted', 'probability', type=type, log_base=np.e, root_degree=3)
        print(dark_numbers)
    
    
if __name__ == "__main__":
    main()