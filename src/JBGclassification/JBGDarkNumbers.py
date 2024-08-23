import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Compute dark number based on one single alpha value
def compute_dark_number_single_alpha(df: pd.DataFrame, real: str, predicted: str, probability: str):
    
    # Compute confusion matrix elements
    TP = len(df[(df[real] == 1) & (df[predicted] == 1)])
    FP = len(df[(df[real] == 0) & (df[predicted] == 1)])
    FN = len(df[(df[real] == 1) & (df[predicted] == 0)])
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Compute the mean certainty of the predictions
    alpha = df[probability].mean()

    # Calculate dark number using single alpha correction
    dark_number = alpha * (precision**(-1) - 1) * (2 - recall)
    
    return dark_number

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
    
    return dark_number

# Compute dark number with non-linear scaling
def compute_dark_number_non_linear(df: pd.DataFrame, real: str, predicted: str, probability: str,  \
                                   log_base: float=np.e, root_degree: int=3):
    
    # Compute confusion matrix elements
    TP = len(df[(df[real] == 1) & (df[predicted] == 1)])
    FP = len(df[(df[real] == 0) & (df[predicted] == 1)])
    FN = len(df[(df[real] == 1) & (df[predicted] == 0)])
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate dark number using non-linear scaling
    dark_number = np.log(precision**(-1)) / np.log(log_base) * (2 - recall**(1/root_degree))
    
    return dark_number


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
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of being class 1 (Iris-virginica)
    y_pred = model.predict(X_test)

    # Create a DataFrame with the real, predicted classes and probabilities
    df = pd.DataFrame({
        'real': y_test,
        'predicted': y_pred,
        'probability': y_prob
    })

    # Calculate the dark number estimates
    dark_number_single_alpha = compute_dark_number_single_alpha(df, 'real', 'predicted', 'probability')
    dark_number_separated_alpha = compute_dark_number_separated_alpha(df, 'real', 'predicted', 'probability')
    dark_number_non_linear = compute_dark_number_non_linear(df, 'real', 'predicted', 'probability')

    # Print the results
    print(f"Dark Number (Single Alpha Correction): {dark_number_single_alpha}")
    print(f"Dark Number (Separated Alpha Correction): {dark_number_separated_alpha}")
    print(f"Dark Number (Non-Linear Scaling): {dark_number_non_linear}")
    
    
if __name__ == "__main__":
    main()