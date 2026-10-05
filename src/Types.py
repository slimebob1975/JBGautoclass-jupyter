from typing import Protocol


""" PIPELINE TYPES
    For a Pipeline the steps are n+1 objects which show the sequential changes done to X 
"""
class Transform(Protocol):
    """ 0 => n, requires both fit() and transform() """
    def fit(self, X, y=None, sample_weight=None):
        """
         Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : None
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample (depending on the object's type)

        Returns
        -------
        self : object
            Fitted scaler.
        """
    
    def transform(self, X):
        """Scale features of X according to feature_range.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    Input data that will be transformed.

                Returns
                -------
                Xt : ndarray of shape (n_samples, n_features)
                    Transformed data.
        """


class Estimator(Protocol):
    """ Final n+1, requires only fit() """

    def fit(self, X, y=None, sample_weight=None):
        """
         Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample (depending on the object's type)

        Returns
        -------
        self : object
            Fitted scaler.
        """

class Detecter(Protocol):
    """ Given to some Estimators, requires detect()
        As the class that tracks detecters is called Detector, that name is already taken
    """
    def detect(self, X, y):
        """ Detects noise likelihood for each sample in the dataset """