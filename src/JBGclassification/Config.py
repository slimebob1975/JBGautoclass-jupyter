from __future__ import annotations
import copy
import enum
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Type, TypeVar, Union

import pandas
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, NMF
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, SGDClassifier)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.naive_bayes import (BernoulliNB, ComplementNB, GaussianNB,
                                 MultinomialNB)
from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid,
                               RadiusNeighborsClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (Binarizer, MaxAbsScaler, MinMaxScaler,
                                   Normalizer, StandardScaler)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (make_scorer, accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score, precision_score,
                             matthews_corrcoef, average_precision_score)

from skclean.models import RobustForest
from skclean.detectors import (KDN, ForestKDN, RkDN)
from skclean.handlers import WeightedBagging, Costing, CLNI, Filter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import (EasyEnsembleClassifier, RUSBoostClassifier, 
     BalancedBaggingClassifier, BalancedRandomForestClassifier)

from JBGExceptions import ConfigException
from JBGExperimental import (JBGRobustLogisticRegression, JBGRobustCentroid, 
                            JBGPartitioningDetector, JBGMCS, JBGInstanceHardness,
                            JBGRandomForestDetector)
import Helpers


class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_components(self, component, components, exception = None) -> None:
        """ Printing Reduction components"""

""" For a Pipeline the steps are n+1 objects which show the sequential changes done to X """
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


class RateType(enum.Enum):
    # I = Individuell bedömning. Modellen kan ge sannolikheter för individuella dataelement.
    I = "Individual"
    # A = Allmän bedömning. Modellen kan endast ge gemensam genomsnittlig sannolikhet för alla dataelementen tillsammans.
    A = "General"
    # U = Okänd. Lade jag till som en framtida utväg ifall ingen av de ovanstående fanns tillgängliga.
    U = "Unknown"


class MetaEnum(enum.Enum):
    @property
    def full_name(self):
        if isinstance(self.value, dict):
            return self.value["full_name"]
        
        return self.value
    
    @classmethod
    def get_sorted_list(cls, none_all_first: bool = True) -> list[tuple[str, str]]:
        if not none_all_first:
            return sorted([(item.full_name, item.name) for item in cls])

        excluded_enums = []
        for name in ["NON", "ALL"]:
            try:
                item = cls[name]
            except KeyError:
                """ Left blank on purpose """
            else:
                excluded_enums.insert(0, item)
        
        listed_enums = sorted([(item.full_name, item.name) for item in cls if item not in excluded_enums])

        prefix_list = [(item.full_name, item.name) for item in excluded_enums]
        
        return prefix_list + listed_enums

    def get_function_name(self, do_or_get: str = 'do') -> str:
        if do_or_get == 'get':
            return f"get_{self.name}"
        elif do_or_get == 'do':
            return f"do_{self.name}"
        else:
            return None

    def has_function(self, do_or_get: str = 'do') -> bool:
        do_or_get = self.get_function_name(do_or_get)
        return hasattr(self, do_or_get) and callable(getattr(self, do_or_get))

    def call_function(self, do_or_get: str = 'do', **kwargs):
        do_or_get = self.get_function_name(do_or_get)
        if hasattr(self, do_or_get) and callable(func := getattr(self, do_or_get)):
            return func(**kwargs)
        
        return None

    def get_function(self, **kwargs):
        return self.call_function(self, 'get', **kwargs)

class Detector(MetaEnum):
    ALL = { "full_name": "All" }
    NON = { "full_name": "None" }
    KDN = { "full_name":"KDN" }
    FKDN = { "full_name": "Forest KDN" }
    RKDN = { "full_name": "Recursive KDN" }
    PDEC = { "full_name": "Partitioning Detector + Label Encoder" }
    MCS = { "full_name": "Markov Chain Sampling + Label Encoder" } 
    INH = { "full_name": "Instance Hardness Detector" }
    RFD = { "full_name": "Random Forest Detector" }

    @classmethod
    def list_callable_detectors(cls) -> list[tuple]:
        """ Gets a list of detectors that are callable (including NON -> None)
            in the form (detector, called function)
        """
        return [(dt, dt.call_detector()) for dt in cls if dt.has_function()]

    def call_detector(self)  -> Union[Detecter, None]: 
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('do')

    def do_NON(self) -> None:
        """ While this return is superfluos, it helps with the listings of detectors """
        return None

    def do_KDN(self) -> KDN:
        return KDN()

    def do_FKDN(self) -> ForestKDN:
        return ForestKDN()

    def do_RKDN(self) -> RkDN:
        return RkDN()

    def do_PDEC(self) -> JBGPartitioningDetector:
        return JBGPartitioningDetector()

    def do_MCS(self) -> JBGMCS:
        return JBGMCS()

    def do_INH(self) -> JBGInstanceHardness:
        return JBGInstanceHardness()

    def do_RFD(self) -> JBGRandomForestDetector:
        return JBGRandomForestDetector()

class AlgorithmGridSearchParams(MetaEnum):
    DUMY =  {"parameters": {'strategy': ('most_frequent', 'prior', 'stratified', 'uniform', 'constant')}}
    SRF1 = {"parameters": {}}
    SRF2 = {"parameters": {}} 
    BARF = {"parameters": {'criterion': ('gini', 'entropy'), 'n_estimators':[10,50,100,200], 
            'class_weight': ('balanced', 'balanced_subsample', None)}}
    BABC = {"parameters": {'n_estimators': [5, 10 , 15], 'max_samples': (0.5, 1.0, 2.0), 
            'max_features': (0.5, 1.0, 2.0), 'warm_start': (True, False)}}
    RUBC = {"parameters": {'n_estimators': (10,30,50,100), 'learning_rate':(0.1, 1.0, 2.0)}}
    EAEC = {"parameters": {'n_estimators': [5, 10 , 15], 'warm_start': (True, False), 'replacement': (True, False)}}
    RTCL = {"parameters": {'method': ('simple', 'weighted'), 'K': [5, 10, 15], 'n_estimators': (50,100,150)}}
    RLRN = {"parameters": {'PN': [0.1, 0.2, 0.5, 0.10], 'NP': [0.1, 0.2, 0.5, 0.10], 'C': [0.1, 1, 10]}}
    RCNT = {"parameters": {}}
    LRN = {"parameters": {'penalty':('l1', 'l2', 'elasticnet', 'none'), 'tol': [1e-3, 1e-4, 1e-5], 'C': [0.1, 1, 10],
           'class_weight': ('balanced', None)}}
    KNN = {"parameters": {'n_neighbors': (5, 10, 15), 'weights': ('uniform', 'distance'), 
           'algorithm': ('ball_tree', 'kd_tree', 'brute'), 'p': (1, 2)}}
    RADN = {"parameters": {'radius': np.arange(0.5, 1.5, 0.1), 'weights': ('uniform','distance'), 'outlier_label': ('most_frequent', None)}}
    DTC = {"parameters": {'criterion': ('gini', 'entropy', 'log_loss'), 'splitter': ('best', 'random'), 
           'class_weight': ('balanced', None)}}
    GNB = {"parameters": {'var_smoothing': (1e-7, 1e-8, 1e-9)}}
    MNB = {"parameters": {'alpha': (0.0, 0.1, 1.0), 'fit_prior': (True, False)}}
    BNB = {"parameters": {'alpha': (0.0, 0.1, 1.0), 'fit_prior': (True, False)}}
    CNB = {"parameters": {'alpha': (0.0, 0.1, 1.0), 'fit_prior': (True, False), 'norm': (True, False)}}
    REC = {"parameters": {'alpha': [0.1, 1.0, 10.0], 'tol': [1e-2, 1e-3, 1e-4], 'class_weight': ('balanced', None)}}
    PCN = {"parameters": {'penalty': ('l2', 'l1', 'elasticnet'), 'alpha': (1e-3, 1e-4, 1e-5), 
           'class_weight': ('balanced', None)}}
    PAC = {"parameters": {'class_weight': ('balanced', None)}}
    RFCL = {"parameters": {'criterion': ('gini', 'entropy', 'log_loss'), 'n_estimators':[10,50,100,200], 
            'max_features': ('sqrt', 'log2'), 'class_weight': ('balanced', 'balanced_subsample', None)}}
    LSVC = {"parameters": {'penalty': ('l1', 'l2'), 'loss': ('hinge', 'squared_hinge'), 'dual': (True, False), 
            'class_weight': ('balanced', None)}} 
    SLSV = {"parameters": {}}
    SGDE = {"parameters": {'loss': ('hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 
            'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'), 
            'penalty': ('l2', 'l1', 'elasticnet')}}
    NCT = {"parameters": {'metric': ('euclidian', 'manhattan'), 'shrink_threshold': np.arange(0, 1.01, 0.01)}}
    SVC = {"parameters": {'C': [0.1,1, 10, 100], 'gamma': [1 , 0.1 ,0.01 ,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}}
    SSVC = {"parameters": {}}
    LDA = {"parameters": {'solver': ('svd','lsqr','eigen'), 'shrinkage': ('auto', None), 'tol': [1e-3, 1e-4, 1e-5]}}
    QDA = {"parameters": {'reg_param': np.arange(0.1, 1.0, 0.1), 'tol': [1e-3, 1e-4, 1e-5]}}
    BGC = {"parameters": {'n_estimators': [5, 10 , 15], 'max_samples': (0.5, 1.0, 2.0), 
            'max_features': (0.5, 1.0, 2.0), 'warm_start': (True, False)}}
    ETC = {"parameters": {'criterion': ('gini', 'entropy', 'log_loss'), 'n_estimators':[10,50,100,200], 
            'max_depth': range(1, 10, 1), 'max_features': ('sqrt', 'log2', None),
            'class_weight': ('balanced', 'balanced_subsample', None)}}
    ABC = {"parameters": {'n_estimators': (10,30,50,100), 'learning_rate':(0.1, 1.0, 2.0)}}
    GBC = {"parameters": {'loss': ['log_loss', 'exponential'], 
        'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        'min_samples_split': np.linspace(0.1, 0.5, 12), 'min_samples_leaf': np.linspace(0.1, 0.5, 12),
        'max_depth':[3,5,8], 'max_features':['log2','sqrt'], 'criterion': ['friedman_mse',  'mae'],
        'subsample':[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0], 'n_estimators':[10]}}
    HIST = {"parameters": {'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        'max_depth' : [25, 50, 75], 'l2_regularization': [0.0, 0.1, 1.5] }} 
    MLPC = {"parameters": {'activation': ('identity', 'logistic', 'tanh', 'relu'), 'solver': ('lbfgs', 'sgd', 'adam')}}
    GPC =  {"parameters": {'warm_start': (True, False), 'multi_class': ('one_vs_rest', 'one_vs_one')}}
    FRFD = {"parameters": {}}
    FPCD = {"parameters": {}}
    FFKD = {"parameters": {}}
    FINH = {"parameters": {}}
    CSTK = {"parameters": {}}
    CSTM = {"parameters": {}}
    CRFD = {"parameters": {}}
    CPCD = {"parameters": {}}
    CFKD = {"parameters": {}}
    CINH = {"parameters": {}}
    WBGK = {"parameters": {}}
    WBGM = {"parameters": {}}   
    WRFD = {"parameters": {}}
    WPCD = {"parameters": {}}
    WFKD = {"parameters": {}}
    WINH = {"parameters": {}}
    CLRF = {"parameters": {}}
    CLPC = {"parameters": {}}
    CLFK = {"parameters": {}}
    CLIH = {"parameters": {}}
    VOTG = {"parameters": {'voting': ('hard', 'soft')}}
    
    @property
    def parameters(self):
        if isinstance(self.value, dict):
            return self.value.get("parameters", {})
        
        return {}

class Algorithm(MetaEnum):
    DUMY = { "full_name": "Dummy Classifier", "search_params": AlgorithmGridSearchParams.DUMY, "rfe_compatible": False}
    SRF1 = { "full_name": "Stacked Random Forests 1", "search_params": AlgorithmGridSearchParams.SRF1, "rfe_compatible": False}
    SRF2 = { "full_name": "Stacked Random Forests 2", "search_params": AlgorithmGridSearchParams.SRF2, "rfe_compatible": False}
    BARF = { "full_name": "Balanced Random Forest", "search_params": AlgorithmGridSearchParams.BARF, "rfe_compatible": True}
    BABC = { "full_name": "Balanced Bagging Classifier", "search_params": AlgorithmGridSearchParams.BABC, "rfe_compatible": False}
    RUBC = { "full_name": "RUS Boost Classifier", "search_params": AlgorithmGridSearchParams.RUBC, "rfe_compatible": True}
    EAEC = { "full_name": "Easy Ensamble Classifier", "search_params": AlgorithmGridSearchParams.EAEC, "rfe_compatible": False}
    RTCL = { "full_name": "Robust Tree Classifier", "search_params": AlgorithmGridSearchParams.RTCL, "rfe_compatible": False}
    RLRN = { "full_name": "Robust Logistic Regression + Label Encoder", "search_params": AlgorithmGridSearchParams.RLRN, "rfe_compatible": False}
    RCNT = { "full_name": "Robust Centroid + Label Encoder", "search_params": AlgorithmGridSearchParams.RCNT, "rfe_compatible": False}
    LRN = { "full_name": "Logistic Regression", "search_params": AlgorithmGridSearchParams.LRN, "rfe_compatible": True}
    KNN = { "full_name": "K-Neighbors Classifier", "search_params": AlgorithmGridSearchParams.KNN, "rfe_compatible": False}
    RADN = { "full_name": "Radius Neighbors Classifier", "search_params": AlgorithmGridSearchParams.RADN, "rfe_compatible": False}
    DTC = { "full_name": "Decision Tree Classifier", "search_params": AlgorithmGridSearchParams.DTC, "rfe_compatible": True}
    GNB = { "full_name": "Gaussian Naive Bayes", "search_params": AlgorithmGridSearchParams.GNB, "rfe_compatible": False}
    MNB = { "full_name": "Multinomial Naive Bayes", "search_params": AlgorithmGridSearchParams.MNB, "rfe_compatible": True}
    BNB = { "full_name": "Bernoulli Naive Bayes", "search_params": AlgorithmGridSearchParams.BNB, "rfe_compatible": True}
    CNB = { "full_name": "Complement Naive Bayes", "search_params": AlgorithmGridSearchParams.CNB, "rfe_compatible": True}
    REC = { "full_name": "Ridge Classifier", "search_params": AlgorithmGridSearchParams.REC, "rfe_compatible": True}
    PCN = { "full_name": "Perceptron", "search_params": AlgorithmGridSearchParams.PCN, "rfe_compatible": True}
    PAC = { "full_name": "Passive Aggressive Classifier", "search_params": AlgorithmGridSearchParams.PAC, "rfe_compatible": True}
    RFCL = { "full_name": "Random Forest Classifier", "search_params": AlgorithmGridSearchParams.RFCL, "rfe_compatible": True}
    LSVC = { "full_name":  "Linear Support Vector", "search_params": AlgorithmGridSearchParams.LSVC, "rfe_compatible": True}
    SLSV = { "full_name": "Stacked Linear SVC", "search_params": AlgorithmGridSearchParams.SLSV, "rfe_compatible": False}
    SGDE = { "full_name": "Stochastic Gradient Descent", "search_params": AlgorithmGridSearchParams.SGDE, "rfe_compatible": True}
    NCT = { "full_name": "Nearest Centroid", "search_params": AlgorithmGridSearchParams.NCT, "rfe_compatible": False}
    SVC = { "full_name": "Support Vector Classification", "limit": 10000, "search_params": AlgorithmGridSearchParams.SVC, "rfe_compatible": False}
    SSVC = { "full_name": "Self Training Classifier", "limit": 10000, "search_params": AlgorithmGridSearchParams.SSVC, "rfe_compatible": False}
    LDA = { "full_name": "Linear Discriminant Analysis", "search_params": AlgorithmGridSearchParams.LDA, "rfe_compatible": True}
    QDA = { "full_name": "Quadratic Discriminant Analysis", "search_params": AlgorithmGridSearchParams.QDA, "rfe_compatible": False}
    BGC = { "full_name": "Bagging Classifier", "search_params": AlgorithmGridSearchParams.BGC, "rfe_compatible": False}
    ETC = { "full_name": "Extra Trees Classifier", "search_params": AlgorithmGridSearchParams.ETC, "rfe_compatible": True}
    ABC = { "full_name": "Ada Boost Classifier", "search_params": AlgorithmGridSearchParams.ABC, "rfe_compatible": True}
    GBC = { "full_name": "Gradient Boosting Classifier", "search_params": AlgorithmGridSearchParams.GBC, "rfe_compatible": True}
    HIST = { "full_name": "Histogram-based Gradient B. Classifier", "search_params": AlgorithmGridSearchParams.HIST, "rfe_compatible": False}
    MLPC = { "full_name": "Multi Layered Peceptron", "search_params": AlgorithmGridSearchParams.MLPC, "rfe_compatible": False}
    GPC = { "full_name": "Gaussian Process Classifier", "search_params": AlgorithmGridSearchParams.GPC, "rfe_compatible": False}
    FRFD = { "full_name": "Filter + RandomForestDetector", "detector": Detector.RFD, "search_params": AlgorithmGridSearchParams.FRFD, "rfe_compatible": False}
    FPCD = { "full_name": "Filter + PartitioningDetector", "detector": Detector.PDEC, "search_params": AlgorithmGridSearchParams.FPCD, "rfe_compatible": False}
    FFKD = { "full_name": "Filter + ForestKDN", "detector": Detector.FKDN, "search_params": AlgorithmGridSearchParams.FFKD, "rfe_compatible": False}
    FINH = { "full_name": "Filter + InstanceHardness", "detector": Detector.INH, "search_params": AlgorithmGridSearchParams.FINH, "rfe_compatible": False}
    CSTK = { "full_name": "Costing + KDN", "detector": Detector.KDN, "search_params": AlgorithmGridSearchParams.CSTK, "rfe_compatible": False}
    CSTM = { "full_name": "Costing + MCS", "detector": Detector.MCS, "search_params": AlgorithmGridSearchParams.CSTM, "rfe_compatible": False}
    CRFD = { "full_name": "Costing + RandomForestDetector", "detector": Detector.RFD, "search_params": AlgorithmGridSearchParams.CRFD, "rfe_compatible": False}
    CPCD = { "full_name": "Costing + PartitioningDetector", "detector": Detector.PDEC, "search_params": AlgorithmGridSearchParams.CPCD, "rfe_compatible": False}
    CFKD = { "full_name": "Costing + ForestKDN", "detector": Detector.FKDN, "search_params": AlgorithmGridSearchParams.CFKD, "rfe_compatible": False}
    CINH = { "full_name": "Costing + InstanceHardness", "detector": Detector.INH, "search_params": AlgorithmGridSearchParams.CINH, "rfe_compatible": False}
    WBGK = { "full_name": "WeightedBagging + KDN", "detector": Detector.KDN, "search_params": AlgorithmGridSearchParams.WBGK, "rfe_compatible": False}
    WBGM = { "full_name": "WeightedBagging + MCS", "detector": Detector.MCS, "search_params": AlgorithmGridSearchParams.WBGM, "rfe_compatible": False}   
    WRFD = { "full_name": "WeightedBagging + RandomForestDetector", "detector": Detector.RFD, "search_params": AlgorithmGridSearchParams.WRFD, "rfe_compatible": False}
    WPCD = { "full_name": "WeightedBagging + PartitioningDetector", "detector": Detector.PDEC, "search_params": AlgorithmGridSearchParams.WPCD, "rfe_compatible": False}
    WFKD = { "full_name": "WeightedBagging + ForestKDN", "detector": Detector.FKDN, "search_params": AlgorithmGridSearchParams.WFKD, "rfe_compatible": False}
    WINH = { "full_name": "WeightedBagging + InstanceHardness", "detector": Detector.INH, "search_params": AlgorithmGridSearchParams.WINH, "rfe_compatible": False}
    CLRF = { "full_name": "CLNI + RandomForestDetector", "detector": Detector.RFD, "search_params": AlgorithmGridSearchParams.CLRF, "rfe_compatible": False}
    CLPC = { "full_name": "CLNI + PartitioningDetector", "detector": Detector.PDEC, "search_params": AlgorithmGridSearchParams.CLPC, "rfe_compatible": False}
    CLFK = { "full_name": "CLNI + ForestKDN", "detector": Detector.FKDN, "search_params": AlgorithmGridSearchParams.CLFK, "rfe_compatible": False}
    CLIH = { "full_name": "CLNI + InstanceHardness", "detector": Detector.INH, "search_params": AlgorithmGridSearchParams.CLIH, "rfe_compatible": False}
    VOTG = { "full_name":  "Voting Classifier", "search_params": AlgorithmGridSearchParams.VOTG, "rfe_compatible": False}


    def get_full_name(self) -> str:
        return self.full_name

    @property
    def limit(self):
        if isinstance(self.value, dict):
            return self.value.get("limit")
        
        return None

    @property
    def detector(self):
        if isinstance(self.value, dict):
            return self.value.get("detector")

        return None

    @property
    def search_params(self):
        if isinstance(self.value, dict):
            return self.value.get("search_params")
        
        return None
    
    @property
    def rfe_compatible(self):
        if isinstance(self.value, dict):
            return self.value.get("rfe_compatible")
        
        return None

    @property
    def fit_params(self):
        if isinstance(self.value, dict):
            return self.value.get("fit_params", {})
        
        return {}

    @classmethod
    def list_callable_algorithms(cls, size: int, max_iterations: int) -> list[tuple]:
        """ Gets a list of algorithms that are callable
            in the form (algorithm, called function)
        """
        algorithms =  [(algo, algo.call_algorithm(max_iterations=max_iterations, size=size)) for algo in cls if algo.has_function()]
        algorithms.sort(key=lambda algotuple: algotuple[0].name)
        return algorithms

    @classmethod
    def get_robust_algorithms(cls) -> list:
        """ This list needs to be extended if we add more robust algorithms"""
        return [Algorithm.RTCL, Algorithm.RLRN, Algorithm.RCNT]

    def get_compound_name(self, prepros: Preprocess)->str:
        return f"{self.name}-{prepros.name}"

    def call_algorithm(self, max_iterations: int, size: int) -> Union[Estimator, None]:
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('do', max_iterations=max_iterations, size=size)

    def use_imb_pipeline(self) -> bool:
        return self in self.get_robust_algorithms()

    def do_DUMY(self, max_iterations: int, size: int)-> DummyClassifier:
        return DummyClassifier()
    
    def do_SRF1(self, max_iterations: int, size: int)-> StackingClassifier:
        estimators = [ \
                ('rfor',RobustForest()),\
                ('bfor',BalancedRandomForestClassifier())\
                ]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    def do_SRF2(self, max_iterations: int, size: int)-> StackingClassifier:
        estimators = [ \
                ('for', RandomForestClassifier()),\
                ('rfor',RobustForest()),\
                ('bfor',BalancedRandomForestClassifier())\
                ]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    def do_BARF(self, max_iterations: int, size: int)-> BalancedRandomForestClassifier:
        return BalancedRandomForestClassifier()
    
    def do_BABC(self, max_iterations: int, size: int)-> BalancedBaggingClassifier:
        return BalancedBaggingClassifier()
    
    def do_RUBC(self, max_iterations: int, size: int)-> RUSBoostClassifier:
        return RUSBoostClassifier()

    def do_EAEC(self, max_iterations: int, size: int)-> EasyEnsembleClassifier:
        return EasyEnsembleClassifier()

    def do_RTCL(self, max_iterations: int, size: int)-> RobustForest:
        return RobustForest()

    def do_RLRN(self, max_iterations: int, size: int)-> JBGRobustLogisticRegression:
        return JBGRobustLogisticRegression()

    def do_RCNT(self, max_iterations: int, size: int)-> JBGRobustCentroid:
        return JBGRobustCentroid()

    def do_LRN(self, max_iterations: int, size: int)-> LogisticRegression:
        return LogisticRegression(max_iter=max_iterations)

    def do_KNN(self, max_iterations: int, size: int)-> KNeighborsClassifier:
        return KNeighborsClassifier()

    def do_RADN(self, max_iterations: int, size: int)-> RadiusNeighborsClassifier:
        return RadiusNeighborsClassifier(outlier_label="most_frequent")

    def do_DTC(self, max_iterations: int, size: int)-> DecisionTreeClassifier:
        return DecisionTreeClassifier()

    def do_GNB(self, max_iterations: int, size: int)-> GaussianNB:
        return GaussianNB()

    def do_MNB(self, max_iterations: int, size: int)-> MultinomialNB:
        return MultinomialNB()

    def do_BNB(self, max_iterations: int, size: int)-> BernoulliNB:
        return BernoulliNB()

    def do_CNB(self, max_iterations: int, size: int)-> ComplementNB:
        return ComplementNB()

    def do_REC(self, max_iterations: int, size: int)-> RidgeClassifier:
        return RidgeClassifier(max_iter=max_iterations)

    def do_PCN(self, max_iterations: int, size: int)-> Perceptron:
        return Perceptron(max_iter=max_iterations)

    def do_PAC(self, max_iterations: int, size: int)-> PassiveAggressiveClassifier:
        return PassiveAggressiveClassifier(max_iter=max_iterations)

    def do_RFCL(self, max_iterations: int, size: int)-> RandomForestClassifier:
        return RandomForestClassifier()

    def do_LSVC(self, max_iterations: int, size: int)-> LinearSVC:
        return LinearSVC(max_iter=max_iterations) 

    def do_SLSV(self, max_iterations: int, size: int)-> Pipeline:
        return Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(max_iter=max_iterations, penalty="l1", dual=False))),
            ('classification', LinearSVC(max_iter=max_iterations, penalty="l2", dual=True))])

    def do_SVC(self, max_iterations: int, size: int):
        if size < self.limit:
            return SVC(max_iter=max_iterations)
        
        # TODO: communicate with terminal to warn? 
        # print("\nNotice: SVC model was exchange for LinearSVC since n_samples > {0}\n".format(self.LIMIT_SVC))
        return self.do_LSVC(max_iterations=max_iterations, size=size)

    def do_SSVC(self, max_iterations: int, size: int) -> SelfTrainingClassifier:
        return SelfTrainingClassifier(self.do_SVC(max_iterations, size))

    def do_SGDE(self, max_iterations: int, size: int)-> SGDClassifier:
        return SGDClassifier(max_iter=max_iterations)   

    def do_NCT(self, max_iterations: int, size: int)-> NearestCentroid:     
        return NearestCentroid()

    def do_LDA(self, max_iterations: int, size: int)-> LinearDiscriminantAnalysis:     
        return LinearDiscriminantAnalysis()

    def do_QDA(self, max_iterations: int, size: int)-> QuadraticDiscriminantAnalysis:     
        return QuadraticDiscriminantAnalysis()

    def do_BGC(self, max_iterations: int, size: int)-> BaggingClassifier:     
        return BaggingClassifier()

    def do_ETC(self, max_iterations: int, size: int)-> ExtraTreesClassifier:     
        return ExtraTreesClassifier()

    def do_ABC(self, max_iterations: int, size: int)-> AdaBoostClassifier:     
        return AdaBoostClassifier()

    def do_GBC(self, max_iterations: int, size: int)-> GradientBoostingClassifier:     
        return GradientBoostingClassifier()

    def do_HIST(self, max_iterations: int, size: int)-> HistGradientBoostingClassifier:     
        return HistGradientBoostingClassifier()

    def do_MLPC(self, max_iterations: int, size: int)-> MLPClassifier:
        return MLPClassifier(max_iter=max_iterations)

    def do_GPC(self, max_iterations: int, size: int)-> GaussianProcessClassifier:
        return GaussianProcessClassifier(max_iter_predict=max_iterations)

    def do_WBGK(self, max_iterations: int, size: int)-> WeightedBagging:
        return self.call_WB(self.detector)

    def do_WBGM(self, max_iterations: int, size: int)-> WeightedBagging: 
        return self.call_WB(self.detector)

    def do_WRFD(self, max_iterations: int, size: int)-> WeightedBagging:
        return self.call_WB(self.detector)

    def do_WPCD(self, max_iterations: int, size: int)-> WeightedBagging: 
        return self.call_WB(self.detector)

    def do_WFKD(self, max_iterations: int, size: int)-> WeightedBagging: 
        return self.call_WB(self.detector)

    def do_WINH(self, max_iterations: int, size: int)-> WeightedBagging: 
        return self.call_WB(self.detector)
    
    def call_WB(self, detector) -> WeightedBagging:
        return WeightedBagging(detector=detector.call_detector())
    
    def do_CSTK(self, max_iterations: int, size: int)-> Costing:
        return self.call_CST(self.detector)

    def do_CSTM(self, max_iterations: int, size: int)-> Costing: 
        return self.call_CST(self.detector)

    def do_CRFD(self, max_iterations: int, size: int)-> Costing:
        return self.call_CST(self.detector)

    def do_CPCD(self, max_iterations: int, size: int)-> Costing: 
        return self.call_CST(self.detector)

    def do_CFKD(self, max_iterations: int, size: int)-> Costing:
        return self.call_CST(self.detector)

    def do_CINH(self, max_iterations: int, size: int)-> Costing:
        return self.call_CST(self.detector)
    
    def call_CST(self, detector) -> Costing:
        return Costing(detector=detector.call_detector())

    def do_CLRF(self, max_iterations: int, size: int)-> CLNI:
        return self.call_CLNI(self.detector)
    
    def do_CLCP(self, max_iterations: int, size: int)-> CLNI:
        return self.call_CLNI(self.detector)
    
    def do_CLFK(self, max_iterations: int, size: int)-> CLNI:
        return self.call_CLNI(self.detector)

    def do_CLIH(self, max_iterations: int, size: int)-> CLNI:
        return self.call_CLNI(self.detector)
    
    def call_CLNI(self, detector) -> Costing:
        return CLNI(classifier=SVC(), detector=detector.call_detector())

    def do_FRFD(self, max_iterations: int, size: int)-> Filter:
        return self.call_FLT(self.detector)
    
    def do_FPCD(self, max_iterations: int, size: int)-> Filter:
        return self.call_FLT(self.detector)
    
    def do_FFKD(self, max_iterations: int, size: int)-> Filter:
        return self.call_FLT(self.detector)

    def do_FINH(self, max_iterations: int, size: int)-> Filter:
        return self.call_FLT(self.detector)
    
    def call_FLT(self, detector) -> Filter:
        return Filter(classifier=SVC(), detector=detector.call_detector())
    
    def do_VOTG(self, max_iterations: int, size: int) -> Filter:
        clf1 = LinearSVC()
        clf2 = RandomForestClassifier()
        clf3 = LogisticRegression()
        estimators=[('lsvc', clf1), ('rfc', clf2), ('lrn', clf3)]
        return VotingClassifier(estimators=estimators)

class AlgorithmTuple:

    def __init__(self, list) -> None:
        if isinstance(list, Iterable):
            algorithms = []
            for algorithm in list:
                if isinstance(algorithm, Algorithm):
                    algorithms.append(algorithm)
                elif isinstance(algorithm, str):
                    algorithms.append(Algorithm[algorithm])
                else:
                    raise ValueError("Each element in input list must be an Algorithm instance")
            self.algorithms = tuple(algorithms)
        else:
            raise ValueError("Input to AlgorithmTuple must be an iterable")

    def __str__(self) -> str:
        output = ""
        for algorithm in self.algorithms:
            output += ',' + str(algorithm.name)
        return output[1:]

    def get_full_name(self) -> str:
        return self.get_full_names()
    
    def get_full_names(self) -> str:
        output = ""
        for algorithm in self.algorithms:
            output += ', ' + str(algorithm.get_full_name())
        return output[2:]

    def list_callable_algorithms(self, size: int, max_iterations: int) -> list[tuple]:
        """ Gets a list of algorithms that are callable
            in the form (algorithm, called function)
        """
        algorithms =  [(algo, algo.call_algorithm(max_iterations=max_iterations, size=size)) for algo in self.algorithms if algo.has_function()]
        algorithms.sort(key=lambda algotuple: algotuple[0].name)
        return algorithms

class Preprocess(MetaEnum):
    NOS = "No Scaling"
    STA = "Standard Scaler"
    MIX = "Min-Max Scaler"
    MMX = "Max-Absolute Scaler"
    NRM = "Normalizer"
    BIN = "Binarizer"

    def get_full_name(self) -> str:
        return self.full_name

    @classmethod
    def list_callable_preprocessors(cls, is_text_data: bool) -> list[tuple]:
        """ Gets a list of preprocessors that are callable (including NOS -> None)
            in the form (preprocessor, called function)
        """
        return [(pp, pp.call_preprocess()) for pp in cls if pp.has_function() ]#and (pp.name != "BIN" or is_text_data)]

    def do_NOS(self) -> NonScaler:
        """ While this return is superfluos, it helps with the listings of preprocessors """
        return self.NonScaler()

    def NonScaler(self):
        return FunctionTransformer(lambda X: X)

    def do_STA(self) -> StandardScaler:
        return StandardScaler(with_mean=False)

    def do_MIX(self) -> MinMaxScaler:
        return MinMaxScaler()

    def do_MMX(self) -> MaxAbsScaler:
        return MaxAbsScaler()

    def do_NRM(self) -> Normalizer:
        return Normalizer()

    def do_BIN(self) -> Binarizer:
        return Binarizer()

    def call_preprocess(self) -> Union[Transform, None]:
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('do')

class PreprocessTuple:

    def __init__(self, list) -> None:
        #print("preprocess input list:", list)
        if isinstance(list, Iterable):
            preprocessors = []
            for preprocessor in list:
                if isinstance(preprocessor, Preprocess):
                    preprocessors.append(preprocessor)
                elif isinstance(preprocessor, str):
                    preprocessors.append(Preprocess[preprocessor])
                else:
                    raise ValueError("Each element in input list must be an Preprocess instance")
            self.preprocessors = tuple(preprocessors)
        else:
            raise ValueError("Input to PreprocessTuple must be an iterable")
        #print("Resulting preprocesstuple:", str(self.preprocessors))


    def __str__(self) -> str:
        output = ""
        for preprocessor in self.preprocessors:
            output += ',' + str(preprocessor.name)
        return output[1:]

    def get_full_names(self) -> str:
        output = ""
        for preprocessor in self.preprocessors:
            output += ', ' + str(preprocessor.get_full_name())
        return output[2:]

    def get_full_name(self) -> str:
        return self.get_full_names()

    def list_callable_preprocessors(self, is_text_data: bool) -> list[tuple]:
        """ Gets a list of preprocessors that are callable (including NOS -> None)
            in the form (preprocessor, called function)
        """
        return [(pp, pp.call_preprocess()) for pp in self.preprocessors if pp.has_function() ]#and (pp.name != "BIN" or is_text_data)]

class Reduction(MetaEnum):
    NOR = "No Reduction"
    RFE = "Recursive Feature Elimination"
    PCA = "Principal Component Analysis"
    NYS = "Nystroem Method"
    TSVD = "Truncated SVD"
    FICA = "Fast Indep. Component Analysis"
    NMF = "Non-Negative Matrix Factorization"
    GRP = "Gaussion Random Projection"
    ISO = "Isometric Mapping"
    LLE = "Locally Linearized Embedding"

    def get_full_name(self) -> str:
        return self.full_name

    @classmethod
    def list_callable_reductions(cls) -> list[tuple]:
        """ Gets a list of reductions that are callable (including NOR -> None)
            in the form (reduction, called function)
        """
        return [(rd, rd.call_transformation_theory()) for rd in cls if rd.has_function() ]

    def call_transformation(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('do', logger=logger, X=X, num_selected_features=num_selected_features)

    def call_transformation_theory(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        """ This is a possible idea to optimise perform_feature_selection(), with code from there below
        feature_selection = self.handler.config.get_feature_selection()
        num_selected_features = self.handler.config.get_num_selected_features()
        if feature_selection.has_function():
            self.handler.logger.print_info(f"{feature_selection.full_name} transformation of dataset under way...")
            self.X, feature_selection_transform = feature_selection.call_transformation(
                logger=self.handler.logger, X=self.X, num_selected_features=num_selected_features
            )
        """
        #logger.print_info(f"{self.full_name} transformation of dataset under way...")
        new_X = X
        transform = None
        try:
            new_X, transform = self.call_function('do', logger=logger, X=X, num_selected_features=num_selected_features)
        except TypeError:
            """ Acceptable error, call_function() returned None """

        return new_X, transform

    def _do_transformation(self, logger: Logger, X: pandas.DataFrame, transformation, components):
        try:
            feature_selection_transform = transformation
            feature_selection_transform.fit(X)
            X = feature_selection_transform.transform(X)
        except Exception as e:
            #logger.print_components(self.name, components, str(e))
            feature_selection_transform = None
        else:
            # Else in a try-except clause means that there were no exceptions
            #logger.print_info(f"...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})")

            return X, feature_selection_transform

    def get_NOR(self, num_samples: int, num_features: int, num_selected_features: int = None) -> NonReduction:
        return self.NonReduction()
    
    def do_NOR(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        """ While this return is superfluos, it helps with the listings of reductions """
        return self.NonReduction(X), self.NonReduction()

    def NonReduction(self):
        return FunctionTransformer(lambda X: X)
    
    # For now, use temporary fixed argument for estimator. Can be changed later!
    def do_RFE(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):

        tf = self.get_RFE(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)
    
    def get_RFE(self, num_samples: int, num_features: int, num_selected_features: int = None):
        
        return RFE(estimator=LinearSVC(), n_features_to_select=num_selected_features)
    
    def get_PCA(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        elif num_samples >= num_features:
            components = 'mle'
        else:
            components = Config.PCA_VARIANCE_EXPLAINED
        return PCA(n_components=None)
    
    def do_PCA(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):

        tf = self.get_PCA(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def get_NYS(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = max(Config.LOWER_LIMIT_REDUCTION, min(num_samples,num_features))
        return Nystroem(n_components=components)
    
    def do_NYS(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_NYS(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def get_TSVD(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = max(Config.LOWER_LIMIT_REDUCTION, min(num_samples,num_features))

        return TruncatedSVD(n_components=components)
    
    def do_TSVD(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_TSVD(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def get_FICA(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = max(Config.LOWER_LIMIT_REDUCTION, min(num_samples,num_features))
        return FastICA(n_components=components)
    
    def do_FICA(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_FICA(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)
        
    def get_NMF(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = max(Config.LOWER_LIMIT_REDUCTION, min(num_samples,num_features))
        return NMF(n_components=components)
    
    def do_NMF(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_NMF(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def find_NMF_components(self, logger: Logger, X: pandas.DataFrame, max_components: int = None):
        
        norm = np.linalg.norm(X.to_numpy())
        tol = 0.05

        # Loop to find the best possible number of component for NMF
        components = 2
        while components <= max_components:

            # Make a copy of X so that we do not destroy
            feature_transform = NMF(n_components=components).fit(X)
            err = feature_transform.reconstruction_err_
            rerr = err / norm
            #logger.print_info(f'Components: {components}. NMF reconstruction relative error: {rerr}')
            if rerr < tol:
                #logger.print_info(f'Tolerance {tol} fulfilled for {components} components')
                break
            else:
                components = min(components*2, max_components)

        return components

    def get_GRP(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = 'auto'
        return GaussianRandomProjection(n_components=components)

    def do_GRP(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_GRP(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)
        
    def get_ISO(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = Config.NON_LINEAR_REDUCTION_COMPONENTS
        return Isomap(n_components=components)

    def do_ISO(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_ISO(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def get_LLE(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = Config.NON_LINEAR_REDUCTION_COMPONENTS
        return LocallyLinearEmbedding(n_components=components)
    
    def do_LLE(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_LLE(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def call_reduction(self, num_samples: int, num_features: int, num_selected_features: int = None) -> Reduction:
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('get', num_samples=num_samples, num_features=num_features, \
            num_selected_features=num_selected_features)

class ReductionTuple:

    def __init__(self, list) -> None:
        if isinstance(list, Iterable):
            reductions = []
            for reduction in list:
                if isinstance(reduction, Reduction):
                    reductions.append(reduction)
                elif isinstance(reduction, str):
                    reductions.append(Reduction[reduction])
                else:
                    raise ValueError("Each element in input list must be an Reduction instance")
            self.reductions = tuple(reductions)
        else:
            raise ValueError("Input to ReductionTuple must be an iterable")

    def __str__(self) -> str:
        output = ""
        for reduction in self.reductions:
            output += ',' + str(reduction.name)
        return output[1:]

    def get_full_names(self) -> str:
        output = ""
        for reduction in self.reductions:
            output += ', ' + str(reduction.get_full_name())
        return output[2:]

    def get_full_name(self) -> str:
        return self.get_full_names()

    def list_callable_reductions(self, num_samples: int, num_features: int, \
        num_selected_features: int = None) -> list[tuple]:
        """ Gets a list of reductions that are callable
            in the form (reduction, called function)
        """
        reductions =  [(red, red.call_reduction(num_samples, num_features, num_selected_features)) \
            for red in self.reductions if red.has_function()]
        reductions.sort(key=lambda redtuple: redtuple[0].name)
        return reductions

class ScoreMetric(MetaEnum):
    accuracy = {"full_name": "Accuracy", "callable": accuracy_score, "kwargs": None}
    balanced_accuracy = {"full_name": "Balanced Accuracy", "callable": balanced_accuracy_score, "kwargs": {"adjusted": False}}
    balanced_accuracy_adjusted = {"full_name": "Balanced Accuracy (Adjusted)", "callable": balanced_accuracy_score, "kwargs": {"adjusted": True}}
    f1_micro = {"full_name": "Balanced F1 Micro", "callable": f1_score, "kwargs": {"average": 'micro'}}
    f1_macro = {"full_name": "Balanced F1 Macro", "callable": f1_score, "kwargs": {"average": 'macro'}}
    f1_weighted = {"full_name": "Balanced F1 Weighted", "callable": f1_score, "kwargs": {"average": 'weighted'}}
    recall_micro = {"full_name": "Recall Micro", "callable": recall_score, "kwargs": {"average": 'micro'}}
    recall_macro = {"full_name": "Recall Macro", "callable": recall_score, "kwargs": {"average": 'macro'}}
    recall_weighted = {"full_name": "Recall Weighted", "callable": recall_score, "kwargs": {"average": 'weighted'}}
    precision_micro = {"full_name": "Precision Micro", "callable": precision_score, "kwargs": {"average": 'micro'}}
    precision_macro = {"full_name": "Precision Macro", "callable": precision_score, "kwargs": {"average": 'macro'}}
    precision_weighted = {"full_name": "Precision Weighted", "callable": precision_score, "kwargs": {"average": 'weighted'}}
    mcc = {"full_name":"Matthews Corr. Coefficient", "callable": matthews_corrcoef, "kwargs": None}
    average_precision_micro = {"full_name": "Average Precision Micro", "callable": average_precision_score, "kwargs": {"average": 'micro'}}
    average_precision_macro = {"full_name": "Average Precision Macro", "callable": average_precision_score, "kwargs": {"average": 'macro'}}
    average_precision_weighted = {"full_name": "Average Precision Weighted", "callable": average_precision_score, "kwargs": {"average": 'weighted'}}

    def get_full_name(self) -> str:
        return self.full_name
    
    @property
    def callable(self):
        if isinstance(self.value, dict):
            return self.value.get("callable")

        return None

    @property
    def kwargs(self):
        if isinstance(self.value, dict):
            return self.value.get("kwargs")

        return None
    
    def get_mechanism(self) -> Union[str, Callable]:
        """ Returns the scoring mechanism based on the ScoreMetric"""
        return self.get_parametrized_scorer()

    def get_parametrized_scorer(self) -> Callable:
        """ Returns a scorer callable based on the ScoreMetric """
        
        if self.kwargs:
            return make_scorer(self.callable, **(self.kwargs))
        else:
            return make_scorer(self.callable)

T = TypeVar('T', bound='Config')

@dataclass
class Config:
    
    MAX_ITERATIONS = 20000
    CONFIG_FILENAME_START = "autoclassconfig_"
    CONFIG_SAMPLE_FILE = CONFIG_FILENAME_START + "template.py.txt"
    PCA_VARIANCE_EXPLAINED = 0.999
    LOWER_LIMIT_REDUCTION = 100
    NON_LINEAR_REDUCTION_COMPONENTS = 2

    DEFAULT_MODELS_PATH =  ".\\model\\"
    DEFAULT_MODEL_EXTENSION = ".sav"
    DEFAULT_TRAIN_OPTION = "Train a new model"

    TEXT_DATATYPES = ["nvarchar", "varchar", "char", "text", "enum", "set"]
    INT_DATATYPES = ["bigint", "int", "smallint", "tinyint"]

    TEMPLATE_TAGS = {
        "name": "<name>",
        "connection.odbc_driver": "<odbc_driver>",
        "connection.host": "<host>",
        "connection.trusted_connection": "<trusted_connection>",
        "connection.class_catalog": "<class_catalog>", 
        "connection.class_table": "<class_table>",
        "connection.class_table_script": "<class_table_script>",
        "connection.class_username": "<class_username>",
        "connection.class_password": "<class_password>",
        "connection.data_catalog": "<data_catalog>",
        "connection.data_table": "<data_table>",
        "connection.class_column": "<class_column>",
        "connection.data_text_columns": "<data_text_columns>",
        "connection.data_numerical_columns": "<data_numerical_columns>",
        "connection.id_column": "<id_column>",
        "connection.data_username": "<data_username>",
        "connection.data_password": "<data_password>",
        "mode.train": "<train>",
        "mode.predict": "<predict>",
        "mode.mispredicted": "<mispredicted>",
        "mode.use_metas": "<use_metas>",
        "mode.use_stop_words": "<use_stop_words>",
        "mode.specific_stop_words_threshold": "<specific_stop_words_threshold>",
        "mode.hex_encode": "<hex_encode>",
        "mode.use_categorization": "<use_categorization>",
        "mode.category_text_columns": "<category_text_columns>",
        "mode.test_size": "<test_size>",
        "mode.smote": "<smote>",
        "mode.undersample": "<undersample>",
        "mode.algorithm": "<algorithm>",
        "mode.preprocessor": "<preprocessor>",
        "mode.feature_selection": "<feature_selection>",
        "mode.num_selected_features": "<num_selected_features>",
        "mode.scoring": "<scoring>",
        "mode.max_iterations": "<max_iterations>",
        "io.verbose": "<verbose>",
        "io.model_path": "<model_path>",
        "io.model_name": "<model_name>",
        "debug.on": "<on>",
        "debug.data_limit": "<data_limit>"
    }

    @dataclass
    class Connection:
        odbc_driver: str = "ODBC Driver 17 for SQL Server"
        host: str = ""
        trusted_connection: bool = True
        class_catalog: str = ""
        class_table: str = ""
        class_table_script: str = "./sql/autoClassCreateTable.sql.txt"
        class_username: str = ""
        class_password: str = ""
        data_catalog: str = ""
        data_table: str = ""
        class_column: str = ""
        data_text_columns: list = field(default_factory=list)
        data_numerical_columns: list = field(default_factory=list)
        id_column: str = "id"
        data_username: str = ""
        data_password: str = ""

        def validate(self) -> None:
            """ Throws TypeError if invalid """
            # database connection information
            database_connection_information = [
                isinstance(self.host, str),
                isinstance(self.class_catalog, str),
                isinstance(self.class_table, str),
                isinstance(self.class_table_script, str),
                isinstance(self.class_username, str),
                isinstance(self.class_password, str),
                isinstance(self.data_catalog, str),
                isinstance(self.data_table, str),
                isinstance(self.class_column, str),
                isinstance(self.data_text_columns, list),
                isinstance(self.data_numerical_columns, list),
                isinstance(self.id_column, str),
                isinstance(self.trusted_connection, bool)
            ]
            
            if not all(database_connection_information):
                raise TypeError(
                    "Specified database connection information is invalid")

            if not all(isinstance(x,str) for x in self.data_text_columns):
                raise TypeError("Data text columns needs to be a list of strings")
            if not all(isinstance(x,str) for x in self.data_numerical_columns):
                raise TypeError("Data numerical columns needs to be a list of strings")

            
            # Login credentials
            login_credentials = [
                isinstance(self.data_username, str),
                isinstance(self.data_password, str)
            ]

            if not all(login_credentials):
                raise TypeError("Specified login credentials are invalid!")

            # Overriding values
            if self.trusted_connection:
                self.data_password = ""
                self.class_password = ""

                username = Config.get_username()
                self.class_username = username
                self.data_username = username

        def update_columns(self, updated_columns: dict) -> dict:
            """ Given the dictionary updated_columns set attributes to values """
            errors = []
            for attribute, value in updated_columns.items():
                if hasattr(self, attribute):
                    setattr(self, attribute, value)
                else:
                    errors.append(attribute)
            
            if errors:
                return {
                    "response": f"Following columns do not exist: {', '.join(errors)}",
                    "success": False
                }

            return {
                "response": "Columns updated",
                "success": True
            }

        def driver_is_implemented(self) -> bool:
            """ Returns whether the driver of the config is implemented """
            # TODO: Probably want to make this sturdier, but that would require
            # rewriting the dependancy on odbc_driver and the str
            if self.odbc_driver in ["SQL Server", "Mock Server"]:
                return True
            
            return False

        def _get_formatted_catalog(self, type) -> str:
            """ Gets a class table formatted for the type, which is based on odbc_driver
            """
            if not self.driver_is_implemented():
                return ""

            catalog_attr = type + "_catalog"
            catalog = getattr(self, catalog_attr)
            return f"[{ period_to_brackets(catalog) }]"

        def get_formatted_class_catalog(self) -> str:
            """ Gets a class table formatted for the type, which is based on odbc_driver
            """
            return self._get_formatted_catalog("class")

        def get_formatted_data_catalog(self) -> str:
            """ Gets a data table formatted for the type, which is based on odbc_driver
            """
            return self._get_formatted_catalog("data")

        def _get_formatted_table(self, type: str, include_database: bool = True) -> str:
            """ Gets a class table formatted for the type, which is based on odbc_driver
            """
            if not self.driver_is_implemented():
                return ""

            table_attr = type + "_table"
            table = getattr(self, table_attr)
            formatted_table = f"[{ period_to_brackets(table) }]"

            if not include_database:
                return formatted_table

            formatted_catalog = self._get_formatted_catalog(type)

            return f"{formatted_catalog}.{formatted_table}"

        def get_formatted_class_table(self, include_database: bool = True) -> str:
            """ Gets the class table as a formatted string for the correct driver
                In the type of [schema].[catalog].[table]
            """
            return self._get_formatted_table("class", include_database)

        def get_formatted_data_table(self, include_database: bool = True) -> str:
            """ Gets the data table as a formatted string for the correct driver
                In the type of [schema].[catalog].[table]
            """
            return self._get_formatted_table("data", include_database)

            
        def get_catalog_params(self, type) -> dict:
            """ Gets params to connect to a database """
            params = {
                "driver": self.odbc_driver,
                "host": self.host,
                "catalog": "",
                "trusted_connection": self.trusted_connection,
                "username": "",
                "password": ""
            }

            if type == "class":
                params["catalog"] = self.class_catalog
                params["username"] = self.class_username
                params["password"] = self.class_password
            elif type == "data":
                params["catalog"] = self.data_catalog
                params["username"] = self.data_username
                params["password"] = self.data_password
            else:
                raise ConfigException(f"Type {type} not acceptable as a connection type")
            
            return params

        def __str__(self) -> str:
            str_list = [
                " 1. Database settings ",
                f" * ODBC driver (when applicable):           {self.odbc_driver}",
                f" * Classification Host:                     {self.host}",
                f" * Trusted connection:                      {self.trusted_connection}",
                f" * Classification Table:                    {self.class_catalog}",
                f" * Classification Table:                    {self.class_table}",
                f" * Classification Table creation script:    {self.class_table_script}",
                f" * Classification Db username (optional):   {self.class_username}",
                f" * Classification Db password (optional)    {self.class_password}\n",
                f" * Data Catalog:                            {self.data_catalog}",
                f" * Data Table:                              {self.data_table}",
                f" * Classification column:                   {self.class_column}",
                f" * Text Data columns (CSV):                 {', '.join(self.data_text_columns)}",
                f" * Numerical Data columns (CSV):            {', '.join(self.data_numerical_columns)}",
                f" * Unique data id column:                   {self.id_column}",
                f" * Data username (optional):                {self.data_username}",
                f" * Data password (optional):                {self.data_password}",
            ]
            
            return "\n".join(str_list)

    @dataclass
    class IO:
        verbose: bool = True
        model_path: str = "./model/"
        model_name: str = "iris"

        def validate(self) -> None:
            """ Throws TypeError if invalid """
        
            for item in [
                "verbose",
            ]:
                if not isinstance(getattr(self, item), bool):
                    raise TypeError(f"Argument {item} must be True or False")

        
        def __str__(self) -> str:
            str_list = [
                " 3. I/O specifications ",
                f" * Verbosity:                               {self.verbose}",
                f" * Path where to save generated model:      {self.model_path}",
                f" * Name of generated or loaded model:       {self.model_name}"
            ]

            return "\n".join(str_list)        

    @dataclass
    class Mode:
        train: bool = True
        predict: bool = True
        mispredicted: bool = True
        use_metas: bool = True
        use_stop_words: bool = True
        specific_stop_words_threshold: float = 1.0
        hex_encode: bool = True
        use_categorization: bool = True
        category_text_columns: list = field(default_factory=list)
        test_size: float = 0.2
        smote: bool = False
        undersample: bool = False
        algorithm: AlgorithmTuple = AlgorithmTuple([Algorithm.LDA])
        preprocessor: PreprocessTuple = PreprocessTuple([Preprocess.NOS])
        feature_selection: ReductionTuple = ReductionTuple([Reduction.NOR])
        num_selected_features: int = None
        scoring: ScoreMetric = ScoreMetric.accuracy
        max_iterations: int = None

        def validate(self) -> None:
            """ Throws TypeError if invalid """

            if not isinstance(self.category_text_columns, list):
                raise TypeError(f"Argument category_text_columns must be a list of strings")

            if not all(isinstance(x,str) for x in self.category_text_columns):
                raise TypeError(f"Argument category_text_columns must be a list of strings")
            
            if not Helpers.positive_int_or_none(self.num_selected_features):
                raise ValueError(
                    "Argument num_selected_features must be a positive integer")

            if self.max_iterations is None:
                self.max_iterations = Config.MAX_ITERATIONS
            elif not Helpers.positive_int_or_none(self.max_iterations):
                raise ValueError(
                    "Argument max_iterations must be a positive integer")

            # Type checking + at least one is True
            mode_types = [
                isinstance(self.train, bool),
                isinstance(self.predict, bool),
                isinstance(self.mispredicted, bool),
                isinstance(self.use_metas, bool),
                (self.train or self.predict)
            ]
            
            if not all(mode_types):
                raise ValueError(
                    "Class must be set for either training, predictions and/or mispredictions!")

            if self.mispredicted and not self.train:
                raise ValueError(
                    "Class must be set for training if it is set for misprediction")

            # Stop words threshold and test size
            if isinstance(self.specific_stop_words_threshold, float):
                if self.specific_stop_words_threshold > 1.0 or self.specific_stop_words_threshold < 0.0:
                    raise ValueError(
                        "Argument specific_stop_words_threshold must be between 0 and 1!")
            else:
                raise TypeError(
                    "Argument specific_stop_words_threshold must be a float between 0 and 1!")

            
            if isinstance(self.test_size, float):
                if self.test_size > 1.0 or self.test_size < 0.0:
                    raise ValueError(
                        "Argument test_size must be between 0 and 1!")
            else:
                raise TypeError(
                    "Argument test_size must be a float between 0 and 1!")

            if not (isinstance(self.algorithm, AlgorithmTuple)):
                raise TypeError("Argument algorithm is invalid")

            if not (isinstance(self.preprocessor, PreprocessTuple)):
                raise TypeError("Argument preprocessor is invalid")

            if not (isinstance(self.feature_selection, ReductionTuple)):
                raise TypeError(f"Argument feature_selection is invalid: {str(self.feature_selection)}")

            for item in [
                "use_stop_words",
                "hex_encode",
                "smote",
                "undersample"
            ]:
                if not isinstance(getattr(self, item), bool):
                    raise TypeError(f"Argument {item} must be True or False")

        def __str__(self) -> str:
            str_list = [
                " 2. Classification mode settings ",
                f" * Train new model:                         {self.train}",
                f" * Make predictions with model:             {self.predict}",
                f" * Display mispredicted training data:      {self.mispredicted}",
                f" * Pass on meta data for predictions:       {self.mispredicted}",
                f" * Use stop words:                          {self.use_stop_words}",
                f" * Material specific stop words threshold:  {self.specific_stop_words_threshold}",
                f" * Hex encode text data:                    {self.hex_encode}",
                f" * Categorize text data where applicable:   {self.use_categorization}",
                f" * Force categorization to these columns:   {', '.join(self.category_text_columns)}",
                f" * Test size for trainings:                 {self.test_size}",
                f" * Use SMOTE:                               {self.smote}",
                f" * Use undersampling of majority class:     {self.undersample}",
                f" * Algorithms of choice:                    {self.algorithm.get_full_name()}",
                f" * Preprocessing method of choice:          {self.preprocessor.get_full_name()}",
                f" * Scoring method:                          {self.scoring.get_full_name()}",
                f" * Feature selection:                       {self.feature_selection.get_full_name()}",
                f" * Number of selected features:             {self.num_selected_features}",
                f" * Maximum iterations (where applicable):   {self.max_iterations}"
            ]

            return "\n".join(str_list)

    @dataclass
    class Debug:
        on: bool = True
        data_limit: int = None

        def validate(self) -> None:
            """ Throws TypeError if invalid """

            # TODO: Set the value based on count_data_rows(), but first decide where that method should be
            # The method is (and should probably stay) in DataLayer--question is where we set this
            if not Helpers.positive_int_or_none(self.data_limit):
                raise ValueError(
                    "Argument data_limit must be a positive integer")

        def __str__(self) -> str:
            str_list = [
                " 4. Debug settings  ",
                f" * Debugging on:                            {self.on}",
                f" * How many data rows to consider:          {self.data_limit}"
            ]

            return "\n".join(str_list)   
    
    connection: Connection = field(default_factory=Connection)
    mode: Mode  = field(default_factory=Mode)
    io: IO = field(default_factory=IO)
    debug: Debug = field(default_factory=Debug)
    name: str = "iris"
    config_path: Path = None
    script_path: Path = None
    _filename: str = None
    save: bool = False
    
    @property
    def filename(self) -> str:
        if self._filename is None:
            return f"{self.CONFIG_FILENAME_START}{self.name}_{self.connection.data_username}.py"

        return self._filename
    
    @property
    def filepath(self) -> Path:
        return self.config_path / self.filename

    def __post_init__(self) -> None:
        pwd = os.path.dirname(os.path.realpath(__file__))
        
        if self.config_path is None:
            self.config_path = Path(pwd) / "./config/"

        if self.script_path is None:
            self.script_path = Path(pwd)

        
        """Post init is called after init, which is the best place to check the types & values"""

        # 1: Top config params
        if not isinstance(self.name, str):
            raise TypeError(f"Argument name must be a string")

        # 2: Connection params
        self.connection.validate()

        # 3: Mode/training
        self.mode.validate()
        
        # 4: IO
        self.io.validate()

        # 5: Debug
        self.debug.validate()

        # This is True if training in GUI, always False if not
        if self.save:
            self.save_to_file()

    def __str__(self) -> str:
        str_list = [
            " -- Configuration settings --",
            str(self.connection) + "\n",
            str(self.mode) + "\n",
            str(self.io) + "\n",
            str(self.debug)
        ]
        return  "\n".join(str_list)

    @classmethod
    def get_username(self) -> str:
        """ calculates username """

        # TODO: Change to not use os.getlogin()
        return os.getlogin()
    
    def update_connection_columns(self, updated_columns = dict) -> dict:
        """ Wrapper function to not show inner workings """
        return self.connection.update_columns(updated_columns)

    def get_model_filename(self, pwd: Path = None) -> str:
        """ Set the name and path of the model file
            The second parameter allows for injecting the path for reliable testing
        """
        if pwd is None:
            pwd = self.script_path
        
        model_path = pwd / Path(self.io.model_path)
        
        return model_path / (self.io.model_name + Config.DEFAULT_MODEL_EXTENSION)

    # Extracts the config information to save with a model
    def get_clean_config(self):
        configuration = Config()
        configuration.connection = copy.deepcopy(self.connection)
        configuration.connection.data_catalog = ""
        configuration.connection.data_table = ""
        configuration.mode = copy.deepcopy(self.mode)
        configuration.mode.train = None
        configuration.mode.predict = None
        configuration.mode.mispredicted = None
        configuration.io = copy.deepcopy(self.io)
        configuration.io.model_name = ""
        configuration.debug = copy.deepcopy(self.debug)
        configuration.debug.data_limit = 0
        configuration.save = False
        
        return configuration

    # Saves config to be read from the command line
    def save_to_file(self, filepath: Path = None, username: str = None) -> None:
        template_path = self.config_path / self.CONFIG_SAMPLE_FILE
        
        with open(template_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
        
        for tag in self.TEMPLATE_TAGS:
            #print(tag)
            template = self.TEMPLATE_TAGS[tag]
            location = tag.split(".")
            
            if (len(location) == 1):
                replace = getattr(self, location[0])
            else:
                head = getattr(self, location[0])
                replace = getattr(head, location[1])

                # Exception for class/data username, if given
                if username is not None and "username" in location[1]:
                    replace = username

            # Check if it's one of the enum variables
            if (isinstance(replace, enum.Enum)):
                replace = replace.name

            # Replace the lists with text representations
            if (isinstance(replace, list)):
                replace = ",".join(replace)

            for i in range(len(lines)):
                lines[i] = lines[i].replace(template, str(replace))

        # This fixes the issue with debug/num_rows
        if hasattr(self.debug, "num_rows"):
            print("woo?")
       
        filepath = filepath if filepath else self.filepath
        
        with open(filepath, "w", encoding="utf-8") as fout:
           fout.writelines(lines)

    @classmethod
    def load_config_from_model_file(cls: Type[T], filename: str, config: T = None) -> T:
        try:
            file_values = pickle.load(open(filename, 'rb'))
            saved_config = file_values[0]
        except Exception as e:
            raise ConfigException(f"Something went wrong on loading model from file: {e}")
        
        if config is not None:
            saved_config.mode.train = config.mode.train
            saved_config.mode.predict = config.mode.predict
            saved_config.mode.mispredicted = config.mode.mispredicted
            saved_config.mode.use_metas = config.mode.use_metas
            saved_config.connection.data_catalog = config.connection.data_catalog
            saved_config.connection.data_table = config.connection.data_table
            saved_config.io.model_name = config.io.model_name
            saved_config.debug.data_limit = config.debug.data_limit
        
        return saved_config


    @classmethod
    def load_config_from_module(cls: Type[T], argv) -> T:
        module = Helpers.check_input_arguments(argv)
        
        version = '1.0'
        if (hasattr(module, 'version')):
            version = module.version

        if (version == '1.0'):
            raise ConfigException("V1.0 config is deprecated")
        
        return Config.load_config(module)

    @classmethod
    def load_config(cls: Type[T], module) -> T:
        if "num_rows" in module.debug:
            data_limit = Helpers.set_none_or_int(module.debug["num_rows"])
        else:
            data_limit = Helpers.set_none_or_int(module.debug["data_limit"])
        num_selected_features = Helpers.set_none_or_int(module.mode["num_selected_features"])
        max_iterations = Helpers.set_none_or_int(module.mode["max_iterations"])
        data_text_columns = Helpers.get_from_string_or_list(module.connection["data_text_columns"])
        data_numerical_columns = Helpers.get_from_string_or_list( module.connection["data_numerical_columns"])
        category_text_columns = Helpers.get_from_string_or_list(module.mode["category_text_columns"])
        use_metas = module.mode["predict"]
        if "use_metas" in module.mode:
            use_metas = module.mode["use_metas"]
       
        config = cls(
            Config.Connection(
                odbc_driver=module.connection["odbc_driver"],
                host=module.connection["host"],
                trusted_connection=module.connection["trusted_connection"],
                class_catalog=module.connection["class_catalog"],
                class_table=module.connection["class_table"],
                class_table_script=module.connection["class_table_script"],
                class_username=module.connection["class_username"],
                class_password=module.connection["class_password"],
                data_catalog=module.connection["data_catalog"],
                data_table=module.connection["data_table"],
                class_column=module.connection["class_column"],
                data_text_columns=data_text_columns,
                data_numerical_columns=data_numerical_columns,
                id_column=module.connection["id_column"],
                data_username=module.connection["data_username"],
                data_password=module.connection["data_password"]
            ),
            Config.Mode(
                train=module.mode["train"],
                predict=module.mode["predict"],
                mispredicted=module.mode["mispredicted"],
                use_metas=use_metas,
                use_stop_words=module.mode["use_stop_words"],
                specific_stop_words_threshold=float(
                    module.mode["specific_stop_words_threshold"]),
                hex_encode=module.mode["hex_encode"],
                use_categorization=module.mode["use_categorization"],
                category_text_columns=category_text_columns,
                test_size=float(module.mode["test_size"]),
                smote=module.mode["smote"],
                undersample=module.mode["undersample"],
                algorithm=AlgorithmTuple(module.mode["algorithm"].split(",")),
                preprocessor=PreprocessTuple(module.mode["preprocessor"].split(",")),
                feature_selection=ReductionTuple(module.mode["feature_selection"].split(",")),
                num_selected_features=num_selected_features,
                scoring=ScoreMetric[module.mode["scoring"]],
                max_iterations=max_iterations
            ),
            Config.IO(
                verbose=module.io["verbose"],
                model_path=module.io["model_path"],
                model_name=module.io["model_name"]
            ),
            Config.Debug(
                on=module.debug["on"],
                data_limit=data_limit
            ),
            name=module.name
        )

        return config
        
    
    # Methods to hide implementation of Config
    def update_configuration(self, updates: dict) -> bool:
        """ Updates the config with new, wholesale, bits """
        # TODO: Break out validation to be able to call that here as well
        for key, item in updates.items():
            if not hasattr(self, key):
                raise ConfigException(f"Key {key} does not exist in Config")

            setattr(self, key, item)

        self.__post_init__()


    def is_text_data(self) -> bool:
        return len(self.connection.data_text_columns) > 0
    
    def is_numerical_data(self) -> bool:
        return len(self.connection.data_numerical_columns) > 0

    def force_categorization(self) -> bool:
        return  len(self.mode.category_text_columns) > 0

    def column_is_numeric(self, column: str) -> bool:
        """ Checks if the column is numerical """
        return column in self.get_numerical_column_names()

    def column_is_text(self, column: str) -> bool:
        """ Checks if the column is text based """
        return column in self.get_text_column_names()

    def use_imb_pipeline(self) -> bool:
        """ Returns True if either smote or undersampler should is True """
        if not self.mode.smote and not self.mode.undersample:
            return False

        return True

    def get_classification_script_path(self) -> Path:
        """ Gives a calculated path based on config"""
        
        return self.script_path / self.connection.class_table_script
    
    def get_feature_selection(self) -> Reduction:
        """ Gets the given feature selection Reduction """
        return self.mode.feature_selection

    def get_none_or_positive_value(self, attribute: str) -> int:
        value = self.get_attribute(attribute)
        
        if value is None or value == "":
            return 0

        return value

    def get_attribute(self, attribute: str):
        """ Gets an attribute from a attribute.subattribute string """
        location = attribute.split(".")
        length = len(location)
        if length > 2:
            raise ConfigException(f"Invalid format {attribute}")

        if length == 1:
            try:
                value = getattr(self, attribute)
            except AttributeError:
                raise ConfigException(f"There is no attribute {attribute}")

            return value

        try:
            head = getattr(self, location[0])
            value = getattr(head, location[1])
        except AttributeError:
            raise ConfigException(f"There is no attribute {attribute}")

        return value    
    
    def get_connection(self) -> Connection:
        return self.connection

    def get_quoted_attribute(self, attribute: str, quotes: str = "\'") -> str:
        """ Gets an attribute as per get_attribute, returns it in quotation marks """

        return to_quoted_string(self.get_attribute(attribute), quotes)
    
    @staticmethod
    def get_model_name(model: str, project_name: str) -> str:
        if model == Config.DEFAULT_TRAIN_OPTION:
            return project_name
        
        return model.replace(Config.DEFAULT_MODEL_EXTENSION, "")

    def get_num_selected_features(self) -> int:
        return self.get_none_or_positive_value("mode.num_selected_features")
        
    def feature_selection_in(self, selection: list[Reduction]) -> bool:
        """ Checks if the selection is one of the given Reductions"""
        for reduction in selection:
            if self.mode.feature_selection == reduction:
                return True

        return False

    def use_feature_selection(self) -> bool:
        """ Checks if feature selection should be used """
        return self.mode.feature_selection is not ReductionTuple([Reduction.NOR])

    def get_test_size(self) -> float:
        """ Gets the test_size """
        return self.mode.test_size

    def get_test_size_percentage(self) -> int:
        """ Gets the test_size as a percentage """
        return int(self.mode.test_size * 100.0)

    def get_max_limit(self) -> int:
        """ Get the max limit. Name might change depending on GUI names"""
        return self.get_none_or_positive_value("debug.data_limit")

    def get_max_iterations(self) -> int:
        """ Get max iterations """
        return self.get_none_or_positive_value("mode.max_iterations")

    def is_verbose(self) -> bool:
        """ Returns what the io.verbose is set to"""
        return self.io.verbose
        
    def get_column_names(self) -> list[str]:
        """ Gets the column names based on connection columns """
        columns = self.get_data_column_names()

        if id_column := self.get_id_column_name():
            columns.append(id_column)

        if class_column := self.get_class_column_name():
            columns.append(class_column)
        
        return columns
        

    def get_categorical_text_column_names(self) -> list[str]:
        """ Gets the specified categorical text columns"""
        return self.mode.category_text_columns
        
        
    def get_text_column_names(self) -> list[str]:
        """ Gets the specified text columns"""
        return self.connection.data_text_columns


    def get_numerical_column_names(self) -> list[str]:
        """ Gets the specified numerical columns"""
        return self.connection.data_numerical_columns


    def get_data_column_names(self) -> list[str]:
        """ Gets data columns, so not Class or ID """
        return self.get_text_column_names() + self.get_numerical_column_names()
       

    def get_class_column_name(self) -> str:
        """ Gets the name of the column"""
        return self.connection.class_column

    def get_id_column_name(self) -> str:
        """ Gets the name of the ID column"""
        return self.connection.id_column

    def is_categorical(self, column_name: str) -> bool:
        """ Returns if a specific column is categorical"""
        return self.force_categorization() and column_name in self.get_categorical_text_column_names()

    def should_train(self) -> bool:
        """ Returns if this is a training config """
        return self.mode.train

    def should_predict(self) -> bool:
        """ Returns if this is a prediction config """
        return self.mode.predict

    def should_display_mispredicted(self) -> bool:
        """ Returns if this is a misprediction config """
        return self.mode.mispredicted

    def should_use_metas(self) -> bool:
        """ Returns if this is a use metas config """
        return self.mode.use_metas

    def use_stop_words(self) -> bool:
        """ Returns whether stop words should be used """
        return self.mode.use_stop_words

    def get_stop_words_threshold(self) -> float:
        """ Returns the threshold for the stop words """
        return self.mode.specific_stop_words_threshold

    def get_stop_words_threshold_percentage(self) -> int:
        """ Returns the threshold as an integer between 0-100"""
        return int(self.mode.specific_stop_words_threshold * 100.0)

    def should_hex_encode(self) -> bool:
        """ Returns whether dataset should be hex encoded """
        return self.mode.hex_encode

    def use_categorization(self) -> bool:
        """ Returns if categorization should be used """
        return self.mode.use_categorization

    def get_smote(self) -> Union[SMOTE, None]:
        """ Gets the SMOTE for the model, or None if it shouldn't be """
        if self.mode.smote:
            return SMOTE(sampling_strategy='auto')

        return None
    
    def use_smote(self) -> bool:
        """ Simple check if it's used or note """
        return self.mode.smote
    
    def get_undersampler(self) -> Union[RandomUnderSampler, None]:
        """ Gets the UnderSampler, or None if there should be none"""
        if self.mode.undersample:
            return RandomUnderSampler(sampling_strategy='auto')

        return None

    def use_undersample(self) -> bool:
        """ Simple check if it's used or note """
        return self.mode.undersample
    
    
    def set_num_selected_features(self, num_features: int) -> None:
        """ Updates the config with the number """
        self.mode.num_selected_features = num_features
    

    def update_attribute(self, attribute: Union[str, dict], new_value) -> None:
        """ attribute is either a string on the form 'name.subname' 
        or a dict on the form { "name" : "subname", "type": "name"}
        """
        
        if isinstance(attribute, dict):
            try:
                head = getattr(self, attribute["type"])
                setattr(head, attribute["name"], new_value)
            except KeyError:
                raise ConfigException(f"Incorrect attribute format { attribute }")
            except AttributeError:
                raise ConfigException(f"There is no attribute { attribute }")

            return

        location = attribute.split(".")

        if len(location) > 2:
            raise ConfigException(f"There is no attribute {attribute}")
        try:
            if len(location) == 1:
                if hasattr(self, attribute):
                    setattr(self, attribute, new_value)
                else:
                    raise ConfigException(f"There is no attribute {attribute}")
            else:
                head = getattr(self, location[0])
                if hasattr(head, location[1]):
                    setattr(head, location[1], new_value)
                else:
                    raise ConfigException(f"There is no attribute {attribute}")
                
        except AttributeError:
            raise ConfigException(f"There is no attribute {attribute}")

    def update_attributes(self, updates: dict,  type: str = None) -> None:
        """ Updates several attributes inside the config """
        try:
            if type is None:
                """ Attempts to find the fields based on a split, useful if values belong to different parts """
                for attribute, value in updates.items():
                    self.update_attribute(attribute, value)
            else:
                attribute_dict = {"type": type}
                for attribute, value in updates.items():
                    attribute_dict["name"] = attribute
                    self.update_attribute(attribute_dict, value)
        except ConfigException:
            raise

    def get_scoring_mechanism(self)  -> Union[str, Callable]:
        """ While the actual function is in the mechanism, this allows us to hide where Scoring is """
        return self.mode.scoring.get_mechanism()

    def get_algorithm(self) -> Algorithm:
        """ Get algorithm from Config"""
        return Algorithm(self.mode.algorithm)

    def get_preprocessor(self) -> Preprocess:
        """ get preprocessor from Config """
        return Preprocess(self.mode.preprocessor)
    
    def get_class_table(self) -> str:
        """ Gets the class table with database """
        return f"{self.connection.class_catalog}.{self.connection.class_table}"
    
    def get_data_catalog(self) -> str:
        """ Gets the data catalog """
        return self.connection.data_catalog

    def get_data_table(self) -> str:
        """ Gets the data table """
        return self.connection.data_table

    def get_data_username(self) -> str:
        """ Gets the data user name """
        return self.connection.data_username

    def get_class_catalog_params(self) -> dict:
        """ Gets params to connect to class database """
        return self.connection.get_catalog_params("class")

    def get_data_catalog_params(self) -> dict:
        """ Gets params to connect to data database """
        return self.connection.get_catalog_params("data")


def to_quoted_string(x, quotes:str = '\'') -> str:
        value = str(x)

        return quotes + value + quotes

def period_to_brackets(string: str) -> str:
    """ Takes a string and replaces . with ].[
        Used with tables, schemas and databases    
    """

    return string.replace(".", "].[")


def main():
    if len(sys.argv) > 1:
        config = Config.load_config_from_module(sys.argv)
    else:
       config = Config()

    updates = {
        "debug": Config.Debug(
                on=True,
                data_limit=125
            ),
    }
    print(config.debug)
    config.update_configuration(updates)
    
if __name__ == "__main__":
    main()