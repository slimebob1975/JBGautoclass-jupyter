from __future__ import annotations
import enum
from typing import Callable, Iterable, Protocol, Union

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

from imblearn.ensemble import (EasyEnsembleClassifier, RUSBoostClassifier, 
     BalancedBaggingClassifier, BalancedRandomForestClassifier)

from JBGExperimental import (JBGRobustLogisticRegression, JBGRobustCentroid, 
                            JBGPartitioningDetector, JBGMCS, JBGInstanceHardness,
                            JBGRandomForestDetector)

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

META_ENUM_DEFAULT_TERMS = ("ALL", "DUMY", "NOR", "NOS")

class MetaEnum(enum.Enum):
    @property
    def full_name(self):
        if isinstance(self.value, dict):
            return self.value["full_name"]
        
        return self.value
    
    @classmethod
    def get_sorted_list(cls, default_terms_first: bool = True) -> list[tuple[str, str]]:
        if not default_terms_first:
            return sorted([(item.full_name, item.name) for item in cls])

        excluded_enums = []
        for name in META_ENUM_DEFAULT_TERMS:
            try:
                item = cls[name]
            except KeyError:
                """ Left blank on purpose """
            else:
                excluded_enums.insert(0, item)
        
        listed_enums = sorted([(item.full_name, item.name) for item in cls if item not in excluded_enums])

        default_list = [(item.full_name, item.name) for item in excluded_enums]
        
        return default_list + listed_enums

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

    def __eq__(self, other: MetaEnum) -> bool:
        """
        This bases the comparisons on the name
        """
        if (type(self) == type(other)):
            return self.name == other.name
        
        return False

        # TODO: Make a less naive "equals" comparison
        # Att jämföra två Algorithm objekt bör ske i två steg. 
        # Att det är samma typ av objekt, dvs samma förkortning 
        # eller typ på objektet. Vill man gå vidare kan man 
        # plocka ut alla parameterar via get_params och jämföra 
        # att de är exakt lika också

    def __lt__(self, other: MetaEnum) -> bool:
        """
        This bases the comparisons on the name
        """
        return self.name < other.name

        # TODO: Implement so that default values (all/none/dumy/etc) are first

    def __gt__(self, other: MetaEnum) -> bool:
        """
        This bases the comparisons on the name
        """
        return self.name > other.name

    def __le__(self, other: MetaEnum) -> bool:
        """
        This bases the comparisons on the name
        """
        return self.name <= other.name

    def __ge__(self, other: MetaEnum) -> bool:
        """
        This bases the comparisons on the name
        """
        return self.name >= other.name

class MetaTuple:
    """ Takes a list of MetaEnums and turns it into a Tuple"""

    @property
    def type(self) -> MetaEnum:
        if hasattr(self, "_type"):
            return self._type
        
        return MetaEnum

    @property
    def full_name(self):
        return self.get_full_names()


    def __init__(self, initiating_values: Iterable) -> None:
        
        if isinstance(initiating_values, Iterable):
            metaEnums = []
            for metaEnum in initiating_values:
                if isinstance(metaEnum, self.type):
                    metaEnums.append(metaEnum)
                elif isinstance(metaEnum, str):
                    metaEnums.append(self.type[metaEnum])
                else:
                    raise ValueError(f"Each element in input list must be an instance of {self.type.__name__}")
            self.metaEnums = tuple(metaEnums)
        else:
            raise ValueError(f"Input to {self.__class__.__name__} must be an iterable")


    def __eq__(self, other: MetaTuple) -> bool:
        """
        This compares that the list of metaEnums contains the same metaEnums, does not care about order
        """
        return sorted(self.metaEnums) == sorted(other.metaEnums)
        

    def __str__(self) -> str:
        """ 
        Returns a comma-separated list of the abbreviations
        The list is the same order as the values were initiated in
        """
        return ", ".join([x.name for x in self.metaEnums])

    def get_full_names(self) -> str:
        """ 
        Returns a comma-separated list of the full names
        The list is the same order as the values were initiated in
        """
        return ", ".join([x.full_name for x in self.metaEnums])
    
    def get_full_names_and_abbrevs(self) -> str:
        """ 
        Returns a comma-separated list of the full names + abbreviations
        The list is the same order as the values were initiated in
        """
        list = [f"{x.full_name} ({x})" for x in self.metaEnums]
        return ", ".join(list)


    def list_callables(self, do_or_get: str, **kwargs) -> list[tuple]:
        """ Gets a list of metaEnums that are callable
            in the form (metaEnum, called function)
        """
        # TODO: Semantically speaking, it should be "get" for all MetaEnum
        # Tuples, but that requires a larger operation to rewrite the function
        # names in the MetaEnums. Needs done, but not right now
        metaEnums = [(
            metaEnum,
            metaEnum.call_function(do_or_get, **kwargs)) 
                for metaEnum in self.metaEnums if metaEnum.has_function(do_or_get)
            ]
        metaEnums.sort(key=lambda enumtuple: enumtuple[0].name)
        return metaEnums    



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
    REC = {"parameters": {'alpha': [1.0, 0.1, 0.01, 0.001, 0.0001], 'tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], "fit_intercept": [True, False], 
           'class_weight': ('balanced', None)}}
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
    SVC = {"parameters": {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto'], 'max_iter': [-1],
                          'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'class_weight': ('balanced', None)}}
    STCL = {"parameters": {}}
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
    STCL = { "full_name": "Self Training Classifier", "limit": 10000, "search_params": AlgorithmGridSearchParams.STCL, "rfe_compatible": False}
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

    def do_STCL(self, max_iterations: int, size: int) -> SelfTrainingClassifier:
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

class Preprocess(MetaEnum):
    NOS = "No Scaling"
    STA = "Standard Scaler"
    MIX = "Min-Max Scaler"
    MAX = "Max-Absolute Scaler"
    NRM = "Normalizer"
    BIN = "Binarizer"

    def get_full_name(self) -> str:
        return self.full_name

    @classmethod
    def list_callable_preprocessors(cls) -> list[tuple]:
        """ Gets a list of preprocessors that are callable (including NOS -> None)
            in the form (preprocessor, called function)
        """
        return [(pp, pp.call_preprocess()) for pp in cls if pp.has_function() ]

    def do_NOS(self) -> NonScaler:
        """ While this return is superfluos, it helps with the listings of preprocessors """
        return self.NonScaler()

    def NonScaler(self):
        return FunctionTransformer(lambda X: X)

    def do_STA(self) -> StandardScaler:
        return StandardScaler(with_mean=False)

    def do_MIX(self) -> MinMaxScaler:
        return MinMaxScaler()

    def do_MAX(self) -> MaxAbsScaler:
        return MaxAbsScaler()

    def do_NRM(self) -> Normalizer:
        return Normalizer()

    def do_BIN(self) -> Binarizer:
        return Binarizer()

    def call_preprocess(self) -> Union[Transform, None]:
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('do')

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

class AlgorithmTuple(MetaTuple):

    _type = Algorithm

    def list_callable_algorithms(self, size: int, max_iterations: int) -> list[tuple]:
        """ Gets a list of algorithms that are callable
            in the form (algorithm, called function)
        """
        return self.list_callables("do", max_iterations=max_iterations, size=size)

class PreprocessTuple(MetaTuple):

    _type = Preprocess

    def list_callable_preprocessors(self) -> list[tuple]:
        """ Gets a list of preprocessors that are callable (including NOS -> None)
            in the form (preprocessor, called function)
        """
        return self.list_callables("do")

    
class ReductionTuple(MetaTuple):

    _type = Reduction

    def list_callable_reductions(self, num_samples: int, num_features: int, \
        num_selected_features: int = None) -> list[tuple]:
        """ Gets a list of preprocessors that are callable (including NOS -> None)
            in the form (preprocessor, called function)
        """
        return self.list_callables(
            "get",
            num_samples=num_samples,
            num_features=num_features,
            num_selected_features=num_selected_features
        )