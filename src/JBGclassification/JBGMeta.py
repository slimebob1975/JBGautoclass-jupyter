from __future__ import annotations

import enum
from functools import total_ordering
from typing import Callable, Iterable, Protocol, Type, TypeVar, Union

import numpy as np
import pandas
from imblearn.ensemble import (BalancedBaggingClassifier,
                               BalancedRandomForestClassifier,
                               EasyEnsembleClassifier, RUSBoostClassifier)
from imblearn.over_sampling import (RandomOverSampler, SMOTE, SMOTEN,
                                    SMOTENC, ADASYN, BorderlineSMOTE,
                                    KMeansSMOTE, SVMSMOTE)
from imblearn.under_sampling import (ClusterCentroids, CondensedNearestNeighbour,
                                     EditedNearestNeighbours, AllKNN, 
                                     InstanceHardnessThreshold, NearMiss, 
                                     NeighbourhoodCleaningRule, OneSidedSelection,
                                     RandomUnderSampler, TomekLinks,
                                     RepeatedEditedNearestNeighbours)
from JBGTransformers import (JBGMCS, JBGInstanceHardness,
                             JBGPartitioningDetector, JBGRandomForestDetector,
                             JBGRobustCentroid, JBGRobustLogisticRegression,
                             NNClassifier3PL, MLPKerasClassifier)
from skclean.detectors import KDN, ForestKDN, RkDN
from skclean.handlers import CLNI, Costing, Filter, WeightedBagging
from skclean.models import RobustForest
from sklearn.decomposition import NMF, PCA, FastICA, TruncatedSVD
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, SGDClassifier)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import (roc_auc_score, accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, make_scorer,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.naive_bayes import (BernoulliNB, ComplementNB, GaussianNB,
                                 MultinomialNB)
from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid,
                               RadiusNeighborsClassifier)
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import (Binarizer, FunctionTransformer,
                                   MaxAbsScaler, MinMaxScaler, Normalizer,
                                   StandardScaler)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from Types import Detecter, Estimator, Transform

PCA_VARIANCE_EXPLAINED = 0.999
LOWER_LIMIT_REDUCTION = 100
NON_LINEAR_REDUCTION_COMPONENTS = 2

class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def print_components(self, component, components, exception = None) -> None:
        """ Printing Reduction components"""
        

class RateType(enum.Enum):
    # I = Individuell bedömning. Modellen kan ge sannolikheter för individuella dataelement.
    I = "Individual"
    # A = Allmän bedömning. Modellen kan endast ge gemensam genomsnittlig sannolikhet för alla dataelementen tillsammans.
    A = "General"
    # U = Okänd. Lade jag till som en framtida utväg ifall ingen av de ovanstående fanns tillgängliga.
    U = "Unknown"


META_ENUM_DEFAULT_TERMS = ("ALL", "DUMY", "NOR", "NOS", "NOG", "NUG")

@total_ordering
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

    def __lt__(self, other: MetaEnum) -> bool:
        """
        This bases the comparisons on the name
        """
        return self.name < other.name


T = TypeVar('T', bound='MetaTuple')

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

    @classmethod
    def from_string(cls: Type[T], commaseparated_list_as_string: str) -> T:
        """
        Creates a MetaEnum based on a (comma-separated) list in string format
        Strips spaces around each string value
        """
        return cls([x.strip() for x in commaseparated_list_as_string.split(",")])
    
    @classmethod
    def createTuple(cls, initiating_values: Iterable) -> None:
        """
        Overloaded initiation function
        """
        return cls(initiating_values)
    
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
        
    def get_abbreviations(self) -> list[str]:
        """ 
        Returns a tuple of the abbreviations
        The list is the same order as the values were initiated in
        """
        return [x.name for x in self.metaEnums]
    
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

class Library(MetaEnum):
    SCIKIT = "Scikit Learn"
    CLEAN = "Scikit Clean"
    IMBLRN = "Imbalanced Learn"
    TORCH = "PyTorch"
    KERAS = "Keras/Tensorflow"
    
    def get_full_name(self) -> str:
        return self.full_name

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
    GBC = {"parameters": {'loss': ['log_loss', 'exponential'], 'learning_rate': [0.01, 0.1, 0.2],
        'min_samples_split': np.linspace(0.1, 0.5, 12), 'min_samples_leaf': np.linspace(0.1, 0.5, 12),
        'max_depth':[3,5,8], 'max_features':['log2','sqrt'], 'criterion': ['friedman_mse', 'squared_error'],
        'subsample':[0.5, 0.8, 0.9, 1.0], 'n_estimators':[10]}}
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
    PYNN = {"parameters": {'activation': ('relu', 'tanh', 'sigmoid'), 'optimizer': ('adam', 'sgd'), \
                           'learning_rate': [0.01, 0.05, 0.1], 'max_epochs': [10, 30, 50], 'dropout_prob': [0.1, 0.3, 0.5], \
                           'num_hidden_layers': [2, 3], 'hidden_layer_size': [16, 48, 100], 'train_split': [True, False]}}
    KERA = {"parameters": {'verbose': [1], 'epochs': [10, 50, 100, 200], 'optimizer': ["adam", "rmsprop"], \
                           'optimizer__learning_rate': [0.001, 0.01, 0.1]}}
    #FUTV = {"parameters": \
    #    {"mlpc__" + str(key): val for key, val in MLPC["parameters"].items()} | \
    #    {"rfcl__" + str(key): val for key, val in RFCL["parameters"].items()}
    #    }
    FUTV = {"parameters": {}}

    #FUTS =  {"parameters": {'cv': (5, 10, 20)}  | \
    #    {"mlpc__" + str(key): val for key, val in MLPC["parameters"].items()} | \
    #    {"rfcl__" + str(key): val for key, val in RFCL["parameters"].items()} 
    #    }
    FUTS = {"parameters": {}}
    
    @property
    def parameters(self):
        if isinstance(self.value, dict):
            return self.value.get("parameters", {})
        
        return {}

class Algorithm(MetaEnum):
    DUMY = { "full_name": "Dummy Classifier", "search_params": AlgorithmGridSearchParams.DUMY, "rfe_compatible": False, "lib": Library.SCIKIT}
    SRF1 = { "full_name": "Stacked Random Forests 1", "search_params": AlgorithmGridSearchParams.SRF1, "rfe_compatible": False, "lib": Library.SCIKIT}
    SRF2 = { "full_name": "Stacked Random Forests 2", "search_params": AlgorithmGridSearchParams.SRF2, "rfe_compatible": False, "lib": Library.SCIKIT}
    BARF = { "full_name": "Balanced Random Forest", "search_params": AlgorithmGridSearchParams.BARF, "rfe_compatible": True, "lib": Library.IMBLRN}
    BABC = { "full_name": "Balanced Bagging Classifier", "search_params": AlgorithmGridSearchParams.BABC, "rfe_compatible": False, "lib": Library.IMBLRN}
    RUBC = { "full_name": "RUS Boost Classifier", "search_params": AlgorithmGridSearchParams.RUBC, "rfe_compatible": True, "lib": Library.IMBLRN}
    EAEC = { "full_name": "Easy Ensamble Classifier", "search_params": AlgorithmGridSearchParams.EAEC, "rfe_compatible": False, "lib": Library.IMBLRN}
    RTCL = { "full_name": "Robust Tree Classifier", "search_params": AlgorithmGridSearchParams.RTCL, "rfe_compatible": False, "lib": Library.CLEAN }
    RLRN = { "full_name": "Robust Logistic Regression + Label Encoder", "search_params": AlgorithmGridSearchParams.RLRN, "rfe_compatible": False, "lib": Library.CLEAN}
    RCNT = { "full_name": "Robust Centroid + Label Encoder", "search_params": AlgorithmGridSearchParams.RCNT, "rfe_compatible": False, "lib": Library.CLEAN}
    LRN = { "full_name": "Logistic Regression", "search_params": AlgorithmGridSearchParams.LRN, "rfe_compatible": True, "lib": Library.SCIKIT}
    KNN = { "full_name": "K-Neighbors Classifier", "search_params": AlgorithmGridSearchParams.KNN, "rfe_compatible": False, "lib": Library.SCIKIT}
    RADN = { "full_name": "Radius Neighbors Classifier", "search_params": AlgorithmGridSearchParams.RADN, "rfe_compatible": False, "lib": Library.SCIKIT}
    DTC = { "full_name": "Decision Tree Classifier", "search_params": AlgorithmGridSearchParams.DTC, "rfe_compatible": True, "lib": Library.SCIKIT}
    GNB = { "full_name": "Gaussian Naive Bayes", "search_params": AlgorithmGridSearchParams.GNB, "rfe_compatible": False, "lib": Library.SCIKIT}
    MNB = { "full_name": "Multinomial Naive Bayes", "search_params": AlgorithmGridSearchParams.MNB, "rfe_compatible": True, "lib": Library.SCIKIT}
    BNB = { "full_name": "Bernoulli Naive Bayes", "search_params": AlgorithmGridSearchParams.BNB, "rfe_compatible": True, "lib": Library.SCIKIT}
    CNB = { "full_name": "Complement Naive Bayes", "search_params": AlgorithmGridSearchParams.CNB, "rfe_compatible": True, "lib": Library.SCIKIT}
    REC = { "full_name": "Ridge Classifier", "search_params": AlgorithmGridSearchParams.REC, "rfe_compatible": True, "lib": Library.SCIKIT}
    PCN = { "full_name": "Perceptron", "search_params": AlgorithmGridSearchParams.PCN, "rfe_compatible": True, "lib": Library.SCIKIT}
    PAC = { "full_name": "Passive Aggressive Classifier", "search_params": AlgorithmGridSearchParams.PAC, "rfe_compatible": True, "lib": Library.SCIKIT}
    RFCL = { "full_name": "Random Forest Classifier", "search_params": AlgorithmGridSearchParams.RFCL, "rfe_compatible": True, "lib": Library.SCIKIT}
    LSVC = { "full_name":  "Linear Support Vector", "search_params": AlgorithmGridSearchParams.LSVC, "rfe_compatible": True, "lib": Library.SCIKIT}
    SLSV = { "full_name": "Stacked Linear SVC", "search_params": AlgorithmGridSearchParams.SLSV, "rfe_compatible": False, "lib": Library.SCIKIT}
    SGDE = { "full_name": "Stochastic Gradient Descent", "search_params": AlgorithmGridSearchParams.SGDE, "rfe_compatible": True, "lib": Library.SCIKIT}
    NCT = { "full_name": "Nearest Centroid", "search_params": AlgorithmGridSearchParams.NCT, "rfe_compatible": False, "lib": Library.SCIKIT}
    SVC = { "full_name": "Support Vector Classification", "limit": 10000, "search_params": AlgorithmGridSearchParams.SVC, "rfe_compatible": False, "lib": Library.SCIKIT}
    STCL = { "full_name": "Self Training Classifier", "limit": 10000, "search_params": AlgorithmGridSearchParams.STCL, "rfe_compatible": False, "lib": Library.SCIKIT}
    LDA = { "full_name": "Linear Discriminant Analysis", "search_params": AlgorithmGridSearchParams.LDA, "rfe_compatible": True, "lib": Library.SCIKIT}
    QDA = { "full_name": "Quadratic Discriminant Analysis", "search_params": AlgorithmGridSearchParams.QDA, "rfe_compatible": False, "lib": Library.SCIKIT}
    BGC = { "full_name": "Bagging Classifier", "search_params": AlgorithmGridSearchParams.BGC, "rfe_compatible": False, "lib": Library.SCIKIT}
    ETC = { "full_name": "Extra Trees Classifier", "search_params": AlgorithmGridSearchParams.ETC, "rfe_compatible": True, "lib": Library.SCIKIT}
    ABC = { "full_name": "Ada Boost Classifier", "search_params": AlgorithmGridSearchParams.ABC, "rfe_compatible": True, "lib": Library.SCIKIT}
    GBC = { "full_name": "Gradient Boosting Classifier", "search_params": AlgorithmGridSearchParams.GBC, "rfe_compatible": True, "lib": Library.SCIKIT}
    HIST = { "full_name": "Histogram-based Gradient B. Classifier", "search_params": AlgorithmGridSearchParams.HIST, "rfe_compatible": False, "lib": Library.SCIKIT}
    MLPC = { "full_name": "Multi Layered Peceptron", "search_params": AlgorithmGridSearchParams.MLPC, "rfe_compatible": False, "lib": Library.SCIKIT}
    GPC = { "full_name": "Gaussian Process Classifier", "search_params": AlgorithmGridSearchParams.GPC, "rfe_compatible": False, "lib": Library.SCIKIT}
    FRFD = { "full_name": "Filter + RandomForestDetector", "detector": Detector.RFD, "search_params": AlgorithmGridSearchParams.FRFD, "rfe_compatible": False, "lib": Library.CLEAN}
    #FPCD = { "full_name": "Filter + PartitioningDetector", "detector": Detector.PDEC, "search_params": AlgorithmGridSearchParams.FPCD, "rfe_compatible": False, "lib": Library.CLEAN}
    #FFKD = { "full_name": "Filter + ForestKDN", "detector": Detector.FKDN, "search_params": AlgorithmGridSearchParams.FFKD, "rfe_compatible": False, "lib": Library.CLEAN}
    #FINH = { "full_name": "Filter + InstanceHardness", "detector": Detector.INH, "search_params": AlgorithmGridSearchParams.FINH, "rfe_compatible": False, "lib": Library.CLEAN}
    CSTK = { "full_name": "Costing + KDN", "detector": Detector.KDN, "search_params": AlgorithmGridSearchParams.CSTK, "rfe_compatible": False, "lib": Library.CLEAN}
    #CSTM = { "full_name": "Costing + MCS", "detector": Detector.MCS, "search_params": AlgorithmGridSearchParams.CSTM, "rfe_compatible": False, "lib": Library.CLEAN}
    #CRFD = { "full_name": "Costing + RandomForestDetector", "detector": Detector.RFD, "search_params": AlgorithmGridSearchParams.CRFD, "rfe_compatible": False, "lib": Library.CLEAN}
    #CPCD = { "full_name": "Costing + PartitioningDetector", "detector": Detector.PDEC, "search_params": AlgorithmGridSearchParams.CPCD, "rfe_compatible": False, "lib": Library.CLEAN}
    #CFKD = { "full_name": "Costing + ForestKDN", "detector": Detector.FKDN, "search_params": AlgorithmGridSearchParams.CFKD, "rfe_compatible": False, "lib": Library.CLEAN}
    #CINH = { "full_name": "Costing + InstanceHardness", "detector": Detector.INH, "search_params": AlgorithmGridSearchParams.CINH, "rfe_compatible": False, "lib": Library.CLEAN}
    WBGK = { "full_name": "WeightedBagging + KDN", "detector": Detector.KDN, "search_params": AlgorithmGridSearchParams.WBGK, "rfe_compatible": False, "lib": Library.CLEAN}
    #WBGM = { "full_name": "WeightedBagging + MCS", "detector": Detector.MCS, "search_params": AlgorithmGridSearchParams.WBGM, "rfe_compatible": False, "lib": Library.CLEAN}   
    #WRFD = { "full_name": "WeightedBagging + RandomForestDetector", "detector": Detector.RFD, "search_params": AlgorithmGridSearchParams.WRFD, "rfe_compatible": False, "lib": Library.CLEAN}
    #WPCD = { "full_name": "WeightedBagging + PartitioningDetector", "detector": Detector.PDEC, "search_params": AlgorithmGridSearchParams.WPCD, "rfe_compatible": False, "lib": Library.CLEAN}
    #WFKD = { "full_name": "WeightedBagging + ForestKDN", "detector": Detector.FKDN, "search_params": AlgorithmGridSearchParams.WFKD, "rfe_compatible": False, "lib": Library.CLEAN}
    #WINH = { "full_name": "WeightedBagging + InstanceHardness", "detector": Detector.INH, "search_params": AlgorithmGridSearchParams.WINH, "rfe_compatible": False, "lib": Library.CLEAN}
    CLRF = { "full_name": "CLNI + RandomForestDetector", "detector": Detector.RFD, "search_params": AlgorithmGridSearchParams.CLRF, "rfe_compatible": False, "lib": Library.CLEAN}
    #CLPC = { "full_name": "CLNI + PartitioningDetector", "detector": Detector.PDEC, "search_params": AlgorithmGridSearchParams.CLPC, "rfe_compatible": False, "lib": Library.CLEAN}
    #CLFK = { "full_name": "CLNI + ForestKDN", "detector": Detector.FKDN, "search_params": AlgorithmGridSearchParams.CLFK, "rfe_compatible": False, "lib": Library.CLEAN}
    #CLIH = { "full_name": "CLNI + InstanceHardness", "detector": Detector.INH, "search_params": AlgorithmGridSearchParams.CLIH, "rfe_compatible": False, "lib": Library.CLEAN}
    VOTG = { "full_name":  "Voting Classifier", "search_params": AlgorithmGridSearchParams.VOTG, "rfe_compatible": False, "lib": Library.SCIKIT}
    TORA = { "full_name":  "PyTorch ReLu+Adam", "search_params": AlgorithmGridSearchParams.PYNN, "rfe_compatible": False, "lib": Library.TORCH}
    #TORS = { "full_name":  "PyTorch ReLu+SGD", "search_params": AlgorithmGridSearchParams.PYNN, "rfe_compatible": False, "lib": Library.TORCH}
    #TOTA = { "full_name":  "PyTorch Tanh+Adam", "search_params": AlgorithmGridSearchParams.PYNN, "rfe_compatible": False, "lib": Library.TORCH}
    #TOTS = { "full_name":  "PyTorch Tanh+SGD", "search_params": AlgorithmGridSearchParams.PYNN, "rfe_compatible": False, "lib": Library.TORCH}
    #TOSA = { "full_name":  "PyTorch Sigm+Adam", "search_params": AlgorithmGridSearchParams.PYNN, "rfe_compatible": False, "lib": Library.TORCH}
    #TOSS = { "full_name":  "PyTorch Sigm+SGD", "search_params": AlgorithmGridSearchParams.PYNN, "rfe_compatible": False, "lib": Library.TORCH}
    KERA = { "full_name": "Keras MLP Classifier", "search_params": AlgorithmGridSearchParams.KERA, "rfe_compatible": False, "lib": Library.KERAS}
    FUTV = { "full_name":  "FUT Voting Classifier", "search_params": AlgorithmGridSearchParams.FUTV, "rfe_compatible": False, "lib": Library.SCIKIT}
    FUTS = { "full_name":  "FUT Stacking Classifier", "search_params": AlgorithmGridSearchParams.FUTS, "rfe_compatible": False, "lib": Library.SCIKIT}


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
    def lib(self):
        if isinstance(self.value, dict):
            return self.value.get("lib")
        
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
        return WeightedBagging(classifier=BaggingClassifier(), detector=detector.call_detector())
    
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
        return Costing(classifier=BaggingClassifier(), detector=detector.call_detector())

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
        estimators=[('lsvc', clf1), ('rfcl', clf2), ('lrn', clf3)]
        return VotingClassifier(estimators=estimators)
    
    def do_TORA(self, max_iterations: int, size: int)-> NNClassifier3PL:     
        return self.call_PYNN(activation='relu', optimizer='adam')
    
    def do_TORS(self, max_iterations: int, size: int)-> NNClassifier3PL:     
        return self.call_PYNN(activation='relu', optimizer='sgd')
    
    def do_TOTA(self, max_iterations: int, size: int)-> NNClassifier3PL:     
        return self.call_PYNN(activation='tanh', optimizer='adam')
    
    def do_TOTS(self, max_iterations: int, size: int)-> NNClassifier3PL:     
        return self.call_PYNN(activation='tanh', optimizer='sgd')
    
    def do_TOSA(self, max_iterations: int, size: int)-> NNClassifier3PL:     
        return self.call_PYNN(activation='sigmoid', optimizer='adam')
    
    def do_TOSS(self, max_iterations: int, size: int)-> NNClassifier3PL:     
        return self.call_PYNN(activation='sigmoid', optimizer='sgd')
    
    def call_PYNN(self, activation: str, optimizer: str)-> NNClassifier3PL:     
        return NNClassifier3PL(activation=activation, optimizer=optimizer, verbose=False, train_split=True)
    
    def do_KERA(self, max_iterations: int, size: int)-> NNClassifier3PL:     
        return  MLPKerasClassifier()
    
    def do_FUTV(self, max_iterations: int, size: int) -> Filter:
        clf1 = MLPClassifier(max_iter=max_iterations)
        clf2 = RandomForestClassifier()
        estimators=[('mlpc', clf1), ('rfcl', clf2)]
        return VotingClassifier(estimators=estimators, voting = 'soft')
     
    def do_FUTS(self, max_iterations: int, size: int) -> Filter:
        clf1 = MLPClassifier(max_iter=max_iterations)
        clf2 = RandomForestClassifier()
        clf3 = AdaBoostClassifier()
        estimators=[('mlpc', clf1), ('rfcl', clf2), ('abc', clf3)]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
class Oversampling(MetaEnum):
    NOG = "No Oversampling"
    RND = "Random"
    SME = "SMOTE"
    SNC = "SMOTE-NC"
    SNN = "SMOTE-N"
    ADA = "ADASYN"
    BRD = "Borderline SMOTE"
    KMS = "Kmeans SMOTE"
    SVM = "SVM SMOTE"

    def get_full_name(self) -> str:
        return self.full_name

    def get_callable_oversampler(self) -> tuple:
        """ Gets the undersampling method that is callable 
        """
        return self.call_oversampler()

    def do_NOG(self) -> NonOversampler:
        """ While this return is superfluos, it helps with the listings of oversamplers """
        return self.NonOversampler()

    def NonOversampler(self):
        return FunctionTransformer(lambda X: X)

    def do_RND(self) -> RandomOverSampler:
        return RandomOverSampler(sampling_strategy = 'auto')
    
    def do_SME(self) -> SMOTE:
        return SMOTE(sampling_strategy = 'auto')
    
    def do_SNC(self) -> SMOTENC:
        return SMOTENC(categorical_features = 'auto', sampling_strategy = 'auto')
    
    def do_SNN(self) -> SMOTEN:
        return SMOTEN(sampling_strategy = 'auto')
    
    def do_ADA(self) -> ADASYN:
        return ADASYN(sampling_strategy = 'auto')
    
    def do_BRD(self) -> BorderlineSMOTE:
        return BorderlineSMOTE(sampling_strategy = 'auto')
    
    def do_KMS(self) -> KMeansSMOTE:
        return KMeansSMOTE(sampling_strategy = 'auto')
    
    def do_SVM(self) -> SVMSMOTE:
        return SVMSMOTE(sampling_strategy = 'auto')

    def call_oversampler(self) -> Union[Transform, None]:
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('do')
    
    @classmethod
    def defaultOversampler(cls):
        return cls.NOG
    
class Undersampling(MetaEnum):
    NUG = "No Undersampling"
    RND = "Random"
    CCS = "Cluster Centroids"
    CNN = "Condensed Nearest Neighbours"
    ENN = "Edited Nearest Neighours"
    RNN = "Repeated Edited NN"
    AKN = "All KNN"
    IHT = "Instance Hardess Threshold"
    NMS = "Near Miss"
    NCR = "Neighbourhood Cleaning Rule"
    OSS = "One Sided Selection"
    TKL = "Tomek Links"

    def get_full_name(self) -> str:
        return self.full_name

    def get_callable_undersampler(self) -> tuple:
        """ Gets the undersampling method that is callable 
        """
        return self.call_undersampler()

    def do_NUG(self) -> NonUndersampler:
        """ While this return is superfluos, it helps with the listings of oversamplers """
        return self.NonUndersampler()

    def NonUndersampler(self):
        return FunctionTransformer(lambda X: X)

    def do_RND(self) -> RandomUnderSampler:
        return RandomUnderSampler(sampling_strategy = 'auto')
    
    def do_CCS(self) -> ClusterCentroids:
        return ClusterCentroids(sampling_strategy = 'auto')
    
    def do_CNN(self) -> CondensedNearestNeighbour:
        return CondensedNearestNeighbour(sampling_strategy = 'auto')
    
    def do_ENN(self) -> EditedNearestNeighbours:
        return EditedNearestNeighbours(sampling_strategy = 'auto')
    
    def do_RNN(self) -> RepeatedEditedNearestNeighbours:
        return RepeatedEditedNearestNeighbours(sampling_strategy = 'auto')
    
    def do_AKN(self) -> AllKNN:
        return AllKNN(sampling_strategy = 'auto')
    
    def do_IHT(self) -> InstanceHardnessThreshold:
        return InstanceHardnessThreshold(sampling_strategy = 'auto')
    
    def do_NMS(self) -> NearMiss:
        return NearMiss(sampling_strategy = 'auto')
    
    def do_NCR(self) -> NeighbourhoodCleaningRule:
        return NeighbourhoodCleaningRule(sampling_strategy = 'auto')
    
    def do_OSS(self) -> OneSidedSelection:
        return OneSidedSelection(sampling_strategy = 'auto')
    
    def do_TKL(self) -> TomekLinks:
        return TomekLinks(sampling_strategy = 'auto')

    def call_undersampler(self) -> Union[Transform, None]:
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('do')
    
    @classmethod
    def defaultUndersampler(cls):
        return cls.NUG

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
    
    # TODO: For now, use temporary fixed argument for estimator. Can be changed later!
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
            components = None
            #components = PCA_VARIANCE_EXPLAINED
        
        return PCA(n_components=components)
    
    def do_PCA(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):

        tf = self.get_PCA(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def get_NYS(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = max(LOWER_LIMIT_REDUCTION, min(num_samples,num_features))
        return Nystroem(n_components=components)
    
    def do_NYS(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_NYS(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def get_TSVD(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = max(LOWER_LIMIT_REDUCTION, min(num_samples,num_features))

        return TruncatedSVD(n_components=components)
    
    def do_TSVD(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_TSVD(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def get_FICA(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = max(LOWER_LIMIT_REDUCTION, min(num_samples,num_features))
        return FastICA(n_components=components)
    
    def do_FICA(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_FICA(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)
        
    def get_NMF(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = max(LOWER_LIMIT_REDUCTION, min(num_samples,num_features))
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
            components = NON_LINEAR_REDUCTION_COMPONENTS
        return Isomap(n_components=components)

    def do_ISO(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_ISO(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def get_LLE(self, num_samples: int, num_features: int, num_selected_features: int = None):
        if num_selected_features is not None:
            components = num_selected_features
        else:
            components = NON_LINEAR_REDUCTION_COMPONENTS
        return LocallyLinearEmbedding(n_components=components)
    
    def do_LLE(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        tf = self.get_LLE(*X.shape, num_selected_features)
        return self._do_transformation(logger=logger, X=X, transformation=tf, components=tf.n_components_)

    def call_reduction(self, num_samples: int, num_features: int, num_selected_features: int = None) -> Reduction:
        """ Wrapper to general function for DRY, but name/signature kept for ease. """
        return self.call_function('get', num_samples=num_samples, num_features=num_features, \
            num_selected_features=num_selected_features)

class ScoreMetric(MetaEnum):
    auc = {"full_name": "Area under ROC curve", "callable": 'roc_auc_ovo', "kwargs": None}
    accuracy = {"full_name": "Accuracy", "callable": accuracy_score, "kwargs": None}
    balanced_accuracy = {"full_name": "Balanced Accuracy", "callable": balanced_accuracy_score, "kwargs": {"adjusted": False}}
    balanced_accuracy_adjusted = {"full_name": "Balanced Accuracy (Adjusted)", "callable": balanced_accuracy_score, "kwargs": {"adjusted": True}}
    f1_micro = {"full_name": "Balanced F1 Micro", "callable": f1_score, "kwargs": {"average": 'micro', "pos_label": None}}
    f1_macro = {"full_name": "Balanced F1 Macro", "callable": f1_score, "kwargs": {"average": 'macro', "pos_label": None}}
    f1_weighted = {"full_name": "Balanced F1 Weighted", "callable": f1_score, "kwargs": {"average": 'weighted', "pos_label": None}}
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
        
        if isinstance(self.callable, str):
            return self.callable
        
        else:

            if self.kwargs:
                return make_scorer(self.callable, **(self.kwargs))
            else:
                return make_scorer(self.callable)
        
    @classmethod
    def defaultScoreMetric(cls):
        return cls.accuracy

class AlgorithmTuple(MetaTuple):

    _type = Algorithm

    def list_callable_algorithms(self, size: int, max_iterations: int) -> list[tuple]:
        """ Gets a list of algorithms that are callable
            in the form (algorithm, called function)
        """
        return self.list_callables("do", max_iterations=max_iterations, size=size)
    
    @classmethod
    def defaultAlgorithmTuple(cls):
        return cls([Algorithm.LDA])

class PreprocessTuple(MetaTuple):

    _type = Preprocess

    def list_callable_preprocessors(self) -> list[tuple]:
        """ Gets a list of preprocessors that are callable (including NOS -> None)
            in the form (preprocessor, called function)
        """
        return self.list_callables("do")

    @classmethod
    def defaultPreprocessTuple(cls):
        return cls([Preprocess.NOS])
    
class ReductionTuple(MetaTuple):

    _type = Reduction

    def list_callable_reductions(self, num_samples: int, num_features: int, \
        num_selected_features: int = None) -> list[tuple]:
        """ Gets a list of preprocessors that are callable (including NOR -> None)
            in the form (preprocessor, called function)
        """
        return self.list_callables(
            "get",
            num_samples=num_samples,
            num_features=num_features,
            num_selected_features=num_selected_features
        )
    
    @classmethod
    def defaultReductionTuple(cls):
        return cls([Reduction.NOR])