
import copy
import enum
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Type, TypeVar, Union

import pandas
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectFromModel
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, SGDClassifier)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.naive_bayes import (BernoulliNB, ComplementNB, GaussianNB,
                                 MultinomialNB)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (Binarizer, MaxAbsScaler, MinMaxScaler,
                                   Normalizer, StandardScaler)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from IAFExceptions import ConfigException
import Helpers


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


class Algorithm(enum.Enum):
    ALL = "All"
    LRN = "Logistic Regression"
    KNN = "K-Neighbors Classifier"
    CART = "Decision Tree Classifier"
    GNB = "Gaussian Naive Bayes"
    MNB = "Multinomial Naive Bayes"
    BNB = "Bernoulli Naive Bayes"
    CNB = "Complement Naive Bayes"
    REC = "Ridge Classifier"
    PCN = "Perceptron"
    PAC = "Passive Aggressive Classifier"
    RFC1 = "Random Forest Classifier 1"
    RFC2 = "Random Forest Classifier 2"
    LIN1 = "Linear Support Vector L1"
    LIN2 = "Linear Support Vector L2"
    LINP = "Linear SV L1+L2"
    SGD = "Stochastic Gradient Descent"
    SGD1 = "Stochastic GD L1"
    SGD2 = "Stochastic GD L2"
    SGDE = "Stochastic GD Elast."
    NCT = "Nearest Centroid"
    SVC = "Support Vector Classification"
    LDA = "Linear Discriminant Analysis"
    QDA = "Quadratic Discriminant Analysis"
    BDT = "Bagging CLassifier"
    ETC = "Extra Trees Classifier"
    ABC = "Ada Boost Classifier"
    GBC = "Gradient Boosting Classifier"
    MLPR = "ML Neural Network Relu"
    MLPL = "ML Neural Network Sigm"

    @classmethod
    def list_callable_algorithms(cls, size: int, max_iterations: int) -> list[tuple]:
        """ Gets a list of algorithms that are callable
            in the form (algorithm, called function)
        """
        return [(algo, algo.call_algorithm(max_iterations=max_iterations, size=size)) for algo in cls if algo.has_algorithm_function()]

    def do_LRN(self, max_iterations: int, size: int)-> LogisticRegression:
        return LogisticRegression(solver='liblinear', multi_class='ovr')


    def do_KNN(self, max_iterations: int, size: int)-> KNeighborsClassifier:
        return KNeighborsClassifier()


    def do_CART(self, max_iterations: int, size: int)-> DecisionTreeClassifier:
        return DecisionTreeClassifier()


    def do_GNB(self, max_iterations: int, size: int)-> GaussianNB:
        return GaussianNB()


    def do_MNB(self, max_iterations: int, size: int)-> MultinomialNB:
        return MultinomialNB(alpha=.01)


    def do_BNB(self, max_iterations: int, size: int)-> BernoulliNB:
        return BernoulliNB(alpha=.01)


    def do_CNB(self, max_iterations: int, size: int)-> ComplementNB:
        return ComplementNB(alpha=.01)


    def do_REC(self, max_iterations: int, size: int)-> RidgeClassifier:
        return RidgeClassifier(tol=1e-2, solver="sag")


    def do_PCN(self, max_iterations: int, size: int)-> Perceptron:
        return Perceptron(max_iter=max_iterations)


    def do_PAC(self, max_iterations: int, size: int)-> PassiveAggressiveClassifier:
        return PassiveAggressiveClassifier(max_iter=max_iterations)


    def do_RFC1(self, max_iterations: int, size: int)-> RandomForestClassifier:
        return self.call_RandomForest("sqrt")


    def do_RFC2(self, max_iterations: int, size: int)-> RandomForestClassifier:
        return self.call_RandomForest("log2")

    def call_RandomForest(self, max_features: str) -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=100, max_features=max_features)


    def do_LIN1(self, max_iterations: int, size: int)-> LinearSVC:
        return self.call_LinearSVC(max_iterations, "l1", False) 


    def do_LIN2(self, max_iterations: int, size: int)-> LinearSVC:  
        return self.call_LinearSVC(max_iterations, "l2", False)


    def do_LINP(self, max_iterations: int, size: int)-> Pipeline:
        return Pipeline([
            ('feature_selection', SelectFromModel(self.call_LinearSVC(max_iterations, "l1", False))),
            ('classification', self.call_LinearSVC(max_iterations, "l2"))])
    
    def do_SVC(self, max_iterations: int, size: int):
        # TODO: Move this into the algorithms having a dict as a value with description/limit as keys and limit: None if no limit
        LIMIT_SVC = 10000
        if size < LIMIT_SVC:
            return SVC(gamma='auto', probability=True)
        
        # TODO: communicate with terminal to warn? 
        # print("\nNotice: SVC model was exchange for LinearSVC since n_samples > {0}\n".format(self.LIMIT_SVC))
        return self.call_LinearSVC(max_iterations, "l1", False)
    

    def call_LinearSVC(self, max_iterations: int, penalty: str, dual: bool = None) -> LinearSVC:
        if dual is None:
            return LinearSVC(penalty=penalty, max_iter=max_iterations)
        
        return LinearSVC(penalty=penalty, dual=dual, tol=1e-3, max_iter=max_iterations)

    def do_SGD(self, max_iterations: int, size: int)-> SGDClassifier:
        return self.call_SGD(max_iterations)    


    def do_SGD1(self, max_iterations: int, size: int)-> SGDClassifier:
        return self.call_SGD(max_iterations, "l1")  
        

    def do_SGD2(self, max_iterations: int, size: int)-> SGDClassifier:
        return self.call_SGD(max_iterations, "l2")


    def do_SGDE(self, max_iterations: int, size: int)-> SGDClassifier:
        return self.call_SGD(max_iterations, "elasticnet")

    def call_SGD(self, max_iterations: int,  penalty: str = None) -> SGDClassifier:
        if penalty is None:
           return  SGDClassifier()
        
        return SGDClassifier(alpha=.0001, max_iter=max_iterations, penalty=penalty)

    def do_NCT(self, max_iterations: int, size: int)-> NearestCentroid:     
        return NearestCentroid()


    def do_LDA(self, max_iterations: int, size: int)-> LinearDiscriminantAnalysis:     
        return LinearDiscriminantAnalysis()


    def do_QDA(self, max_iterations: int, size: int)-> QuadraticDiscriminantAnalysis:     
        return QuadraticDiscriminantAnalysis()


    def do_BDT(self, max_iterations: int, size: int)-> BaggingClassifier:     
        return BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators = 100, random_state = 7)


    def do_ETC(self, max_iterations: int, size: int)-> ExtraTreesClassifier:     
        return ExtraTreesClassifier(n_estimators = 100)


    def do_ABC(self, max_iterations: int, size: int)-> AdaBoostClassifier:     
        return AdaBoostClassifier(n_estimators = 30, random_state = 7)


    def do_GBC(self, max_iterations: int, size: int)-> GradientBoostingClassifier:     
        return GradientBoostingClassifier(n_estimators = 100, random_state = 7)

    def do_MLPR(self, max_iterations: int, size: int)-> MLPClassifier:
        return self.call_MLP(max_iterations, 'relu')  

    def do_MLPL(self, max_iterations: int, size: int)-> MLPClassifier:
        return self.call_MLP(max_iterations, 'logistic')

    def call_MLP(self, max_iterations: int, activation: str) -> MLPClassifier:
        return MLPClassifier(activation = activation, solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), max_iter=max_iterations, random_state=1)


    def get_function_name(self) -> str:
        return f"do_{self.name}"
    
    def call_algorithm(self, max_iterations: int, size: int):
        do = self.get_function_name()
        if hasattr(self, do) and callable(func := getattr(self, do)):
            return func(max_iterations, size)
        
        return None

    def has_algorithm_function(self) -> bool:
        do = self.get_function_name()
        return hasattr(self, do) and callable(getattr(self, do))
    

class Preprocess(enum.Enum):
    ALL = "All"
    NON = "None"
    STA = "Standard Scaler"
    MIX = "Min-Max Scaler"
    MMX = "Max-Absolute Scaler"
    NRM = "Normalizer"
    BIN = "Binarizer"

    @classmethod
    def list_callable_preprocessors(cls, is_text_data: bool) -> list[tuple]:
        """ Gets a list of preprocessors that are callable (including NON -> None)
            in the form (preprocessor, called function)
        """
        return [(pp, pp.call_preprocess()) for pp in cls if pp.has_preprocess_function() and (pp.name != "BIN" or is_text_data)]

    def do_NON(self) -> None:
        """ While this return is superfluos, it helps with the listings of preprocessors """
        return None

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

    def get_function_name(self) -> str:
        return f"do_{self.name}"
    
    def call_preprocess(self):
        do = self.get_function_name()
        if hasattr(self, do) and callable(func := getattr(self, do)):
            return func()
        
        return None
        

    def has_preprocess_function(self) -> bool:
        do = self.get_function_name()
        return hasattr(self, do) and callable(getattr(self, do))


class Reduction(enum.Enum):
    NON = "None"
    RFE = "Recursive Feature Elimination"
    PCA = "Principal Component Analysis"
    NYS = "Nystroem Method"
    TSVD = "Truncated SVD"
    FICA = "Fast Indep. Component Analysis"
    GRP = "Gaussion Random Projection"
    ISO = "Isometric Mapping"
    LLE = "Locally Linearized Embedding"

    def call_transformation(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        do = self.get_function_name()
        if hasattr(self, do) and callable(func := getattr(self, do)):
            return func(logger, X, num_selected_features)
        
        return None

    def get_function_name(self) -> str:
        return f"do_{self.name}"

    def has_transformation_function(self) -> bool:
        do = self.get_function_name()
        return hasattr(self, do) and callable(getattr(self, do))

    def _do_transformation(self, logger: Logger, X: pandas.DataFrame, transformation, components):
        try:
            feature_selection_transform = transformation
            feature_selection_transform.fit(X)
            X = feature_selection_transform.transform(X)
        except Exception as e:
            logger.print_components(self.name, components, str(e))
            feature_selection_transform = None
        else:
            # Else in a try-except clause means that there were no exceptions
            logger.print_info(f"...new shape of data matrix is: ({X.shape[0]},{X.shape[1]})")

        return X, feature_selection_transform

    def do_PCA(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        components = None
        components_options = []
        if num_selected_features != None and num_selected_features > 0:
            components_options.append(num_selected_features)
            components = num_selected_features
        if X.shape[0] >= X.shape[1]:
            components_options.append('mle')
            components = 'mle'
        components_options.append(Config.PCA_VARIANCE_EXPLAINED)
        components_options.append(min(X.shape[0], X.shape[1]) - 1)
        X_original = X
        # Make transformation
        for components in components_options:
            logger.print_components("PCA", components)
            transformation = PCA(n_components=components)
            X, feature_selection_transform = self._do_transformation(logger=logger, X=X_original, transformation=transformation, components=components)
            
            if feature_selection_transform is not None:
                break

        return X, feature_selection_transform

    def do_NYS(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        if num_selected_features != None and num_selected_features > 0:
            components = num_selected_features
        else:
            components = max(Config.LOWER_LIMIT_REDUCTION, min(X.shape))
            logger.print_components("Nystroem", components)
        # Make transformation
        transformation = Nystroem(n_components=components)
        
        return self._do_transformation(logger=logger, X=X, transformation=transformation, components=components)

    def do_TSVD(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        if num_selected_features != None and num_selected_features > 0:
            components = num_selected_features
        else:
            components = max(Config.LOWER_LIMIT_REDUCTION, min(X.shape))
            logger.print_components("Truncated SVD", components)
        # Make transformation
        transformation = TruncatedSVD(n_components=components)
        
        return self._do_transformation(logger=logger, X=X, transformation=transformation, components=components)

    def do_FICA(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        if num_selected_features != None and num_selected_features > 0:
            components = num_selected_features
        else:
            components = max(Config.LOWER_LIMIT_REDUCTION, min(X.shape))
            logger.print_components("Fast ICA", components)
            
        # Make transformation
        transformation = FastICA(n_components=components)
        
        return self._do_transformation(logger=logger, X=X, transformation=transformation, components=components)


    def do_GRP(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        if num_selected_features != None and num_selected_features > 0:
            components = num_selected_features
        else:
            components = 'auto'
            logger.print_components("GRP", components)
        # Make transformation
        transformation = GaussianRandomProjection(n_components=components)
        
        return self._do_transformation(logger=logger, X=X, transformation=transformation, components=components)

    def do_ISO(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        if num_selected_features != None and num_selected_features > 0:
            components = num_selected_features
        else:
            components = Config.NON_LINEAR_REDUCTION_COMPONENTS
            logger.print_components("ISO", components)
        # Make transformation
        transformation = Isomap(n_components=components)
        
        return self._do_transformation(logger=logger, X=X, transformation=transformation, components=components)

    def do_LLE(self, logger: Logger, X: pandas.DataFrame, num_selected_features: int = None):
        if num_selected_features != None and num_selected_features > 0:
            components = num_selected_features
        else:
            components = Config.NON_LINEAR_REDUCTION_COMPONENTS
            logger.print_components("LLE", components)
        # Make transformation
        transformation = LocallyLinearEmbedding(n_components=components)
        
        return self._do_transformation(logger=logger, X=X, transformation=transformation, components=components)


class Scoretype(enum.Enum):
    accuracy = "Accuracy"
    balanced_accuracy = "Balanced Accuracy"
    f1_micro = "Balanced F1 Micro"
    f1_weighted = "Balanced F1 Weighted"
    recall_micro = "Recall Micro"
    recall_macro = "Recall Macro"
    recall_weighted = "Recall Weighted"
    precision_micro = "Precision Micro"
    precision_macro = "Precision Macro"
    precision_weighted = "Precision Weighted"
    mcc = "Matthews Corr. Coefficient"

    def get_mechanism(self) -> Union[str, Callable]:
        """ Returns the scoring mechanism based on the Scoretype"""
        if self == Scoretype.mcc:
            return make_scorer(matthews_corrcoef)

        return self.name

T = TypeVar('T', bound='Config')

@dataclass
class Config:
    
    MAX_ITERATIONS = 20000
    CONFIG_FILENAME_START = "autoclassconfig_"
    CONFIG_SAMPLE_FILE = CONFIG_FILENAME_START + "template.py"
    PCA_VARIANCE_EXPLAINED = 0.999
    LOWER_LIMIT_REDUCTION = 100
    NON_LINEAR_REDUCTION_COMPONENTS = 2

    DEFAULT_MODELS_PATH =  ".\\src\\IAFautoclass\\model\\"
    DEFAULT_MODEL_EXTENSION = ".sav"

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
        "debug.num_rows": "<num_rows>"
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
        data_text_columns: str = ""
        data_numerical_columns: str = ""
        id_column: str = "id"
        data_username: str = ""
        data_password: str = ""

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
                f" * Text Data columns (CSV):                 {self.data_text_columns}",
                f" * Numerical Data columns (CSV):            {self.data_numerical_columns}",
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
        use_stop_words: bool = True
        specific_stop_words_threshold: float = 1.0
        hex_encode: bool = True
        use_categorization: bool = True
        category_text_columns: str = ""
        test_size: float = 0.2
        smote: bool = False
        undersample: bool = False
        algorithm: Algorithm = Algorithm.ALL
        preprocessor: Preprocess = Preprocess.NON
        feature_selection: Reduction = Reduction.NON
        num_selected_features: int = None
        scoring: Scoretype = Scoretype.accuracy
        max_iterations: int = None

        def __str__(self) -> str:
            str_list = [
                " 2. Classification mode settings ",
                f" * Train new model:                         {self.train}",
                f" * Make predictions with model:             {self.predict}",
                f" * Display mispredicted training data:      {self.mispredicted}",
                f" * Use stop words:                          {self.use_stop_words}",
                f" * Material specific stop words threshold:  {self.specific_stop_words_threshold}",
                f" * Hex encode text data:                    {self.hex_encode}",
                f" * Categorize text data where applicable:   {self.use_categorization}",
                f" * Force categorization to these columns:   {self.category_text_columns}",
                f" * Test size for trainings:                 {self.test_size}",
                f" * Use SMOTE:                               {self.smote}",
                f" * Use undersampling of majority class:     {self.undersample}",
                f" * Algorithm of choice:                     {self.algorithm.value}",
                f" * Preprocessing method of choice:          {self.preprocessor.value}",
                f" * Scoring method:                          {self.scoring.value}",
                f" * Feature selection:                       {self.feature_selection.value}",
                f" * Number of selected features:             {self.num_selected_features}",
                f" * Maximum iterations (where applicable):   {self.max_iterations}"
            ]

            return "\n".join(str_list)

    @dataclass
    class Debug:
        on: bool = True
        num_rows: int = None

        def __str__(self) -> str:
            str_list = [
                " 4. Debug settings  ",
                f" * Debugging on:                            {self.on}",
                f" * How many data rows to consider:          {self.num_rows}"
            ]

            return "\n".join(str_list)   
    
    connection: Connection = field(default_factory=Connection)
    mode: Mode  = field(default_factory=Mode)
    io: IO = field(default_factory=IO)
    debug: Debug = field(default_factory=Debug)
    name: str = "iris"
    config_path: str = None
    filename: str = None
    save: bool = False
    

    def __post_init__(self) -> None:
        if self.config_path is None:
            pwd = os.path.dirname(os.path.realpath(__file__))
            self.config_path = Path(pwd) / "./config/"

        if self.filename is None:
            self.filename = f"{self.CONFIG_FILENAME_START}{self.name}_{self.connection.data_username}.py"
        
        
        """Post init is called after init, which is the best place to check the types & values"""

        # 1: Top config params
        if not isinstance(self.name, str):
            raise TypeError(f"Argument name must be a string")

        # 2: Connection params
        # 2.1: database connection information
        database_connection_information = [
            isinstance(self.connection.host, str),
            isinstance(self.connection.class_catalog, str),
            isinstance(self.connection.class_table, str),
            isinstance(self.connection.class_table_script, str),
            isinstance(self.connection.class_username, str),
            isinstance(self.connection.class_password, str),
            isinstance(self.connection.data_catalog, str),
            isinstance(self.connection.data_table, str),
            isinstance(self.connection.class_column, str),
            isinstance(self.connection.data_text_columns, str),
            isinstance(self.connection.data_numerical_columns, str),
            isinstance(self.connection.id_column, str),
            isinstance(self.connection.trusted_connection, bool)
        ]
        
        if not all(database_connection_information):
            raise TypeError(
                "Specified database connection information is invalid")

        
        # 2.2: Login credentials
        login_credentials = [
            isinstance(self.connection.data_username, str),
            isinstance(self.connection.data_password, str)
        ]

        if not all(login_credentials):
            raise TypeError("Specified login credentials are invalid!")
        
        # 3: Mode/training
        if not isinstance(self.mode.category_text_columns, str):
            raise TypeError(f"Argument category_text_columns must be a string")
        
        if not positive_int_or_none(self.mode.num_selected_features):
            raise ValueError(
                "Argument num_selected_features must be a positive integer")

        if self.mode.max_iterations is None:
            self.mode.max_iterations = self.MAX_ITERATIONS
        elif not positive_int_or_none(self.mode.max_iterations):
            raise ValueError(
                "Argument max_iterations must be a positive integer")

        # Type checking + at least one is True
        mode_types = [
            isinstance(self.mode.train, bool),
            isinstance(self.mode.predict, bool),
            isinstance(self.mode.mispredicted, bool),
            (self.mode.train or self.mode.predict)
        ]
        
        if not all(mode_types):
            raise ValueError(
                "Class must be set for either training, predictions and/or mispredictions!")

        if self.mode.mispredicted and not self.mode.train:
            raise ValueError(
                "Class must be set for training if it is set for misprediction")

        # Stop words threshold and test size
        if isinstance(self.mode.specific_stop_words_threshold, float):
            if self.mode.specific_stop_words_threshold > 1.0 or self.mode.specific_stop_words_threshold < 0.0:
                 raise ValueError(
                    "Argument specific_stop_words_threshold must be between 0 and 1!")
        else:
            raise TypeError(
                "Argument specific_stop_words_threshold must be a float between 0 and 1!")

        
        if isinstance(self.mode.test_size, float):
            if self.mode.test_size > 1.0 or self.mode.test_size < 0.0:
                raise ValueError(
                    "Argument test_size must be between 0 and 1!")
        else:
            raise TypeError(
                "Argument test_size must be a float between 0 and 1!")

        if not (isinstance(self.mode.algorithm, Algorithm)):
            raise TypeError("Argument algorithm is invalid")

        if not (isinstance(self.mode.preprocessor, Preprocess)):
            raise TypeError("Argument preprocessor is invalid")

        if not (isinstance(self.mode.feature_selection, Reduction)):
            raise TypeError("Argument feature_selection is invalid")

        for item in [
            "use_stop_words",
            "hex_encode",
            "smote",
            "undersample"
        ]:
            if not isinstance(getattr(self.mode, item), bool):
                raise TypeError(f"Argument {item} must be True or False")
        
        
        # 4: IO
        for item in [
            "verbose",
        ]:
            if not isinstance(getattr(self.io, item), bool):
                raise TypeError(f"Argument {item} must be True or False")

        # 5: Debug
        # TODO: Set the value based on count_data_rows(), but first decide where that method should be
        if not positive_int_or_none(self.debug.num_rows):
            raise ValueError(
                "Argument num_rows must be a positive integer")

        # Overriding values
        if self.connection.trusted_connection:
            self.connection.data_password = ""
            self.connection.class_password = ""

            # TODO: Change to not use os.getlogin()
            username = os.getlogin()
            self.connection.class_username = username
            self.connection.data_username = username

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

    def get_model_filename(self, pwd: str = None) -> str:
        """ Set the name and path of the model file
            The second parameter allows for injecting the path for reliable testing
        """
        if pwd is None:
            pwd = os.path.dirname(os.path.realpath(__file__))
        
        model_path = Path(pwd) / self.io.model_path
        
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
        configuration.debug.num_rows = 0
        configuration.save = False
        
        return configuration

    # Saves config to be read from the command line
    def save_to_file(self, filename: str = None, username: str = None) -> None:
        template_path = self.config_path / self.CONFIG_SAMPLE_FILE
        
        with open(template_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
        
        for tag in self.TEMPLATE_TAGS:
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


            for i in range(len(lines)):
                lines[i] = lines[i].replace(template, str(replace))
       
        if filename is None:
            filename = self.config_path / self.filename
        with open(filename, "w", encoding="utf-8") as fout:
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
            saved_config.connection.data_catalog = config.connection.data_catalog
            saved_config.connection.data_table = config.connection.data_table
            saved_config.io.model_name = config.io.model_name
            saved_config.debug.num_rows = config.debug.num_rows
        
        return saved_config


    @classmethod
    def load_config_from_module(cls: Type[T], argv) -> T:
        module = Helpers.check_input_arguments(argv)
        
        version = '1.0'
        if (hasattr(module, 'version')):
            version = module.version

        if (version == '1.0'):
            # Will not write test case for this, since it's deprecated
            return Config.load_config_1(module)
        
        if (version == '2.0'):
            return Config.load_config_2(module)

    @classmethod
    def load_config_2(cls: Type[T], module) -> T:
        num_rows = set_none_or_int(module.debug["num_rows"])
        num_selected_features = set_none_or_int(module.mode["num_selected_features"])
        max_iterations = set_none_or_int(module.mode["max_iterations"])

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
                data_text_columns=module.connection["data_text_columns"],
                data_numerical_columns=module.connection["data_numerical_columns"],
                id_column=module.connection["id_column"],
                data_username=module.connection["data_username"],
                data_password=module.connection["data_password"]
            ),
            Config.Mode(
                train=module.mode["train"],
                predict=module.mode["predict"],
                mispredicted=module.mode["mispredicted"],
                use_stop_words=module.mode["use_stop_words"],
                specific_stop_words_threshold=float(
                    module.mode["specific_stop_words_threshold"]),
                hex_encode=module.mode["hex_encode"],
                use_categorization=module.mode["use_categorization"],
                category_text_columns=module.mode["category_text_columns"],
                test_size=float(module.mode["test_size"]),
                smote=module.mode["smote"],
                undersample=module.mode["undersample"],
                algorithm=Algorithm[module.mode["algorithm"]],
                preprocessor=Preprocess[module.mode["preprocessor"]],
                feature_selection=Reduction[module.mode["feature_selection"]],
                num_selected_features=num_selected_features,
                scoring=Scoretype[module.mode["scoring"]],
                max_iterations=max_iterations
            ),
            Config.IO(
                verbose=module.io["verbose"],
                model_path=module.io["model_path"],
                model_name=module.io["model_name"]
            ),
            Config.Debug(
                on=module.debug["on"],
                num_rows=num_rows
            ),
            name=module.name
        )

        return config
        
    @classmethod
    def load_config_1(cls: Type[T], module) -> T:
        num_rows = set_none_or_int(module.debug["num_rows"])
        num_selected_features = set_none_or_int(module.mode["num_selected_features"])
        max_iterations = set_none_or_int(module.mode["max_iterations"])

        config = cls(
            Config.Connection(
                odbc_driver=module.sql["odbc_driver"],
                host=module.sql["host"],
                trusted_connection=module.sql["trusted_connection"],
                class_catalog=module.sql["class_catalog"],
                class_table=module.sql["class_table"],
                class_table_script=module.sql["class_table_script"],
                class_username=module.sql["class_username"],
                class_password=module.sql["class_password"],
                data_catalog=module.sql["data_catalog"],
                data_table=module.sql["data_table"],
                class_column=module.sql["class_column"],
                data_text_columns=module.sql["data_text_columns"],
                data_numerical_columns=module.sql["data_numerical_columns"],
                id_column=module.sql["id_column"],
                data_username=module.sql["data_username"],
                data_password=module.sql["data_password"]
            ),
            Config.Mode(
                train=module.mode["train"],
                predict=module.mode["predict"],
                mispredicted=module.mode["mispredicted"],
                use_stop_words=module.mode["use_stop_words"],
                specific_stop_words_threshold=float(
                    module.mode["specific_stop_words_threshold"]),
                hex_encode=module.mode["hex_encode"],
                use_categorization=module.mode["use_categorization"],
                category_text_columns=module.mode["category_text_columns"],
                test_size=float(module.mode["test_size"]),
                smote=module.mode["smote"],
                undersample=module.mode["undersample"],
                algorithm=Algorithm[module.mode["algorithm"]],
                preprocessor=Preprocess[module.mode["preprocessor"]],
                feature_selection=Reduction[module.mode["feature_selection"]],
                num_selected_features=num_selected_features,
                scoring=Scoretype[module.mode["scoring"]],
                max_iterations=max_iterations
            ),
            Config.IO(
                verbose=module.io["verbose"],
                model_path=module.io["model_path"],
                model_name=module.io["model_name"]
            ),
            Config.Debug(
                on=module.debug["debug_on"],
                num_rows=num_rows
            ),
            name=module.project["name"]
        )

        return config
    
    # Methods to hide implementation of Config
    # TODO: Change these three to lists and the implementation accordingly
    def is_text_data(self) -> bool:
        return self.connection.data_text_columns != ""
    
    def is_numerical_data(self) -> bool:
        return self.connection.data_numerical_columns != ""

    def force_categorization(self) -> bool:
        return  self.mode.category_text_columns != ""

    def column_is_numeric(self, column: str) -> bool:
        """ Checks if the column is numerical """
        return column in self.get_numerical_column_names()

    def column_is_text(self, column: str) -> bool:
        """ Checks if the column is text based """
        return column in self.get_text_column_names()

    def get_feature_selection(self) -> Reduction:
        """ Gets the given feature selection Reduction """
        return self.mode.feature_selection

    def get_none_or_positive_value(self, attribute: str) -> int:
        value = self.get_attribute(attribute)
        
        if value is None or value == "":
            return 0

        return value

    def get_attribute(self, attribute: str):
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
        return self.mode.feature_selection != Reduction.NON

    def get_test_size(self) -> float:
        """ Gets the test_size """
        return self.mode.test_size

    def get_test_size_percentage(self) -> int:
        """ Gets the test_size as a percentage """
        return int(self.mode.test_size * 100.0)

    def get_max_limit(self) -> int:
        """ Get the max limit. Name might change depending on GUI names"""
        return self.get_none_or_positive_value("debug.num_rows")

    def get_max_iterations(self) -> int:
        """ Get max iterations """
        return self.get_none_or_positive_value("mode.max_iterations")

    def is_verbose(self) -> bool:
        """ Returns what the io.verbose is set to"""
        return self.io.verbose

    def clean_column_names_list(self, column_names: str) -> list[str]:
        """ This takes a comma-delimeted string and returns the list with no empty values"""
        splitted = column_names.split(",")

        return [ elem for elem in splitted if elem != ""] # Removes all empty column names
        
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
        return self.clean_column_names_list(self.mode.category_text_columns)
        
        
    def get_text_column_names(self) -> list[str]:
        """ Gets the specified text columns"""
        return self.clean_column_names_list(self.connection.data_text_columns)


    def get_numerical_column_names(self) -> list[str]:
        """ Gets the specified numerical columns"""
        return self.clean_column_names_list(self.connection.data_numerical_columns)


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
        
    def get_undersampler(self) -> Union[RandomUnderSampler, None]:
        """ Gets the UnderSampler, or None if there should be none"""
        if self.mode.undersample:
            return RandomUnderSampler(sampling_strategy='auto')

        return None

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
        return self.mode.algorithm

    def get_preprocessor(self) -> Preprocess:
        """ get preprocessor from Config """
        return self.mode.preprocessor
    
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

    
   

    
# TODO: Move to helper?
def positive_int_or_none(value: int) -> bool:
    if value is None:
        return True
    
    if isinstance(value, int) and value >= 0:
        return True

    return False

# TODO: Move to algorithm?
def get_model_name(algo: Algorithm, prepros: Preprocess)->str:
    return f"{algo.name}-{prepros.name}"



# TODO: Move to Helpers?
def set_none_or_int(value) -> int:
    if value == "None":
        return None

    if int(value) < 0:
        return None

    return int(value)


def main():
    if len(sys.argv) > 1:
        config = Config.load_config_from_module(sys.argv)
    else:
       config = Config()

    #print(isinstance(config.mode.scoring, enum.Enum))
    #print(config.mode.scoring.name)
    #config.export_configuration_to_file()
    # print(Algorithm.ALL.value)
    print(config)


if __name__ == "__main__":
    main()
