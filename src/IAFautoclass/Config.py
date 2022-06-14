
from dataclasses import dataclass
import enum
import getopt
import importlib
import os
import sys

from sympy import O

"""
TODO: Once I know where that driver function fits
elif sql.IAFSqlHelper.drivers().find(odbc_driver) == -1:
    raise ValueError("Specified ODBC driver cannot be found!")
"""


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


class Preprocess(enum.Enum):
    ALL = "All"
    NON = "None"
    STA = "Standard Scaler"
    MIX = "Min-Max Scaler"
    MMX = "Max-Absolute Scaler"
    NRM = "Normalizer"
    BIN = "Binarizer"


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


class Scoretype(enum.Enum):
    accuracy = "Accuracy"
    balanced_accuracy = "Balanced Accuracy"
    f1_macro = "Balanced F1 Micro"
    f1_weighted = "Balanced F1 Weighted"
    recall_micro = "Recall Micro"
    recall_macro = "Recall Macro"
    recall_weighted = "Recall Weighted"
    precision_micro = "Precision Micro"
    precision_macro = "Precision Macro"
    precision_weighted = "Precision Weighted"
    mcc = "Matthews Corr. Coefficient"


@dataclass
class Config:
    name: str = "iris"
    odbc_driver: str = "ODBC Driver 17 for SQL Server"
    host: str = ""
    trusted_connection: bool = True
    class_catalog: str = ""
    class_table: str = ""
    class_table_script: str = "/sql/autoClassCreateTable.sql.txt"
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
    verbose: bool = True
    redirect_output: bool = False
    model_path: str = "./model/"
    model_name: str = "iris"
    debug_on: bool = True
    num_rows: int = None

    def __post_init__(self) -> None:
        """List of params that need type checked"""
        string_params = [
            "name",
            "category_text_columns"
        ]

        bool_params = [
            "trusted_connection",
            "use_stop_words",
            "hex_encode",
            "verbose",
            "redirect_output",
            "smote",
            "undersample"
        ]

        positive_int_params = [
            "num_selected_features",
            "max_iterations",
            "num_rows"
        ]

        for item in string_params:
            if not isinstance(getattr(self, item), str):
                raise TypeError(f"Argument {item} must be a string")

        for item in bool_params:
            if not isinstance(getattr(self, item), bool):
                raise TypeError(f"Argument {item} must be True or False")

        for item in positive_int_params:
            value = getattr(self, item)
            if value is not None:
                if not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"Argument {item} must be a positive integer")

        # database connection information
        database_connection_information = [
            isinstance(self.host, str),
            isinstance(self.class_catalog, str),
            isinstance(self.class_table, str),
            isinstance(self.class_table_script, str),
            isinstance(self.class_username, str),
            isinstance(self.class_password, str),
            isinstance(self.data_catalog, str),
            isinstance(self.data_table, str)
        ]

        if not all(database_connection_information):
            raise TypeError(
                "Specified database connection information is invalid!")

        # Login credentials
        login_credentials = [
            isinstance(self.data_username, str),
            isinstance(self.data_password, str)
        ]

        if not all(login_credentials):
            raise TypeError("Specified login credentials are invalid!")

        # Training
        # 1. Type checking + at least one is True
        mode_types = [
            isinstance(self.train, bool),
            isinstance(self.predict, bool),
            isinstance(self.mispredicted, bool),
            (self.train or self.predict or self.mispredicted)
        ]
        if not all(mode_types):
            raise ValueError(
                "Class must be set for either training, predictions or mispredictions!")

        # 2. Data columns valid for training/mispredicted
        if self.train or self.mispredicted:
            training_columns = [
                isinstance(self.data_text_columns, str),
                isinstance(self.data_numerical_columns, str),
                isinstance(self.id_column, str),
            ]

            if not all(training_columns):
                raise TypeError(
                    "Specified data columns are invalid for training!")

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

            if not (isinstance(self.algorithm, Algorithm)):
                raise TypeError("Argument algorithm is invalid")

            if not (isinstance(self.preprocessor, Preprocess)):
                raise TypeError("Argument preprocessor is invalid")

            if not (isinstance(self.feature_selection, Reduction)):
                raise TypeError("Argument feature_selection is invalid")


# In case the user has specified some input arguments to command line call
def check_input_arguments(argv):

    command_line_instructions = \
        f"Usage: {argv[0] } [-h/--help] [-f/--file <configfilename>]"

    try:
        short_options = "hf:"
        long_options = ["help", "file"]
        opts, args = getopt.getopt(argv[1:], short_options, long_options)
    except getopt.GetoptError:
        print(command_line_instructions)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h' or opt == '--help':
            print(command_line_instructions)
            sys.exit()
        elif opt == '-f' or opt == '--file':
            if arg.count("..") > 0:
                print(
                    "Configuration file must be in a subfolder to {0}".format(argv[0]))
                sys.exit()
            print("Importing specified configuration file:", arg)
            if not arg[0] == '.':
                arg = os.path.relpath(arg)

            file = arg.split('\\')[-1]
            filename = file.split('.')[0]
            filepath = '\\'.join(arg.split('\\')[:-1])
            paths = arg.split('\\')[:-1]
            try:
                paths.pop(paths.index('.'))
            except Exception as e:
                print("Filepath {0} does not seem to be relative (even after conversion)".format(
                    filepath))
                sys.exit()
            pack = '.'.join(paths)
            sys.path.insert(0, filepath)
            try:
                module = importlib.import_module(pack+"."+filename)
                return module
            except Exception as e:
                print("Filename {0} and pack {1} could not be imported dynamically".format(
                    filename, pack))
                sys.exit(str(e))
        else:
            print("Illegal argument to " + argv[0] + "!")
            print(command_line_instructions)
            sys.exit()


def load_config_from_module(argv) -> Config:
    print(argv)
    module = check_input_arguments(sys.argv)
    config = Config(
        name=module.project["name"],
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
        data_password=module.sql["data_password"],
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
        num_selected_features=module.mode["num_selected_features"],
        scoring=Scoretype[module.mode["scoring"]],
        max_iterations=int(module.mode["max_iterations"]),
        verbose=module.io["verbose"],
        redirect_output=False,
        model_path=module.io["model_path"],
        model_name=module.io["model_name"],
        debug_on=module.debug["debug_on"],
        num_rows=int(module.debug["num_rows"])
    )

    return config


def main():
    if len(sys.argv) > 1:
        config = load_config_from_module(sys.argv)
    else:
        config = Config()

    # print(Algorithm.ALL.value)
    print(config)


if __name__ == "__main__":
    main()
