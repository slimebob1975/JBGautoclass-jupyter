
import copy
from dataclasses import dataclass, field
import enum
import getopt
import importlib
import os
import sys

from pathlib import Path


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
    
    MAX_ITERATIONS = 20000
    LIMIT_IS_CATEGORICAL = 30
    CONFIG_FILENAME_PATH = "./config/"
    CONFIG_FILENAME_START = "autoclassconfig_"
    CONFIG_SAMPLE_FILE = CONFIG_FILENAME_START + "template.py"

    CONFIGURATION_TAGS = ["<name>", "<odbc_driver>", "<host>", "<trusted_connection>", "<class_catalog>", \
                          "<class_table>", "<class_table_script>", "<class_username>", \
                          "<class_password>", "<data_catalog>", "<data_table>", "<class_column>", \
                          "<data_text_columns>", "<data_numerical_columns>", "<id_column>", \
                          "<data_username>", "<data_password>", "<train>", "<predict>", "<mispredicted>", \
                          "<use_stop_words>", "<specific_stop_words_threshold>", "<hex_encode>", \
                          "<use_categorization>", "<category_text_columns>", "<test_size>", "<smote>", "<undersample>", \
                          "<algorithm>", "<preprocessor>", "<feature_selection>", "<num_selected_features>", "<scoring>", \
                          "<max_iterations>", "<verbose>", "<model_path>", "<model_name>",  "<debug_on>", "<num_rows>"]

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
        "debug.debug_on": "<debug_on>",
        "debug.num_rows": "<num_rows>"
    }

    @dataclass
    class Connection:
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
                

    @dataclass
    class IO:
        verbose: bool = True
        redirect_output: bool = False
        model_path: str = "./model/"
        model_name: str = "iris"

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

    @dataclass
    class Debug:
        debug_on: bool = True
        num_rows: int = None
    
    connection: Connection = field(default_factory=Connection)
    mode: Mode  = field(default_factory=Mode)
    io: IO = field(default_factory=IO)
    debug: Debug = field(default_factory=Debug)
    name: str = "iris"
    config_path: str = field(init=False)
    filename: str = field(init=False)

    def __post_init__(self) -> None:
        pwd = os.path.dirname(os.path.realpath(__file__))
        self.config_path = Path(pwd) / self.CONFIG_FILENAME_PATH
        
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
        # TODO: Move in the "if trusted remove password etc" here?
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
            (self.mode.train or self.mode.predict),
            (self.mode.mispredicted and self.mode.train)
        ]
        if not all(mode_types):
            raise ValueError(
                "Class must be set for either training, predictions and/or mispredictions!")


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
            "redirect_output"
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

        self.filename = f"{self.CONFIG_FILENAME_START}{self.name}_{self.connection.data_username}.py"


    # Extracts the config information to save with a model
    # TODO: rename: "get_clean_config"
    def get_configuration_to_save(self):
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
        
        return configuration

    # Saves config to be read from the command line
    # TODO: rename: "save_to_file"
    def export_configuration_to_file(self) -> None:
        with open(self.config_path / self.CONFIG_SAMPLE_FILE, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
        
        for tag in self.TEMPLATE_TAGS:
            template = self.TEMPLATE_TAGS[tag]
            location = tag.split(".")
            
            if (len(location) == 1):
                replace = getattr(self, location[0])
            else:
                head = getattr(self, location[0])
                replace = getattr(head, location[1])

            # Check if it's one of the enum variables
            if (isinstance(replace, enum.Enum)):
                replace = replace.name

            
            for i in range(len(lines)):
                lines[i] = lines[i].replace(template, str(replace))
       
        
        with open(self.config_path / self.filename, "w", encoding="utf-8") as fout:
           fout.writelines(lines)
        


    
def positive_int_or_none(value: int) -> bool:
    if value is None:
        return True
    
    if isinstance(value, int) and value >= 0:
        return True

    return False


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
    module = check_input_arguments(argv)

    version = '1.0'
    if (hasattr(module, 'version')):
        version = module.version

    if (version == '1.0'):
        return load_config_1(module)
    
    if (version == '2.0'):
        return load_config_2(module)
    
def set_none_or_int(value):
    if value == "None":
        return None

    return int(value)

def load_config_2(module) -> Config:
    num_rows = set_none_or_int(module.debug["num_rows"])
    num_selected_features = set_none_or_int(module.mode["num_selected_features"])
    max_iterations = set_none_or_int(module.mode["max_iterations"])

    config = Config(
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
            redirect_output=False,
            model_path=module.io["model_path"],
            model_name=module.io["model_name"]
        ),
        Config.Debug(
            debug_on=module.debug["debug_on"],
            num_rows=num_rows
        ),
        name=module.name
    )

    return config
    
    
def load_config_1(module) -> Config:
    config = Config(
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
            num_selected_features=module.mode["num_selected_features"],
            scoring=Scoretype[module.mode["scoring"]],
            max_iterations=int(module.mode["max_iterations"])
        ),
        Config.IO(
            verbose=module.io["verbose"],
            redirect_output=False,
            model_path=module.io["model_path"],
            model_name=module.io["model_name"]
        ),
        Config.Debug(
            debug_on=module.debug["debug_on"],
            num_rows=int(module.debug["num_rows"])
        ),
        name=module.project["name"]
    )

    return config


def main():
    if len(sys.argv) > 1:
        config = load_config_from_module(sys.argv)
    else:
       config = Config(name="foo")

    #print(isinstance(config.mode.scoring, enum.Enum))
    #print(config.mode.scoring.name)
    #config.export_configuration_to_file()
    # print(Algorithm.ALL.value)
    print(config)


if __name__ == "__main__":
    main()
