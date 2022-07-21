import base64
import time
import langdetect
from dataclasses import dataclass, field
from datetime import datetime
from lexicalrichness import LexicalRichness
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from DataLayer import DataLayer
from IAFLogger import IAFLogger
import Config
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class DatasetException(Exception):
    def __init__(self, message):
        super().__init__(f"DatasetException: {message}")

@dataclass
class DatasetHandler:
    STANDARD_LANG = "sv"
    PCA_VARIANCE_EXPLAINED = 0.999
    
    
    datalayer: DataLayer
    config: Config.Config
    logger: IAFLogger
    dataset: pandas.DataFrame = field(init=False)
    classes: list[str] = field(init=False)
    queries: dict = field(default_factory=dict)
    keys: pandas.Series = field(init=False)
    unpredicted_keys: pandas.Series = field(init=False)
    text_data: bool = field(init=False)
    numerical_data: bool = field(init=False)
    force_categorization: bool = field(init=False)
    use_feature_selection: bool = field(init=False)
    X_original: pandas.DataFrame = field(init=False)
    X: pandas.DataFrame = field(init=False)
    Y: pandas.DataFrame = field(init=False)
    X_train: pandas.DataFrame = field(init=False)
    X_validation: pandas.DataFrame = field(init=False)
    Y_train: pandas.DataFrame = field(init=False)
    Y_validation: pandas.DataFrame = field(init=False)
    Y_unknown: pandas.DataFrame = field(init=False)
    X_unknown: pandas.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.text_data = self.config.connection.data_text_columns != ""
        self.numerical_data = self.config.connection.data_numerical_columns != ""
        self.force_categorization = self.config.mode.category_text_columns != ""
        self.use_feature_selection = self.config.mode.feature_selection != Config.Reduction.NON
    
    # Function for reading in data to classify from database
    def read_in_data(self) -> bool:
        
        try:
            data, query, num_lines = self.datalayer.get_data(self.config.debug.num_rows, self.config.mode.train, self.config.mode.predict)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(e)

        if data is None:
            return False


        # Set the column names of the data array
        column_names = [self.config.connection.id_column] + [self.config.connection.class_column] + \
                self.config.connection.data_text_columns.split(',') + \
                self.config.connection.data_numerical_columns.split(',')
        try:
            column_names.remove("") # Remove any empty column name
        except Exception as e:
            pass
        self.dataset = pandas.DataFrame(data, columns = column_names)
        
        # Make sure the class column is a categorical variable by setting it as string
        try:
            self.dataset.astype({self.config.connection.class_column: 'str'}, copy=False)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Could not convert class column {self.config.connection.class_column} to string variable: {e}")
            
        # Extract unique class labels from dataset
        unique_classes = list(set(self.dataset[self.config.connection.class_column].tolist()))

        #TODO: Clean up the clean-up
        # Make an extensive search through the data for any inconsistencies (like NaNs and NoneType). 
        # Also convert datetime numerical variables to ordinals, i.e., convert them to the number of days 
        # or similar from a certain starting point, if any are left after the conversion above.
        self.logger.print_formatted_info(message="Consistency check")
        change = False
        percent_checked = 0
        index_length = float(len(self.dataset.index))
        try:
            for index in self.dataset.index:
                old_percent_checked = percent_checked
                percent_checked = round(100.0*float(index)/index_length)
                # TODO: \r is to have the Data checked updated, or some-such. Can it be moved?
                if self.config.io.verbose and percent_checked > old_percent_checked:
                    print("Data checked of fetched: " + str(percent_checked) + " %", end='\r')
                for key in self.dataset.columns:
                    item = self.dataset.at[index,key]

                    # Set NoneType objects  as zero or empty strings
                    if (key in self.config.connection.data_numerical_columns.split(",") or \
                        key in self.config.connection.data_text_columns.split(",")) and item == None:
                        if key in self.config.connection.data_numerical_columns.split(","):
                            item = 0
                        else:
                            item = ""
                        change = True

                    # Convert numerical datetime values to ordinals
                    elif key in self.config.connection.data_numerical_columns.split(",") and is_datetime(item):
                        item = datetime.toordinal(item)
                        change = True

                    # Set remaining numerical values that cannot be casted as integer or floating point numbers to zero, i.e., do not
                    # take then into account
                    elif key in self.config.connection.data_numerical_columns.split(",") and \
                        not (is_int(item) or is_float(item)):
                        item = 0
                        change = True

                    # Set text values that cannot be casted as strings to empty strings
                    elif key in self.config.connection.data_text_columns.split(",") and type(item) != str and not is_str(item):
                        item = ""
                        change = True

                    # Remove line breaks from text strings
                    if key in self.config.connection.data_text_columns.split(","):
                        item = item.replace('\n'," ").replace('\r'," ").strip()
                        change = True

                    # Save new value
                    if change:
                        self.dataset.at[index,key] = item
                        change = False
        except Exception as e:
            print(e)
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Something went wrong in inconsistency check at {key}: {item} ({e})")
    
        # Shuffle the upper part of the data, such that all already classified material are
        # put in random order keeping the unclassified material in the bottom
        # Notice: perhaps this operation is now obsolete, since we now use a NEWID() in the 
        # data query above
        self.logger.print_formatted_info("Shuffle data")
        num_un_pred = self.get_num_unpredicted_rows()
        self.dataset['rnd'] = np.concatenate([np.random.rand(num_lines - num_un_pred), [num_lines]*num_un_pred])
        self.dataset.sort_values(by = 'rnd', inplace = True )
        self.dataset.drop(['rnd'], axis = 1, inplace = True )

        # Use the unique id column from the data as the index column and take a copy, 
        # since it will not be used in the classification but only to reconnect each 
        # data row with classification results later on
        try:
            keys = self.dataset[self.config.connection.id_column].copy(deep = True).apply(get_rid_of_decimals)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Could not convert integer: {e}")
            
        
        try:
            self.dataset.set_index(keys.astype('int64'), drop=False, append=False, inplace=True, verify_integrity=False)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise DatasetException(f"Could not set index for dataset: {e}")
        
        self.dataset = self.dataset.drop([self.config.connection.id_column], axis = 1)


        self.keys = keys
        self.queries["read_data"] = query
        self.classes = unique_classes

        if self.config.mode.train:
            self.logger.investigate_dataset(self.dataset) # Returns True if the investigation/printing was not suppressed
        
        return True

    # Collapse all data text columns into a new column, which is necessary
    # for word-in-a-bag-technique
    def convert_textdata_to_numbers(self, label_binarizers:dict = {}, count_vectorizer: CountVectorizer = None, tfid_transformer: TfidfTransformer = None ) -> tuple:

        # Pick out the classification column. This is the 
        # "right hand side" of the problem.
        self.Y = self.dataset[self.config.connection.class_column]

        # Continue with "left hand side":
        # Prepare two empty DataFrames
        text_dataset = pandas.DataFrame()
        num_dataset = pandas.DataFrame()
        categorical_dataset = pandas.DataFrame()
        binarized_dataset = pandas.DataFrame()

        # First, separate all the text data from the numerical data, and
        # make sure to find the categorical data automatically
        if self.text_data:
            text_columns = self.config.connection.data_text_columns.split(',')
            for column in text_columns:
                if (self.config.mode.train and self.config.mode.use_categorization and  \
                    self.is_categorical_data(self.dataset[column])) or column in label_binarizers.keys():
                    categorical_dataset = pandas.concat([categorical_dataset, self.dataset[column]], axis = 1)
                    self.logger.print_info("Text data picked for categorization: ", column)
                else:
                    text_dataset = pandas.concat([text_dataset, self.dataset[column]], axis = 1)
                    self.logger.print_info("Text data NOT picked for categorization: ", column)
                 
        if self.numerical_data:
            num_columns = self.config.connection.data_numerical_columns.split(',')
            for column in num_columns:
                num_dataset = pandas.concat([num_dataset, self.dataset[column]], axis = 1)

        # For concatenation, we need to make sure all text data are 
        # really treated as text, and categorical data as categories
        if self.text_data:
            self.logger.print_info("Text Columns:", str(text_dataset.columns))
            if len(text_dataset.columns) > 0:

                text_dataset = text_dataset.applymap(str)

                # Concatenating text data such that the result is another DataFrame  
                # with a single column
                text_dataset = text_dataset.agg(' '.join, axis = 1)

                # Convert text data to numbers using word-in-a-bag technique
                text_dataset, count_vectorizer, tfid_transformer = self.word_in_a_bag_conversion(text_dataset, count_vectorizer, tfid_transformer)

            # Continue with handling the categorical data.
            # Binarize the data, resulting in more columns.
            if len(categorical_dataset.columns) > 0:
                categorical_dataset = categorical_dataset.applymap(str)
                generate_new_label_binarizers = (len(label_binarizers) == 0)
                for column in categorical_dataset.columns:
                    lb = None
                    if generate_new_label_binarizers:
                        lb = LabelBinarizer()
                        lb.fit(categorical_dataset[column])
                        label_binarizers[column] = lb
                    else:
                        lb = label_binarizers[column]
                    lb_results = lb.transform(categorical_dataset[column])
                    try:
                        if lb_results.shape[1] > 1:
                            lb_results_df = pandas.DataFrame(lb_results, columns=lb.classes_)
                            self.logger.print_info(f"Column {column} was categorized with categories: {lb.classes_}")
                        else:
                            lb_results_df = pandas.DataFrame(lb_results, columns=[column])
                            self.logger.print_info(f"Column {column} was binarized")
                    except ValueError as e:
                        self.logger.print_warning(f"Column {column} could not be binarized: {e}")
                    binarized_dataset = pandas.concat([binarized_dataset, lb_results_df], axis = 1 )

        self.X = pandas.DataFrame()
        if self.text_data and text_dataset.shape[1] > 0:
            text_dataset.set_index(self.dataset.index, drop=False, append=False, inplace=True, \
                                   verify_integrity=False)
            self.X = pandas.concat([text_dataset, self.X], axis = 1)
        if self.numerical_data and num_dataset.shape[1] > 0:
            num_dataset.set_index(self.dataset.index, drop=False, append=False, inplace=True, \
                                  verify_integrity=False)
            self.X = pandas.concat([num_dataset, self.X], axis = 1)
        if self.text_data and binarized_dataset.shape[1] > 0:
            binarized_dataset.set_index(self.dataset.index, drop=False, append=False, inplace=True, \
                                        verify_integrity=False)
            self.X = pandas.concat([binarized_dataset, self.X], axis = 1)

        if self.text_data:
            self.logger.print_formatted_info("After conversion of text data to numerical data")
            self.logger.investigate_dataset( self.X, False, False )

        return label_binarizers, count_vectorizer, tfid_transformer
    
    def set_training_data(self) -> None:
        # We need to know the number of data rows that are currently not
        # classified (assuming that they are last in the dataset because they
        # were sorted above). Separate keys from data elements that are
        # already classified from the rest.
        num_un_pred = self.get_num_unpredicted_rows()
        self.unpredicted_keys = self.keys[-num_un_pred:]

        # Original training data are stored for later reference
        if num_un_pred > 0:
            self.X_original = self.dataset.head(-num_un_pred)
        else:
            self.X_original = self.dataset.copy(deep=True)

    # Use the bag of words technique to convert text corpus into numbers
    def word_in_a_bag_conversion(self, dataset: pandas.DataFrame, count_vectorizer: CountVectorizer = None, tfid_transformer: TfidfTransformer = None ) -> tuple:

        # Start working with datavalues in array
        X = dataset.values

        # Find actual languange if stop words are used
        if count_vectorizer == None:
            my_language = None
            if self.config.mode.use_stop_words:
                try:
                    my_language = langdetect.detect(' '.join(X))
                except Exception as e:
                    my_language = self.STANDARD_LANG
                    self.logger.print_warning(f"Language could not be detected automatically: {e}. Fallback option, use: {my_language}.")
                else:
                    self.logger.print_info(f"Detected language is: {my_language}")

            
            # Calculate the lexical richness
            try:
                lex = LexicalRichness(' '.join(X)) 
                self.logger.print_info("#Words, #Terms and TTR for original text is {0}, {1}, {2:5.2f} %".format(lex.words,lex.terms,100*float(lex.ttr)))
            except Exception as e:
                self.logger.print_warning(f"Could not calculate lexical richness: {e}")

        # Mask all material by encryption (optional)
        if (self.config.mode.hex_encode):
            X = do_hex_base64_encode_on_data(X)

        # Text must be turned into numerical feature vectors ("bag-of-words"-technique).
        # If selected, remove stop words
        if count_vectorizer == None:
            my_stop_words = None
            if self.config.mode.use_stop_words:

                # Get the languange specific stop words and encrypt them if necessary
                my_stop_words = get_stop_words(my_language)
                self.logger.print_info("Using standard stop words: ", str(my_stop_words))
                if (self.config.mode.hex_encode):
                    for word in my_stop_words:
                        word = cipher_encode_string(str(word))

                # Collect text specific stop words (already encrypted if encryption is on)
                text_specific_stop_words = []
                if self.config.mode.specific_stop_words_threshold < 1.0:
                    try:
                        stop_vectorizer = CountVectorizer(min_df = self.config.mode.specific_stop_words_threshold)
                        stop_vectorizer.fit_transform(X)
                        text_specific_stop_words = stop_vectorizer.get_feature_names()
                        self.logger.print_info("Using specific stop words: ", text_specific_stop_words)
                    except ValueError as e:
                        self.logger.print_warning(f"Specified stop words threshold at {self.config.mode.specific_stop_words_threshold} generated no stop words.")
                    
                    my_stop_words = sorted(set(my_stop_words + text_specific_stop_words))
                    self.logger.print_info("Total list of stop words:", my_stop_words)

            # Use the stop words and count the words in the matrix        
            count_vectorizer = CountVectorizer(stop_words = my_stop_words)
            count_vectorizer.fit(X)

        # Do the word in a bag now
        X = count_vectorizer.transform(X)

        # Also generate frequencies instead of occurences to normalize the information.
        if tfid_transformer == None:
            tfid_transformer = TfidfTransformer(use_idf = False)
            tfid_transformer.fit(X) 

        # Generate the sequences
        X = (tfid_transformer.transform(X)).toarray()

        return pandas.DataFrame(X), count_vectorizer, tfid_transformer

    # Feature selection (reduction) function for PCA or Nystroem transformation of data.
    # (RFE feature selection is built into the model while training and does not need to be
    # considered here.)
    # NB: This used to take number_of_compponents, from mode.num_selected_features
    def perform_feature_selection(self, feature_selection_transform = None ):
        # Early return if we shouldn't use feature election, or the selection is RFE
        if (not self.use_feature_selection) or (self.config.mode.feature_selection == Config.Reduction.RFE):
            return
            
        t0 = time.time()
        #X, self.config.mode.num_selected_features, feature_selection_transform = \
        #    self.perform_feature_selection( X, self.config.mode.num_selected_features, feature_selection_transform )
        
        # For only predictions, use the saved transform associated with trained model
        if feature_selection_transform is not None:
            self.X = feature_selection_transform.transform(self.X)

        # In the case of training, we compute a new transformation
        else:
            if self.config.mode.feature_selection.has_transformation_function():
                self.logger.print_info(f"{self.config.mode.feature_selection.value} transformation of dataset under way...")
                self.X, feature_selection_transform = self.config.mode.feature_selection.call_transformation(
                    self.logger, self.X, self.config.mode.num_selected_features)


        t = time.time() - t0
        self.logger.print_info(f"Feature reduction took {str(round(t,2))}  sec.")
        return feature_selection_transform
        
    # Split dataset into training and validation parts
    def split_dataset(self):
        num_lower = self.get_num_unpredicted_rows()
        
        # First, split X and Y in two parts: the upper part, to be used in training,
        # and the lower part, to be classified later on
        [X_upper, X_lower] = np.split(self.X, [self.X.shape[0]-num_lower], axis = 0)
        [Y_upper, Y_lower] = np.split(self.Y, [self.Y.shape[0]-num_lower], axis = 0)

        # Split-out validation dataset from the upper part (do not use random order here)
        testsize = float(self.config.mode.test_size)
        X_train, X_validation, Y_train, Y_validation = train_test_split( 
            X_upper, Y_upper, test_size = testsize, shuffle = False, random_state = None, stratify = None
            )

        self.X_train = X_train
        self.X_validation = X_validation
        self.Y_train = Y_train
        self.Y_validation = Y_validation
        self.Y_unknown = Y_lower
        self.X_unknown = X_lower
        
        return True
        
    # Find out if a DataFrame column contains categorical data or not
    def is_categorical_data(self, column):

        is_categorical = column.value_counts().count() <= self.config.LIMIT_IS_CATEGORICAL \
            or (self.force_categorization and column.name in self.config.mode.category_text_columns.split(","))

        return is_categorical

     # Calculate the number of unclassified rows in data matrix
    def get_num_unpredicted_rows(self):
        num = 0
        for item in self.dataset[self.config.connection.class_column]:
            if item == None or not str(item).strip(): # Corresponds to empty string and SQL NULL
                num += 1
        return num

# Convert dataset to unreadable hex code
def do_hex_base64_encode_on_data(X):

    XX = X.copy()

    with np.nditer(XX, op_flags=['readwrite'], flags=["refs_ok"]) as iterator:
        for x in iterator:
            xval = str(x)
            xhex = cipher_encode_string(xval)
            x[...] = xhex 

    return XX

def cipher_encode_string(a):

    aa = a.split()
    b = ""
    for i in range(len(aa)):
        b += (str(
                base64.b64encode(
                    bytes(aa[i].encode("utf-8").hex(),"utf-8")
                )
            ) + " ")

    return b.strip()

def get_rid_of_decimals(x) -> int:
    return int(round(float(x)))

# Help routines for determining consistency of input data
def is_float(val):
    try:
        float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        int(val)
    except ValueError:
        return False
    return True

def is_bool(val):
    return type(val)==bool

def is_str(val):
    is_other_type = \
        is_float(val) or \
        is_int(val) or \
        is_bool(val) or \
        is_datetime(val)
    return not is_other_type 
    
def is_datetime(val):
    try:
        if isinstance(val, datetime): #Simple, is already instance of datetime
            return True
    except ValueError:
        pass
    # Harder: test the value for many different datetime formats and see if any is correct.
    # If so, return true, otherwise, return false.
    the_formats = ['%Y-%m-%d %H:%M:%S','%Y-%m-%d','%Y-%m-%d %H:%M:%S.%f','%Y-%m-%d %H:%M:%S,%f', \
                    '%d/%m/%Y %H:%M:%S','%d/%m/%Y','%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S,%f']
    for the_format in the_formats:
        try:
            date_time = datetime.strptime(str(val), the_format)
            return isinstance(date_time, datetime)
        except ValueError:
            pass
    return False