
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import langdetect
from stop_words import get_stop_words
import Helpers

class TextDataToNumbersConverter(TransformerMixin, BaseEstimator):

    STANDARD_LANGUAGE = 'sv'
    LIMIT_IS_CATEGORICAL = 30

    # Create new instance of converter object
    def __init__(self, text_columns: list[str] = None, category_columns: list[str] = None, \
        limit_categorize: int = LIMIT_IS_CATEGORICAL, language: str = None, \
        stop_words: bool = True, df: float = 1.0, use_encryption: bool = True):
                
        # Take care of input
        if text_columns:
            self.text_columns_ = text_columns.copy()
        else:
            self.text_columns_ = []
        if category_columns:
            self.category_columns_ = category_columns.copy()
        else:
            self.category_columns_ = []
        self.limit_categorize_ = limit_categorize
        self.language_ = language
        self.stop_words_ = stop_words
        self.df_ = df
        self.use_encryption_ = use_encryption

        # Internal transforms for conversion (placeholders)
        self.tfidvectorizer_ = None
        self.ordinalencoder_ = None
        
        return None
        
    # Fit converter to data
    def fit(self, X: pd.DataFrame):
        
        if self.text_columns_:
        
            # Investigate language option
            if not self.language_:
                try:
                    self.language_ = langdetect.detect(' '.join(X))
                except Exception as e:
                    self.language_ = TextDataToNumbersConverter.STANDARD_LANGUAGE  
            
            # Handle stop words option
            if self.stop_words_:
                the_stop_words = self._get_stop_words()
            else:
                the_stop_words = None

            # Find out what text columns in text list that are categorical and separate them into
            # list of categories
            for column in self.text_columns_:
                if self._is_categorical_column(X, column):
                    self.category_columns_.append(column)
            if self.category_columns_:
                self.text_columns_ = [col for col in self.text_columns_ if col not in self.category_columns_]

        # Prepare data for transform
        if self.text_columns_ or self.category_columns_:
            X_document, X_category = self._separate_and_encrypt_input_data(X)

        # Depending on the division of columns, create conversion objects using fit.
        if self.text_columns_:
            self.tfidvectorizer_ = TfidfVectorizer(stop_words=the_stop_words, max_df=self.df_).fit(X_document)
        if self.category_columns_:
            self.ordinalencoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_category)

    def transform(self, X: pd.DataFrame):
        
        check_is_fitted(self)

        # Prepare placeholders for converted data (are dropped silently in concat below if None)
        X_document = None
        X_category = None

        # Prepare data for transform
        if self.text_columns_ or self.category_columns_:
            X_document, X_category = self._separate_and_encrypt_input_data(X)

        # Depending on the division of columns, transform text and categories. Add new columns names.
        if self.text_columns_:
            X_document = pd.DataFrame.sparse.from_spmatrix(self.tfidvectorizer_.transform(X_document), \
                columns = self.tfidvectorizer_.get_feature_names_out(), index=X.index)
        if self.category_columns_:
            X_category = pd.DataFrame(self.ordinalencoder_.transform(X_category), \
                columns = self.ordinalencoder_.get_feature_names_out(), index=X.index)

        # Remove text and category columns from X and put conversion result there instead.
        # Concatenate matrices and column names. Any Nones are dropped silently as long as X is not None.
        if self.text_columns_ or self.category_columns_:
            X = X.drop(self.text_columns_ + self.category_columns_, axis=1)
        
        X = pd.concat([X, X_category, X_document], axis=1, ignore_index=False)

        return X
    
    # Do we really need this?
    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)
    
    # Some help functions below
    def _is_categorical_column(self, X: pd.DataFrame, column: str) -> bool:
        
        if X is None or X[column] is None:
            return X[column].value_counts().count() <= self.limit_categorize_
        else:
            return False

    def _separate_and_encrypt_input_data(self, X: pd.DataFrame):
         
        # Separate text data from categorical data and collapse text data into one "document" column
        if self.text_columns_:
            X_document = X[self.text_columns_].astype(str).agg(' '.join, axis=1)
        else:
            X_document = None
        if self.category_columns_:
            X_category = X[self.category_columns_].astype(str)
        else:
            X_category = None

        # Use encryption on document part if set
        if X_document is not None and self.use_encryption_:
            X_document = Helpers.do_hex_base64_encode_on_data(X_document)

        return X_document, X_category

    def _get_stop_words(self):

        the_stop_words = get_stop_words(self.language_)
        
        # Use encrypton of stop words if set
        if self.use_encryption_:
            for word in the_stop_words:
                word = Helpers.cipher_encode_string(str(word))

        return the_stop_words


def main():
    print("Testing text to numbers conversion!")
    
    # Construct a fake pandas dataframe
    list_of_columns = [
        [31, 19, 74, 111, 2],
        ['M', 'M', 'F', 'M', 'F'],
        [20, 25, 23, 50, 37],
        ['Sweden', 'Norway', 'Denmark', 'Denmark', 'Norway'],
        ['He was always late for school', 'He had to work day and night', 'She was sick', 'He had to do his brother homework', 'Unknown story for this person']
    ]
    df = pd.concat([pd.Series(col) for col in list_of_columns], axis=1)
    df.index = [11,22,33,44,55]
    df.columns = ['id', 'sex', 'age', 'country','story']
    print(df.head)

    # Construct converter object
    ttnc = TextDataToNumbersConverter(text_columns=['story'], category_columns=['sex','country'], \
        limit_categorize=3, language='en', stop_words=True, df=1.0, use_encryption=False)

    # Fit converter to dataframe
    ttnc.fit(df)

    # Convert dataframe
    df = ttnc.transform(df)
    print(df.head)

# Start main
if __name__ == "__main__":
    main()
