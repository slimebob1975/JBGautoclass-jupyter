import pytest
import pandas as pd

from JBGTransformers import TextDataToNumbersConverter

class TestTextDataToNumbersConverter():
    """ Tests the transform TextDataToNumbersConverter """
    def test_transform(self):
        list_of_columns = [
            [31, 19, 74, 111, 2],
            ['M', 'M', 'F', 'M', 'F'],
            [20, 25, 23, 50, 37],
            ['Sweden', 'Norway', 'Denmark', 'Denmark', 'Norway'],
            ['He was always late for school', 'He had to work day and night', 'She was sick', 'He had to do his brother homework', 'Unknown story for this person']
        ]
        index = [11,22,33,44,55]
        dataset = pd.concat([pd.Series(col) for col in list_of_columns], axis=1)
        dataset.index = index
        dataset.columns = ["id", "sex", "age", "country", "story"]
        
        # Construct converter object
        ttnc = TextDataToNumbersConverter(
            text_columns=['story'],
            category_columns=['sex','country'],
            limit_categorize=3,
            language='en',
            stop_words=True,
            df=1.0,
            use_encryption=False)

        # Fit converter to dataframe
        ttnc.fit(dataset)

        # Convert dataframe
        dataset = ttnc.transform(dataset)
        after_rows = [
            [31, 20, 1.0, 2.0, 0.5774, 0.0000, 0.0000, 0.0000, 0.5774, 0.0000, 0.0000, 0.5774, 0.0, 0.0000, 0.0000, 0.0000],
            [19, 25, 1.0, 1.0, 0.0000, 0.0000, 0.5774, 0.0000, 0.0000, 0.5774, 0.0000, 0.0000, 0.0, 0.0000, 0.0000, 0.5774],
            [74, 23, 0.0, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0, 0.0000, 0.0000, 0.0000],
            [111, 50, 1.0, 0.0, 0.0000, 0.7071, 0.0000, 0.7071, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0000, 0.0000, 0.0000],
            [2, 37, 0.0, 1.0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5774, 0.0000, 0.0, 0.5774, 0.5774, 0.0000],
        ]

        expected_columns = ["id", "age", "sex", "country", "always", "brother", "day", "homework", "late", "night", "person", "school", "sick", "story", "unknown", "work"]
        expected_dataset = pd.DataFrame(after_rows, columns = expected_columns)
        expected_dataset.index = index
        
        pd.testing.assert_frame_equal(dataset, expected_dataset, check_dtype=False, check_exact=False, atol=0.1, rtol=0.1)

    def test_automatically_categorizes_low_cardinality_text_columns(self):
        dataset = pd.DataFrame({
            "channel": ["email", "chat", "email", "chat", None],
            "message": [
                "payment failed after login",
                "please update my address",
                "login failed again today",
                "please change payment date",
                "account login question",
            ],
        })

        ttnc = TextDataToNumbersConverter(
            text_columns=["channel", "message"],
            category_columns=[],
            limit_categorize=2,
            language="en",
            stop_words=False,
            df=1.0,
            use_encryption=False,
        )

        ttnc.fit(dataset)

        assert ttnc.category_columns_ == ["channel"]
        assert ttnc.text_columns_ == ["message"]

        transformed = ttnc.transform(dataset)
        assert "channel" in transformed.columns
        assert "message" not in transformed.columns
        assert transformed["channel"].notna().all()


    def test_auto_categorization_can_be_disabled(self):
        dataset = pd.DataFrame({
            "channel": ["email", "chat", "email", "chat"],
        })

        ttnc = TextDataToNumbersConverter(
            text_columns=["channel"],
            category_columns=[],
            limit_categorize=2,
            language="en",
            stop_words=False,
            df=1.0,
            use_encryption=False,
            use_categorization=False,
        )

        ttnc.fit(dataset)

        assert ttnc.category_columns_ == []
        assert ttnc.text_columns_ == ["channel"]

    def test_forced_category_still_applies_when_auto_categorization_is_disabled(self):
        dataset = pd.DataFrame({
            "channel": ["email", "chat", "email", "chat"],
            "message": ["alpha one", "beta two", "gamma three", "delta four"],
        })

        ttnc = TextDataToNumbersConverter(
            text_columns=["channel", "message"],
            category_columns=["channel"],
            limit_categorize=2,
            language="en",
            stop_words=False,
            df=1.0,
            use_encryption=False,
            use_categorization=False,
        )

        ttnc.fit(dataset)

        assert ttnc.category_columns_ == ["channel"]
        assert ttnc.text_columns_ == ["message"]

    def test_auto_categorization_keeps_high_cardinality_columns_as_text(self):
        dataset = pd.DataFrame({
            "code": ["AA", "BB", "CC", "DD"],
        })

        ttnc = TextDataToNumbersConverter(
            text_columns=["code"],
            category_columns=[],
            limit_categorize=3,
            language="en",
            stop_words=False,
            df=1.0,
            use_encryption=False,
        )

        ttnc.fit(dataset)

        assert ttnc.category_columns_ == []
        assert ttnc.text_columns_ == ["code"]


    def test_auto_categorization_does_not_duplicate_forced_categories(self):
        dataset = pd.DataFrame({
            "channel": ["email", "chat", "email", "chat"],
            "message": ["alpha one", "beta two", "gamma three", "delta four"],
        })

        ttnc = TextDataToNumbersConverter(
            text_columns=["channel", "message"],
            category_columns=["channel"],
            limit_categorize=2,
            language="en",
            stop_words=False,
            df=1.0,
            use_encryption=False,
        )

        ttnc.fit(dataset)

        assert ttnc.category_columns_ == ["channel"]
        assert ttnc.text_columns_ == ["message"]
   

class TestNNClassifier3PL:
    """ Tests the estimator NNClassifier3PL """

    def test_main(self, binary_classification_dataset):
        """ This needs to be tweaked so it can actually be used """

        X, y = binary_classification_dataset
        """
        # Create the net in question
        net = NNClassifier3PL(train_split=True)
        
        # Fit the net to the data
        net.fit(X, y)
        
        # Making prediction for first 5 data points of X
        y_pred = net.predict(X[:5])
        print(y_pred)
        
        # Checking probarbility of each class for first 5 data points of X
        y_proba = net.predict_proba(X[:5])
        print(y_proba)
        """