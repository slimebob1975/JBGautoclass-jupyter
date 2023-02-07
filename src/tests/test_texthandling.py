import pandas as pd

from JBGTextHandling import TextDataToNumbersConverter

# One class per class in the module
class TestTextDataToNumbersConverter():
    
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
   
