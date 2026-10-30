from CSVColumnSchema import inspect_csv_columns


def test_quoted_text_and_numeric_hint(tmp_path):
    path = tmp_path / "shoe.csv"
    path.write_text('id,Shoe Type,title,price,rating,reviews\n'
                    '1,Men,"A, leather shoe",279,"2.9 out of 5 stars","Good, comfortable"\n'
                    '2,Women,"Blue, sneaker",399,"4.0 out of 5 stars","Fine, thanks"\n', encoding="utf-8")
    schema = inspect_csv_columns(path)
    assert schema.rows == 2
    assert set(schema.text_feature_candidates) >= {"title", "rating", "reviews"}
    assert "price" not in schema.text_feature_candidates
    assert "Shoe Type" in schema.class_candidates
