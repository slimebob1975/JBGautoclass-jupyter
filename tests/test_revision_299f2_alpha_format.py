"""299f2 (asked 2026-11-13): the Dark Numbers calculation table showed alphas
as the engine's Python text ("[1, np.float64(0.7205267055383298)]"); they are
shown as numbers, like the Dark Number column."""
from JBGWebUIResults import alpha_text, readable_dark_table


def test_alphas_are_shown_as_numbers():
    assert alpha_text("[1, np.float64(0.7205267055383298)]", "sv") == "1; 0,7205"
    assert alpha_text("[np.float64(0.7069173509483662), 1]", "en") == "0.7069, 1"
    assert alpha_text("[1, 1]", "sv") == "1; 1"
    assert alpha_text("[0.5]", "sv") == "0,5"


def test_anything_else_is_left_as_it_is():
    for value in ("[a, 1]", "[]", "separated", 0.5, None, "[1, np.float64(x)]"):
        assert alpha_text(value, "sv") == value


def test_the_calculation_table_uses_it():
    from JBGWebUI import translator
    detail = {"columns": ["Model type", "alphas", "target"],
              "rows": [["D_cv_test - Cross-trained model", "[1, np.float64(0.72)]", "[1, 2]"]]}
    readable_dark_table(detail, translator("sv"), "sv")
    assert detail["rows"][0][1:] == ["1; 0,72", "[1, 2]"]          # a class name stays
