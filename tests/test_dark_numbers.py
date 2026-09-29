import pandas as pd
import pytest

from JBGDarkNumbers import DarkNumberCalculator


def test_published_linear_dark_number_formula_is_preserved():
    # Confusion matrix: TN=94, FP=2, FN=1, TP=3.
    real = pd.Series([0] * 96 + [1] * 4)
    predicted = pd.Series([0] * 94 + [1] * 2 + [0] + [1] * 3)

    result = DarkNumberCalculator().compute_dark_number(real, predicted, corr=1.0)
    expected = (2 / 96) * (1 + 1 / 4)

    assert result == pytest.approx(expected)


def test_multiclass_dark_numbers_use_one_vs_rest_for_each_target():
    real = pd.Series(["A", "A", "B", "B", "C", "C"])
    predicted = pd.Series(["A", "B", "B", "C", "C", "A"])
    probabilities = pd.Series([0.9, 0.7, 0.8, 0.6, 0.9, 0.65])
    corrs = {"A": 1.0, "B": 1.0, "C": 1.0}

    calculator = DarkNumberCalculator()
    results = calculator.compute_dark_numbers(
        real,
        predicted,
        probabilities,
        type="base",
        corrs=corrs,
    )

    for target in ("A", "B", "C"):
        real_binary = (real == target).astype(int)
        predicted_binary = (predicted == target).astype(int)
        expected = calculator.compute_dark_number(real_binary, predicted_binary, corr=1.0)
        actual = results.loc[results["target"] == target, "dark_number"].iloc[0]
        assert actual == pytest.approx(expected)


def test_dark_numbers_can_be_restricted_to_one_target_class():
    real = pd.Series(["Ja", "Ja", "Nej", "Nej"])
    predicted = pd.Series(["Ja", "Nej", "Nej", "Nej"])
    probabilities = pd.Series([0.9, 0.7, 0.95, 0.9])

    results = DarkNumberCalculator().compute_dark_numbers(
        real,
        predicted,
        probabilities,
        type="base",
        corrs={"Ja": 2.0},
        targets=["Ja"],
    )

    assert results["target"].tolist() == ["Ja"]
    assert results["corr"].tolist() == [2.0]


def test_dark_numbers_reject_unknown_target_class():
    real = pd.Series(["Ja", "Nej"])
    predicted = pd.Series(["Ja", "Nej"])
    probabilities = pd.Series([0.9, 0.9])

    with pytest.raises(ValueError, match="not present"):
        DarkNumberCalculator().compute_dark_numbers(
            real, predicted, probabilities, type="base", corrs={}, targets=["Missing"]
        )


def test_single_alpha_uses_false_negative_probability_when_fp_is_zero():
    real = pd.Series([0, 0, 1, 1])
    predicted = pd.Series([0, 0, 0, 1])
    probabilities = pd.Series([0.95, 0.90, 0.70, 0.95])

    alpha, _ = DarkNumberCalculator().compute_dark_number_single_alpha(
        real,
        predicted,
        probabilities,
    )

    assert alpha == pytest.approx(0.70)


def test_non_linear_alpha_uses_false_negative_probability_when_fp_is_zero():
    real = pd.Series([0, 0, 1, 1])
    predicted = pd.Series([0, 0, 0, 1])
    probabilities = pd.Series([0.95, 0.90, 0.70, 0.95])

    alpha, _ = DarkNumberCalculator().compute_dark_number_non_linear(
        real,
        predicted,
        probabilities,
        use_alpha=True,
    )

    assert alpha == pytest.approx(0.70)


@pytest.mark.parametrize(
    ("method", "alpha", "calculation_type", "fragment"),
    [
        ("LINEAR", "NONE", "base", r"(1 - TN_r)"),
        ("LINEAR", "SINGLE", "single_alpha", r"\alpha"),
        ("LINEAR", "SEPARATED", "separated_alpha", r"\alpha_{FP}"),
        ("NON_LINEAR", "NONE", "non_linear", r"TN_r^{1/3}"),
        ("NON_LINEAR", "SINGLE", "non_linear_alpha", r"\alpha"),
    ],
)
def test_formula_spec_resolves_to_implemented_calculator_branch(method, alpha, calculation_type, fragment):
    spec = DarkNumberCalculator.get_formula_spec(method, alpha, root_degree=3)

    assert spec["calculation_type"] == calculation_type
    assert fragment in spec["latex"]


def test_formula_spec_rejects_unimplemented_non_linear_separated_alpha():
    with pytest.raises(ValueError, match="Unsupported Dark Number method/alpha combination"):
        DarkNumberCalculator.get_formula_spec("NON_LINEAR", "SEPARATED")
