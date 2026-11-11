"""The Dark Numbers formula on the Configuration tab (revision 298c).

As in the Voila GUI, the section shows the formula of the chosen method and
alpha, with the target class and the failure policy. The formulas come from
the engine (``DarkNumberCalculator.get_formula_spec``, LaTeX) and are turned
into MathML here, which browsers draw natively: no MathJax, no network. Only
the small part of LaTeX those formulas use is understood; anything else raises
ValueError, and the page then shows the LaTeX source instead.
"""
from __future__ import annotations

import html
import re

ROOT_DEGREE = 3                      # as in the Voila GUI
_SYMBOLS = {"alpha": "\u03b1"}
_OPERATORS = {"=": "=", "+": "+", "-": "\u2212", "(": "(", ")": ")", "/": "/"}


def latex_to_mathml(latex: str) -> str:
    """MathML for the LaTeX of the engine's Dark Number formulas."""
    atoms, end = _sequence(latex, 0, None)
    if end != len(latex):
        raise ValueError("unbalanced braces")
    return ('<math display="block" class="formula"><mrow>'
            + "".join(_render(atom) for atom in atoms) + "</mrow></math>")


def _render(atom) -> str:
    base, sub, sup = atom
    if sub is not None and sup is not None:
        return f"<msubsup>{base}{sub}{sup}</msubsup>"
    if sub is not None:
        return f"<msub>{base}{sub}</msub>"
    if sup is not None:
        return f"<msup>{base}{sup}</msup>"
    return base


def _sequence(text: str, i: int, stop):
    atoms = []
    while i < len(text):
        char = text[i]
        if char == stop:
            return atoms, i + 1
        if char == "}":
            raise ValueError("unbalanced braces")
        if char.isspace():
            i += 1
        elif text.startswith("\\,", i):
            atoms.append(('<mspace width="0.17em"></mspace>', None, None))
            i += 2
        elif char == "\\":
            match = re.match(r"\\([A-Za-z]+)", text[i:])
            if not match or match.group(1) not in _SYMBOLS:
                raise ValueError("unknown command")
            atoms.append((f"<mi>{_SYMBOLS[match.group(1)]}</mi>", None, None))
            i += match.end()
        elif char in "_^":
            if not atoms:
                raise ValueError("nothing to attach a script to")
            argument, i = _argument(text, i + 1)
            base, sub, sup = atoms.pop()
            if (sub if char == "_" else sup) is not None:
                raise ValueError("double script")
            atoms.append((base, argument, sup) if char == "_" else (base, sub, argument))
        elif char == "{":
            inner, i = _sequence(text, i + 1, "}")
            atoms.append(("<mrow>" + "".join(_render(a) for a in inner) + "</mrow>", None, None))
        elif char.isalpha():
            word = re.match(r"[A-Za-z]+", text[i:]).group(0)
            atoms.append((f"<mi>{html.escape(word)}</mi>", None, None))
            i += len(word)
        elif char.isdigit():
            number = re.match(r"[0-9.]+", text[i:]).group(0)
            atoms.append((f"<mn>{number}</mn>", None, None))
            i += len(number)
        elif char in _OPERATORS:
            atoms.append((f"<mo>{html.escape(_OPERATORS[char])}</mo>", None, None))
            i += 1
        else:
            raise ValueError("unknown character")
    if stop is not None:
        raise ValueError("unbalanced braces")
    return atoms, i


def _argument(text: str, i: int):
    """A script's argument: a braced group or a single character."""
    if i >= len(text):
        raise ValueError("missing script")
    if text[i] == "{":
        inner, i = _sequence(text, i + 1, "}")
    else:
        inner, _ = _sequence(text[i], 0, None)
        i += 1
    return "<mrow>" + "".join(_render(a) for a in inner) + "</mrow>", i


def formula_cards(translate) -> dict:
    """Everything the formula card needs, for every method and alpha.

    ``types``: "METHOD:ALPHA" (the form's option values) -> calculation type,
    for the combinations the engine supports. ``formulas``: per calculation
    type its MathML (or None, with the LaTeX source) and its alpha note."""
    from Config import DarkNumberAlpha, DarkNumberMethod
    from JBGDarkNumbers import DarkNumberCalculator
    types, formulas = {}, {}
    for method in DarkNumberMethod:
        for alpha in DarkNumberAlpha:
            try:
                spec = DarkNumberCalculator.get_formula_spec(method, alpha, root_degree=ROOT_DEGREE)
            except ValueError:
                continue                                  # e.g. non-linear with separated alpha
            types[f"{method.name}:{alpha.name}"] = spec["calculation_type"]
            if spec["calculation_type"] in formulas:
                continue
            try:
                mathml = latex_to_mathml(spec["latex"])
            except ValueError:
                mathml = None
            formulas[spec["calculation_type"]] = {
                "mathml": mathml, "latex": spec["latex"],
                "note": translate(spec["alpha_note"]),
                "non_linear": spec["calculation_type"].startswith("non_linear")}
    return {"types": types, "formulas": formulas, "root_degree": ROOT_DEGREE}


def calculation_type(cards: dict, values: dict) -> str:
    """The calculation type of the saved (or submitted) method and alpha, or ""."""
    mode = (values or {}).get("mode") or {}
    return cards["types"].get(f"{mode.get('dark_number_method')}:{mode.get('dark_number_alpha')}", "")
