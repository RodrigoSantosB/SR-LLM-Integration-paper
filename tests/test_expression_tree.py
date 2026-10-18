"""Distância entre árvores e tamanho (seção "Expression tree distance metric", p.7)."""
import pytest

import expression_tree as ET

GT_BALL = "v = sqrt(2*9.81*h)"
GT_SHM = "x = L*cos(sqrt(1.0/m)*t + 0.0)"


def d(a, b):
    return ET.distance(a, b)


def test_identical_is_zero():
    assert d("a + b", "a + b") == 0.0
    assert d(GT_BALL, GT_BALL) == 0.0
    assert d(GT_SHM, GT_SHM) == 0.0


def test_commutative_pairing():
    assert d("a + b", "b + a") == 0.0
    assert d("a * b", "b * a") == 0.0


def test_non_commutative_kept():
    # "−" e "/" não são comutativos: a − b ≠ b − a
    assert d("a - b", "b - a") > 0.0
    assert d("a / b", "b / a") > 0.0


def test_different_variables_is_one():
    assert d("x", "y") == 1.0


def test_leaf_vs_internal_is_one():
    assert d("h", GT_BALL) == 1.0  # regressão: o parser original dava 0 aqui


def test_constant_difference_bounded():
    v = d("sqrt(19.6*h)", GT_BALL)
    assert 0.0 < v < 0.1
    assert d("sqrt(19.62*h)", GT_BALL) == 0.0  # constantes agrupadas (2*9.81 = 19.62)


def test_number_distance_capped_at_one():
    assert d("100", "1") == 1.0
    assert d("1.5", "1") == pytest.approx(0.05)


def test_operator_mismatch_is_one():
    assert d("a + b", "a * b") == 1.0


def test_prefix_equals_infix():
    assert d("add(mul(a, b), x)", "a*b + x") == 0.0
    assert d("protected_div(x, y)", "x / y") == 0.0
    assert d("sub(x, neg(y))", "x - (-y)") == 0.0


def test_rename_sr_variables():
    rename = {"x2": "h", "X2": "h", "ARG2": "h"}
    assert ET.distance("sqrt(19.62*x2)", GT_BALL, rename) == 0.0
    assert ET.distance("pdiv(ARG2, 2.0)", "h/2", rename) == 0.0


def test_bounded_in_unit_interval():
    for a, b in [("sin(x)*exp(t)", GT_SHM), ("x0", GT_BALL), ("1", "x - y")]:
        assert 0.0 <= d(a, b) <= 1.0


def test_unparseable_is_max_distance():
    assert d("??", GT_BALL) == 1.0


def test_size_counts_nodes():
    assert ET.size(ET.parse("a + b")) == 3
    assert ET.size(ET.parse("sub(x, x)")) == 3  # sem simplificação: o bloat é contado
    assert ET.size(ET.parse("-0.5")) == 1


def test_unary_minus_does_not_abort():
    ET.parse("-0.5*x + -(y)")


def test_legacy_interface():
    t = ET.ExpressionTree().build_from_string(GT_BALL)
    assert t.compute_complexity() == ET.size(ET.parse(GT_BALL))
    assert t.distance(ET.ExpressionTree().build_from_string(GT_BALL)) == 0.0
