"""Testes dos SR com a loss composta dentro da busca (orçamentos pequenos)."""
import math

import numpy as np
import pytest

import expression_tree as ET
from constants import BASELINE_WEIGHTS, DEAP, GPLEARN, PYSR
from loss import Objective
from sr_models import DEAPModel, DummySR, GplearnModel, PySRModel

NAMES = ["m", "L", "h", "b", "t"]
LLM_W = (0.5, 1.0 / 6.0, 1.0 / 3.0)


class FakeScorer:
    """Imita o avaliador LLM: plausibilidade 1 se `key` estiver no texto, senão 0."""

    def __init__(self, key="sin"):
        self.key, self.calls, self.seconds = key, 0, 0.0

    def score_many(self, items, context=""):
        self.calls += 1
        self.seconds += 0.001 * len(items)
        return [1.0 if self.key in txt else 0.0 for _, txt in items]


def data(n=80, col=2, coef=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 10.0, (n, len(NAMES)))
    y = coef * X[:, col] + rng.normal(0.0, 0.1, n)
    return X, y


def run(cls, X, y, weights=(1.0, 0.0, 0.0), scorer=None, scaling="paper", **kw):
    kw = {"pop_size": 30, "generations": 5, "seed": 0, **kw}
    m = cls(NAMES, **kw)
    obj = Objective(y, "y", weights, scorer=scorer, rename=m.rename, scaling=scaling)
    return m.fit(X, obj), obj


def check_fit(m, X):
    assert math.isfinite(m.best_row["loss"]) and math.isfinite(m.best_row["mse"])
    ET.parse(m.best_expr, m.rename)  # texto nativo analisável
    assert m.get_best_equation() == m.best_expr
    assert m.predict(X).shape == (len(X),)
    assert m.history and {"gen", "best_loss", "n_new", "llm_seconds"} <= set(m.history[0])
    assert m.n_generations == m.history[-1]["gen"]
    assert m.sr_seconds >= 0 and m.llm_seconds == 0


# ---------------------------------------------------------------- 1. ajuste simples
@pytest.mark.parametrize("cls", [DEAPModel, GplearnModel, DummySR])
def test_fit_simple(cls):
    X, y = data()
    m, _ = run(cls, X, y)
    check_fit(m, X)


@pytest.mark.slow
def test_fit_simple_pysr():
    X, y = data()
    m, _ = run(PySRModel, X, y)
    check_fit(m, X)
    assert m.best_row["mse"] < np.var(y)


# ---------------------------------------------------------------- 2. DEAP usa todas as colunas
def test_deap_uses_last_column():
    X, y = data(col=4, coef=1.0)  # y = t + ruído
    m, _ = run(DEAPModel, X, y, pop_size=50, generations=10)
    assert "t" in m.best_expr.replace("(", " ").replace(")", " ").replace(",", " ").split()
    assert m.best_row["mse"] < 0.5 * np.var(y)
    assert np.mean((m.predict(X) - y) ** 2) == pytest.approx(m.best_row["mse"])


# ---------------------------------------------------------------- 3. LLM dentro da busca
def _has_sin(expr):
    return "sin" in expr


@pytest.mark.parametrize("cls", [DEAPModel, GplearnModel])
def test_llm_term_changes_search(cls):
    """Hiperparâmetros da Tabela 1: mesma semente e dados; o termo c muda a trajetória."""
    X, y = data()
    plain, _ = run(cls, X, y, scaling="normalized", generations=8)
    sc = FakeScorer("sin")
    llm, _ = run(cls, X, y, weights=LLM_W, scorer=sc, scaling="normalized", generations=8)
    assert sc.calls > 0 and llm.llm_seconds > 0
    assert sc.calls == sum(h["n_new"] > 0 for h in llm.history)  # um lote LLM por geração
    assert [h["best_loss"] for h in plain.history] != [h["best_loss"] for h in llm.history]
    if cls is GplearnModel:  # a preferência do "LLM" muda a expressão escolhida
        assert _has_sin(llm.best_expr) and llm.best_row["plaus"] == 1.0
        assert not _has_sin(plain.best_expr)


def test_llm_term_steers_deap_selection():
    """Com cxpb/mutpb/profundidade da Tabela 1 a população do DEAP quase não varia em poucas
    gerações (o melhor fica na população inicial de altura 1); com mais variação, o termo c
    na fitness leva o torneio a expressões com "sin"."""
    X, y = data(seed=4)
    kw = dict(scaling="normalized", generations=8, init_depth=(1, 3), cxpb=0.5, mutpb=0.2)
    plain, _ = run(DEAPModel, X, y, **kw)
    llm, _ = run(DEAPModel, X, y, weights=LLM_W, scorer=FakeScorer("sin"), **kw)
    assert _has_sin(llm.best_expr) and llm.best_row["plaus"] == 1.0
    assert not _has_sin(plain.best_expr)


@pytest.mark.slow
def test_llm_term_changes_pysr_selection():
    X, y = data()
    plain, _ = run(PySRModel, X, y, scaling="normalized")
    sc = FakeScorer("sin")
    llm, _ = run(PySRModel, X, y, weights=LLM_W, scorer=sc, scaling="normalized")
    assert sc.calls > 0
    assert (plain.best_expr != llm.best_expr
            or plain.history[-1]["best_loss"] != llm.history[-1]["best_loss"])


# ---------------------------------------------------------------- 4. linha de base sem LLM
@pytest.mark.parametrize("cls", [DEAPModel, GplearnModel, DummySR])
def test_baseline_never_calls_scorer(cls):
    X, y = data()
    sc = FakeScorer()
    m, _ = run(cls, X, y, weights=BASELINE_WEIGHTS, scorer=sc)
    assert sc.calls == 0 and m.llm_seconds == 0
    assert math.isnan(m.best_row["plaus"]) and m.best_row["c"] == 0.0


# ---------------------------------------------------------------- 5. conformidade com a Tabela 1
def test_table1_operator_sets():
    d = DEAPModel(NAMES)
    assert set(d.pset.primitives) == {object}  # sem tipagem forte
    prims = {p.name for ps in d.pset.primitives.values() for p in ps}
    assert prims == set(DEAP["binary"]) | set(DEAP["unary"])
    assert "pexp" not in prims and "exp" not in prims
    assert {p.name for ps in DEAPModel(NAMES, extra_unary=("pexp",)).pset.primitives.values()
            for p in ps} >= {"pexp"}
    assert d.params["init_depth"] == (1, 1) and d.params["max_height"] == 8
    assert (d.params["cxpb"], d.params["mutpb"], d.params["tournament"]) == (0.05, 0.01, 3)
    assert d.rename == {n: n for n in NAMES}

    g = GplearnModel(NAMES)
    assert "exp" in g.function_names
    assert set(g.function_names) == set(GPLEARN["binary"]) | set(GPLEARN["unary"])
    assert {f if isinstance(f, str) else f.name for f in g._function_set()} >= {"exp"}
    assert g.rename["X0"] == "m" and g.rename["X4"] == "t"

    p = PySRModel(NAMES)
    assert p.binary_operators == list(PYSR["binary"])
    assert p.unary_operators == list(PYSR["unary"])
    assert p.pysr_params["populations"] == PYSR["populations"]
    assert p.pysr_params["population_size"] == 100
    assert p.pysr_params["elementwise_loss"] == PYSR["loss"]
    assert p.pysr_params["parsimony"] == PYSR["parsimony"]
