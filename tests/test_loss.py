"""Loss composta (Eq. 1) e early stopping (p.6)."""
import math

import numpy as np
import pytest

from loss import EarlyStopper, Objective, check_weights


class CountingScorer:
    def __init__(self, value=0.8):
        self.value, self.calls = value, 0

    def score_many(self, items, context):
        self.calls += len(items)
        return [self.value] * len(items)


Y = np.array([1.0, 2.0, 3.0, 4.0])


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        check_weights((0.5, 0.5, 0.5))
    assert check_weights((1, 0, 0)) == (1.0, 0.0, 0.0)


def test_only_error_term():
    obj = Objective(Y, "v", (1, 0, 0))
    r = obj.evaluate([("x", Y + 1.0)])[0]
    assert r["loss"] == pytest.approx(1.0)  # MSE cru (escala "paper")


def test_paper_scaling_uses_raw_size():
    obj = Objective(Y, "v", (0, 1, 0))
    assert obj.evaluate([("a + b", Y)])[0]["loss"] == 3.0


def test_c_is_one_minus_mean_score():
    sc = CountingScorer(0.8)
    obj = Objective(Y, "v", (0, 0, 1), scorer=sc)
    assert obj.evaluate([("x", Y)])[0]["loss"] == pytest.approx(0.2)


def test_baseline_does_not_call_llm():
    sc = CountingScorer()
    obj = Objective(Y, "v", (0.75, 0.25, 0.0), scorer=sc)
    obj.evaluate([("x", Y), ("y", Y)])
    assert sc.calls == 0


def test_non_finite_prediction_fails_without_llm():
    sc = CountingScorer()
    obj = Objective(Y, "v", (0.5, 1 / 6, 1 / 3), scorer=sc)
    r = obj.evaluate([("x", np.array([np.nan, 1, 2, 3]))])[0]
    assert r["loss"] >= 1e12 and sc.calls == 0


def test_llm_receives_target_equals_expression():
    seen = []

    class Spy:
        def score_many(self, items, context):
            seen.extend(items)
            return [0.5] * len(items)

    Objective(Y, "v", (0.5, 1 / 6, 1 / 3), scorer=Spy(), rename={"x2": "h"}).evaluate([("x2", Y)])
    assert seen[0][1] == "v = h"


def test_normalized_scaling():
    obj = Objective(Y, "v", (1, 0, 0), scaling="normalized")
    assert obj.evaluate([("x", Y + 1.0)])[0]["loss"] == pytest.approx(1.0 / np.var(Y), rel=1e-6)


def test_early_stopping_window_of_three():
    es = EarlyStopper()
    assert not es.update(10.0)
    assert not es.update(5.0)
    assert not es.update(5.0)
    assert not es.update(5.0)   # 10 → 5 em 3 gerações: melhorou
    assert es.update(5.0)       # 5 → 5 em 3 gerações: < 0,1%


def test_early_stopping_keeps_running_while_improving():
    es = EarlyStopper()
    assert not any(es.update(v) for v in [10, 9, 8, 7, 6, 5])
