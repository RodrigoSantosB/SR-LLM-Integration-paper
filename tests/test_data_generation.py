import math

import numpy as np
import pytest

import data_generation as DG
from constants import (BIAS_FRACTION, DAMPING_RANGE, FEATURES, GRAVITY, HEIGHT_RANGE, LENGTH_RANGE,
                       MASS_RANGE, MAX_STEPS, SHM_SPRING_K, TARGET_SAMPLES)

N = 20  # experimentos por teste (rápido)
STEPS = 100
COL = {f: i for i, f in enumerate(FEATURES)}


def eval_gt(scenario, X):
    rhs = DG.SCENARIOS[scenario]["gt"].split("=", 1)[1]
    ns = {"sqrt": np.sqrt, "cos": np.cos, "exp": np.exp, **{f: X[:, i] for f, i in COL.items()}}
    return eval(rhs, {"__builtins__": {}}, ns) * np.ones(len(X))


@pytest.mark.parametrize("scenario", list(DG.SCENARIOS))
def test_shapes_and_ranges(scenario):
    X, y, meta = DG.generate_clean(scenario, n_experiments=N, steps=STEPS)
    assert X.shape == (N * STEPS, 5) and y.shape == (N * STEPS,)
    assert DG.SCENARIOS[scenario]["features"] == FEATURES == ("m", "L", "h", "b", "t")
    for f, (lo, hi) in zip("mLhb", (MASS_RANGE, LENGTH_RANGE, HEIGHT_RANGE, DAMPING_RANGE)):
        assert X[:, COL[f]].min() >= lo and X[:, COL[f]].max() <= hi
    assert (X[:, COL["t"]] >= 0).all()
    assert len(meta["implied_steps"]) == N and (meta["implied_steps"] <= MAX_STEPS).all()


@pytest.mark.parametrize("scenario", list(DG.SCENARIOS))
def test_noiseless_target_equals_gt(scenario):
    X, y, _ = DG.make_dataset(scenario, noise_level=0, n_experiments=N, steps=STEPS)
    np.testing.assert_allclose(y, eval_gt(scenario, X), rtol=0, atol=1e-9)


def test_ball_drop_times_until_impact():
    X, _, _ = DG.generate_clean("ball_drop", n_experiments=N, steps=STEPS)
    assert (X[:, COL["t"]] <= np.sqrt(2 * X[:, COL["h"]] / GRAVITY)).all()


def test_shm_times_within_one_period():
    X, _, _ = DG.generate_clean("shm", n_experiments=N, steps=STEPS)
    assert (X[:, COL["t"]] <= 2 * np.pi * np.sqrt(X[:, COL["m"]] / SHM_SPRING_K)).all()


def test_wave_stop_rule_and_b_zero():
    # regra anterior ("equilibrium"): envelope < 1%, limitada a MAX_STEPS passos
    cap = MAX_STEPS * DG.WAVE_DT
    assert DG.stop_time("wave", 1.0, 1.0, 0.0, "equilibrium") == pytest.approx(cap)
    assert DG.stop_time("wave", 1.0, 1.0, 1.0, "equilibrium") == pytest.approx(2 * math.log(100))
    # regra padrão ("period"): primeiro estado repetido (um período) ou equilíbrio, o que vier antes
    assert DG.stop_time("wave", 1.0, 1.0, 0.0, "period") == pytest.approx(DG.WAVE_PERIOD)
    assert DG.stop_time("wave", 1.0, 1.0, 1.0, "period") == pytest.approx(DG.WAVE_PERIOD)
    assert DG.stop_time("wave", 1.0, 1.0, 1.0) == pytest.approx(DG.WAVE_PERIOD)
    with pytest.raises(ValueError):
        DG.stop_time("wave", 1.0, 1.0, 1.0, "nope")
    X, _, meta = DG.generate_clean("wave", n_experiments=N, steps=STEPS)
    assert (meta["t_stop"] <= DG.WAVE_PERIOD + 1e-12).all()
    # ~100 passos de dt = período/100 ("around 100 steps")
    assert (meta["implied_steps"] <= 100).all()
    _, _, meta_eq = DG.generate_clean("wave", n_experiments=N, steps=STEPS, wave_stop="equilibrium")
    assert (meta_eq["t_stop"] <= cap + 1e-12).all()
    t = X[:, COL["t"]].reshape(N, STEPS)
    assert (np.diff(t, axis=1) >= 0).all() and (t.max(axis=1) <= meta["t_stop"]).all()


@pytest.mark.parametrize("level", [0.01, 0.05])
def test_noise_std_matches_level(level):
    X, y, _ = DG.generate_clean("shm", n_experiments=N, steps=STEPS)
    Xn, yn = DG.add_noise(X, y, mode="both", level=level, seed=1)
    assert np.std(yn - y) == pytest.approx(level * y.std(), rel=0.1)
    np.testing.assert_allclose((Xn - X).std(axis=0), level * X.std(axis=0), rtol=0.1)
    assert abs(np.mean(yn - y)) < 4 * level * y.std() / np.sqrt(len(y))


def test_biased_noise_mean():
    X, y, _ = DG.generate_clean("wave", n_experiments=N, steps=STEPS)
    Xn, yn = DG.add_noise(X, y, mode="features", level=0.01, biased=True, seed=2)
    mu = BIAS_FRACTION * X.mean(axis=0)
    se = 0.01 * X.std(axis=0) / np.sqrt(len(X))
    assert (np.abs((Xn - X).mean(axis=0) - mu) < 5 * se).all()


def test_noise_modes_and_no_inplace():
    X, y, _ = DG.generate_clean("ball_drop", n_experiments=N, steps=STEPS)
    X0, y0 = X.copy(), y.copy()
    Xf, yf = DG.add_noise(X, y, mode="features")
    assert np.array_equal(yf, y) and not np.array_equal(Xf, X)
    Xt, yt = DG.add_noise(X, y, mode="target")
    assert np.array_equal(Xt, X) and not np.array_equal(yt, y)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)
    with pytest.raises(ValueError):
        DG.add_noise(X, y, mode="xyz")


def test_same_seed_same_data():
    a = DG.make_dataset("wave", noise_mode="both", n_experiments=N, steps=STEPS, seed=3)
    b = DG.make_dataset("wave", noise_mode="both", n_experiments=N, steps=STEPS, seed=3)
    c = DG.make_dataset("wave", noise_mode="both", n_experiments=N, steps=STEPS, seed=4)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("scenario", list(DG.SCENARIOS))
def test_snr_40db_at_1pct(scenario):
    X, y, _ = DG.generate_clean(scenario, n_experiments=N, steps=STEPS)
    _, yn = DG.add_noise(X, y, mode="target", level=0.01)
    assert DG.snr_db(y, yn) == pytest.approx(40.0, abs=1.0)


def test_default_size_is_50000():
    X, y, meta = DG.make_dataset("ball_drop")
    assert X.shape == (TARGET_SAMPLES, 5) and len(y) == TARGET_SAMPLES
