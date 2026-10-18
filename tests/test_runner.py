"""Runner: blocos do artigo, sementes pareadas e retomada."""
import csv

import numpy as np

import runner


def cfg(**kw):
    c = runner.make_config("smoke", n_experiments=6, steps=5, **kw)
    return c


def test_full_mode_is_paper_protocol():
    c = runner.make_config("full")
    assert c["n_reps"] == 25 and c["pop"] == 100 and c["gens"] == 50
    assert c["n_experiments"] == 500 and c["steps"] == 100
    assert c["llms"] == ["mistral", "llama", "falcon"] and len(c["prompts"]) == 8


def test_benchmark_has_baseline_and_27_configs():
    c = runner.make_config("full", n_reps=1)
    specs = runner.specs_benchmark(c)
    assert sum(s.llm == "none" for s in specs) == 9          # 3 cenários × 3 SR
    assert sum(s.llm != "none" for s in specs) == 27         # 3 × 3 × 3


def test_prompt_and_noise_blocks_sizes():
    c = runner.make_config("full", n_reps=1)
    assert len(runner.specs_prompts(c)) == 8 * 27
    noise = runner.specs_noise(c, biased=False)
    assert len(noise) == 3 * 3 * 3 * 5 * 2                   # cenários × SR × modos × níveis × (LLM + base)
    assert {s.noise_mode for s in noise} == {"features", "target", "both"}
    assert all(s.biased for s in runner.specs_noise(c, biased=True))


def test_benchmark_prompt_A_shares_keys():
    c = runner.make_config("full", n_reps=1)
    bench = {s.key for s in runner.specs_benchmark(c)}
    prompt_a = {s.key for s in runner.specs_prompts(c) if s.prompt == "A"}
    assert prompt_a <= bench


def test_paired_data_across_llms_and_prompts():
    c = cfg()
    a = runner.Spec("ball_drop", "deap", "mistral", "A")
    b = runner.Spec("ball_drop", "deap", "llama", "D")
    base = runner.Spec("ball_drop", "deap")
    Xa, ya = runner.make_data(a, c)
    for s in (b, base):
        X, y = runner.make_data(s, c)
        assert np.array_equal(X, Xa) and np.array_equal(y, ya)
    assert runner.seed_of("sr", "ball_drop", "deap", 0) == runner.seed_of("sr", "ball_drop", "deap", 0)
    Xr, _ = runner.make_data(runner.Spec("ball_drop", "deap", rep=1), c)
    assert not np.array_equal(Xr, Xa)


class FakeSR:
    rename = {}

    def __init__(self, *a, **k):
        self.best_expr, self.best_row, self.n_generations = "h", {}, 1

    def fit(self, X, obj):
        self.best_row = obj.evaluate([("h", X[:, 2])])[0]
        return self

    def predict(self, X):
        return X[:, 2]


def test_resume_skips_done(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_sr", lambda name, cfg, seed: FakeSR())
    c = cfg()
    c["cache_path"] = str(tmp_path / "cache.jsonl")
    path = str(tmp_path / "r.csv")
    specs = runner.specs_benchmark(c)
    assert runner.run_specs(specs, c, path, log=lambda *_: None) == len(specs)
    assert runner.run_specs(specs, c, path, log=lambda *_: None) == 0
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(specs)
    # interrompe: apaga uma linha e reexecuta só ela
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=runner.COLUMNS)
        w.writeheader()
        w.writerows(rows[:-1])
    assert runner.run_specs(specs, c, path, log=lambda *_: None) == 1
