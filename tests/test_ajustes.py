"""Ajustes pós-piloto (docs/CONFORMIDADE.md, seção 7)."""
import csv

import numpy as np
import pytest

import constants as C
import data_generation as DG
import expression_tree as ET
import llm_models as L
import loss
import runner

GT = {k: v["gt"] for k, v in DG.SCENARIOS.items()}


# ---------------------------------------------------------------- verdade-base
def test_gt_has_no_neutral_nodes():
    assert "+ 0.0" not in GT["shm"] and "1.0*L" not in GT["wave"]
    assert GT["shm"] == "x = L*cos(sqrt(1.0/m)*t)"
    assert GT["wave"].startswith("E = h*exp(-b*t/2)*cos(L - ")


# ---------------------------------------------------------------- distância canônica e estrutura
@pytest.mark.parametrize("pred", ["sqrt(19.62*h)", "(19.62*h)**0.5", "4.43*sqrt(h)", "4.43*h**0.5",
                                  "mul(4.43, psqrt(h))"])
def test_equivalent_forms_ball_drop(pred):
    assert ET.canonical_distance(pred, GT["ball_drop"]) < 0.01
    assert ET.structure_match(pred, GT["ball_drop"]) == 1


def test_equivalent_forms_shm_and_wave():
    assert ET.canonical_distance("L*cos(t/sqrt(m))", GT["shm"]) == 0.0
    assert ET.structure_match("L*cos(sqrt(1/m)*t)", GT["shm"]) == 1
    assert ET.structure_match("h*exp(-0.5*b*t)*cos(L-6.28*t)", GT["wave"]) == 1
    assert ET.canonical_distance("h*exp(-0.5*b*t)*cos(L-6.28*t)", GT["wave"]) < 0.01


def test_wrong_structures():
    for pred in ("0.32*h + 14", "3.91*9.13", "cos(t)", "L"):
        assert ET.structure_match(pred, GT["ball_drop"]) == 0
    assert ET.canonical_distance("0.32*h + 14", GT["ball_drop"]) == 1.0
    # crédito parcial: estrutura certa com uma constante a mais
    assert 0 < ET.canonical_distance("L*cos(0.9*t/sqrt(m))", GT["shm"]) < 0.5


def test_literal_distance_unchanged_and_fallback():
    assert ET.distance("(19.62*h)**0.5", GT["ball_drop"]) == 1.0  # literal: sem equivalência
    assert ET.canonical_distance("((", GT["ball_drop"]) == 1.0      # não analisável
    assert ET.structure_match("((", GT["ball_drop"]) == 0
    assert ET.canonical("1/(h-h)") is None                           # zoo → sem forma canônica


def test_canonical_is_fast_on_bloated_expressions():
    import time
    big = "(-(h)+-0.509)"
    for _ in range(11):
        big = f"({big}*-0.509 + (h - t - log(h)))"
    t0 = time.time()
    ET.canonical_distance(big, GT["ball_drop"])
    assert time.time() - t0 < 2.0


# ---------------------------------------------------------------- consulta ao LLM
class FakeTok:
    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        return f"<s>[INST] {msgs[0]['content']} [/INST]"


def test_format_query_with_and_without_template():
    assert L.format_query(FakeTok("x"), "P", True, "[") == "<s>[INST] P [/INST]["
    assert L.format_query(FakeTok(None), "P", True, "[") == "User: P\nAssistant: ["
    assert L.format_query(FakeTok("x"), "P", False, "") == "P"


class PrefixLLM(L.BaseLLM):
    """Simula um modelo que continua a lista pré-preenchida (ou a repete)."""
    name = "prefix"
    response_prefix = "["

    def __init__(self, outs, cache=None):
        super().__init__(cache)
        self.outs = list(outs)

    def _generate_batch(self, prompts):
        return [self.outs.pop(0) for _ in prompts]


def test_prefix_is_used_in_parsing():
    llm = PrefixLLM(['0.9, 0.6, 0.3, "ok"]', '[0.3, 0.3, 0.3, "repeat"]', "texto livre"])
    s = llm.score_many([("a", "v = a"), ("b", "v = b"), ("c", "v = c")])
    assert s == pytest.approx([0.6, 0.3, 0.5])
    assert llm.n_calls == 3 and llm.n_fail == 1


def test_cache_key_has_format_and_failures_are_counted_from_cache(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = L.ScoreCache(str(path))
    assert cache.key("m", "", "e").startswith(C.LLM_QUERY_FORMAT + "|")
    llm = PrefixLLM(["sem lista", '1, 1, 1, "ok"]'], cache=cache)
    llm.score_many([("a", "v = a"), ("b", "v = b")])
    # nova instância lendo do disco: respostas vêm do cache e a falha é contada
    llm2 = PrefixLLM([], cache=L.ScoreCache(str(path)))
    assert llm2.score_many([("a", "v = a"), ("b", "v = b")]) == pytest.approx([0.5, 1.0])
    assert llm2.n_calls == 0 and llm2.n_cached == 2 and llm2.n_cached_fail == 1
    # um cache de outro formato de consulta não é reaproveitado
    old = L.ScoreCache(str(path), query_format="formato-antigo")
    assert old.get(old.key("prefix", "", "a")) is None


# ---------------------------------------------------------------- parada antecipada
def test_early_stopper_min_generations():
    lit, adj = loss.EarlyStopper(), loss.EarlyStopper(min_generations=10)
    stops_lit = [lit.update(1.0) for _ in range(12)]
    stops_adj = [adj.update(1.0) for _ in range(12)]
    assert stops_lit.index(True) == 3          # literal: 4ª entrada (geração 3)
    assert stops_adj.index(True) == 10         # só depois de 10 gerações


# ---------------------------------------------------------------- operadores extras e perfis
def test_extra_operators_in_models():
    import sr_models
    d = sr_models.DEAPModel(list(C.FEATURES), extra_unary=C.EXTRA_OPS["deap"], init_depth=(1, 3))
    assert {"pexp", "psqrt"} <= set(d.primitive_names) and d.params["init_depth"] == (1, 3)
    g = sr_models.GplearnModel(list(C.FEATURES), extra_unary=C.EXTRA_OPS["gplearn"])
    assert "sqrt" in g.function_names
    p = sr_models.PySRModel(list(C.FEATURES), extra_unary=C.EXTRA_OPS["pysr"])
    assert "sqrt" in p.unary_operators
    assert np.allclose(sr_models.psqrt([-4.0, 9.0]), [2.0, 3.0])


def test_profiles_and_verify_mode():
    lit = runner.make_config("full")
    adj = runner.make_config("full", "adjusted")
    assert lit["profile"] == "literal" and lit["loss_scaling"] == "paper" and not lit["extra_ops"]
    assert lit["min_generations"] == 0 and lit["deap_init_depth"] == C.DEAP["init_depth"]
    assert adj["loss_scaling"] == "normalized" and adj["extra_ops"] and adj["min_generations"] == 10
    v = runner.make_config("verify")
    assert v["n_reps"] == 1 and v["llms"] == ["mistral", "llama"] and v["prompts"] == ["A", "D"]
    with pytest.raises(ValueError):
        runner.make_config("full", "nope")


def test_make_sr_applies_profile():
    cfg = runner.make_config("smoke", "adjusted")
    m = runner.make_sr("deap", cfg, seed=0)
    assert "psqrt" in m.primitive_names and m.min_generations == 10
    assert runner.make_sr("deap", runner.make_config("smoke"), 0).min_generations == 0


def test_run_one_records_new_columns():
    cfg = runner.make_config("smoke", "adjusted", n_experiments=6, steps=5, gens=2, pop=10)
    row = runner.run_one(runner.Spec("ball_drop", "deap", "dummy", "A"), cfg, L.DummyLLM())
    for c in ("distance", "distance_literal", "structure_match", "llm_fail_rate", "llm_fail_rate_new"):
        assert c in row
    assert row["profile"] == "adjusted" and 0.0 <= row["distance"] <= 1.0


def test_result_store_rejects_old_header(tmp_path):
    p = tmp_path / "old.csv"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerow(["key", "MAE"])
    with pytest.raises(ValueError, match="outra versão"):
        runner.ResultStore(str(p))


def test_cli_run_dir_per_profile(tmp_path):
    import cli
    args = cli.parser("x").parse_args(["--mode", "smoke", "--profile", "adjusted", "--out", str(tmp_path)])
    cfg = cli.config_from(args)
    assert cfg["run_dir"].endswith("smoke_adjusted") and cfg["profile"] == "adjusted"
