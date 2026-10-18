"""Testes do avaliador LLM (sem torch/transformers)."""
from pathlib import Path

import pytest

import llm_models as L

LEGACY = Path(__file__).resolve().parents[1] / "src" / "legacy" / "llm_models.py"


class CountingLLM(L.BaseLLM):
    name = "counting"

    def __init__(self, cache=None, reply='[0.9, 0.6, 0.3, "ok"]'):
        super().__init__(cache)
        self.reply = reply
        self.prompts = []

    def _generate_batch(self, prompts):
        self.prompts += prompts
        return [self.reply] * len(prompts)


# ---------------------------------------------------------------- prompt
def _legacy_prompt_lines():
    src = LEGACY.read_text(encoding="utf-8")
    body = src.split('prompt = f"""', 1)[1].split('"""', 1)[0]
    return body.replace('\\"', '"').splitlines()


def test_template_identico_ao_legado():
    lines = _legacy_prompt_lines()
    tpl = L.PROMPT_TEMPLATE.splitlines()
    assert len(lines) == len(tpl)
    for a, b in zip(lines, tpl):
        if "{" in a:  # linhas de substituição
            continue
        assert a == b
    for needle in ("### ROLE", "### METRICS", "### FEW-SHOT EXAMPLES", "Classic kinematics",
                   "dim_corr . 0 (wrong) → 1 (perfect)", "Think step-by-step silently"):
        assert needle in L.PROMPT_TEMPLATE


def test_build_prompt_equivale_ao_fstring_legado():
    eq, ctx = "v = 9.81*t", "Data columns: x"
    legacy = "\n".join(_legacy_prompt_lines())
    expected = legacy.replace("{equation_str}", eq).replace(
        '{f"Context:\\n{context}" if context else ""}', f"Context:\n{ctx}")
    assert L.build_prompt(eq, ctx) == expected
    sem = legacy.replace("{equation_str}", eq).replace('{f"Context:\\n{context}" if context else ""}', "")
    assert L.build_prompt(eq, "") == sem


# ---------------------------------------------------------------- parser
def test_parse_primeira_lista_valida():
    s, ok = L.parse_llm_response('bla ["x"] [0.3, 0.6, 0.9, "a"] [0, 0, 0]')
    assert ok and s == pytest.approx(0.6)


def test_parse_recorta():
    s, ok = L.parse_llm_response('[2, -1, 0.5, "x"]')
    assert ok and s == pytest.approx(0.5)


def test_parse_sem_lista():
    assert L.parse_llm_response("no list here") == (0.5, False)
    assert L.parse_llm_response("[0.1, 0.2]") == (0.5, False)


def test_parse_ignora_few_shot_quando_so_texto_gerado():
    generated = ' [0.2, 0.2, 0.2, "bad"]'
    s, ok = L.parse_llm_response(generated)
    assert ok and s == pytest.approx(0.2)
    # o prompt inteiro começaria pela lista do few-shot (0.95, 0.80, 0.92)
    s_full, _ = L.parse_llm_response(L.build_prompt("v = t", "") + generated)
    assert s_full != pytest.approx(0.2)


# ---------------------------------------------------------------- cache
def test_cache_evita_segunda_chamada_e_persiste(tmp_path):
    path = str(tmp_path / "c.jsonl")
    llm = CountingLLM(cache=L.ScoreCache(path))
    a = llm.score_many([("k1", "v = t")], "ctx")
    b = llm.score_many([("k1", "v = t")], "ctx")
    assert a == b == [pytest.approx(0.6)]
    assert llm.n_calls == 1 and llm.n_cached == 1
    llm2 = CountingLLM(cache=L.ScoreCache(path))
    assert llm2.score_many([("k1", "v = t")], "ctx") == [pytest.approx(0.6)]
    assert llm2.n_calls == 0 and llm2.prompts == []
    llm2.score_many([("k1", "v = t")], "outro contexto")  # contexto diferente → nova pergunta
    assert llm2.n_calls == 1


def test_chaves_duplicadas_no_lote():
    llm = CountingLLM()
    out = llm.score_many([("k", "v = t"), ("k", "v = t"), ("j", "v = 2*t")], "")
    assert len(out) == 3 and out[0] == out[1]
    assert llm.n_calls == 2 and len(llm.prompts) == 2


def test_falha_conta():
    llm = CountingLLM(reply="sem lista")
    assert llm.score_many([("k", "v = t")], "") == [0.5]
    assert llm.n_fail == 1
    llm.reset_stats()
    assert (llm.n_calls, llm.n_cached, llm.n_fail, llm.seconds) == (0, 0, 0, 0.0)


# ---------------------------------------------------------------- variantes A–H
GT = "9.81*t"


def ctx(pid, sc="ball_drop"):
    return L.context_for(pid, sc, gt=GT, target="v")


@pytest.mark.parametrize("sc", ["ball_drop", "shm", "wave"])
def test_composicao_prompts(sc):
    B, C, D = (L.context_for(p, sc, gt=GT, target="y") for p in "BCD")
    assert L.context_for("A", sc, gt=GT) == ""
    assert B.startswith("Data columns:") and all(f"{f} =" in B for f in ("m", "L", "h", "b", "t"))
    assert "Target: y =" in B
    assert C.startswith("Experiment description:")
    assert D == f"By the way, the right formula is {GT}"
    E, F, G, H = (L.context_for(p, sc, gt=GT, target="y") for p in "EFGH")
    assert E == f"{B}\n{C}" and F == f"{B}\n{D}" and G == f"{C}\n{D}" and H == f"{B}\n{C}\n{D}"
    for p in "ABCE":
        assert GT not in L.context_for(p, sc, gt=GT, target="y")


def test_rotulos_e_ids():
    assert tuple(L.PROMPT_LABELS) == L.PROMPT_IDS
    assert L.PROMPT_LABELS["A"] == "No context" and L.PROMPT_LABELS["D"] == "Formula at end"
    with pytest.raises(ValueError):
        L.context_for("D", "ball_drop")


# ---------------------------------------------------------------- LLMs falsos
def test_prompt_aware():
    llm = L.make_llm("prompt_aware", gt=GT)
    items = [("gt", f"v = {GT}"), ("wrong", "v = sin(m) + h")]
    good, bad = llm.score_many(items, ctx("D"))
    assert good == pytest.approx(1.0) and good > bad
    assert llm.score_many(items, ctx("E")) == [0.5, 0.5]
    g2, b2 = llm.score_many(items, ctx("H"))
    assert g2 > b2


def test_dummy_e_evaluate_equation():
    llm = L.make_llm("dummy")
    assert llm.evaluate_equation("v = 9.81*t") == 0.5
    assert llm.evaluate_equation("v = 9.81*t", context="x") == 0.5
    assert llm.score_many([("a", "v = t")], "") == [0.5]


def test_make_llm_desconhecido():
    with pytest.raises(ValueError):
        L.make_llm("gpt")


def test_hierarquia_hf():
    # o módulo foi importado sem torch/transformers (carregados só dentro de HFLLM)
    assert issubclass(L.MistralLLM, L.HFLLM) and issubclass(L.FalconLLM, L.HFLLM)
