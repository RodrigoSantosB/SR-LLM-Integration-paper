"""Ajustes pós-verify (docs/ALTERACOES.md, A8): prefixo por modelo, leitura tolerante e cópias."""
import pytest

import constants as C
import llm_models as L
import runner


class FakeTok:
    """Tokenizador com o modelo de conversa do LLaMA-2-chat."""
    chat_template = "llama-2"

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        return f"<s>[INST] {msgs[0]['content']} [/INST]"


def test_prefix_per_model():
    assert L.response_prefix_for("llama") == "Output: ["
    assert L.response_prefix_for("falcon") == "Output: ["
    assert L.response_prefix_for("mistral") == "["
    assert L.response_prefix_for("outro") == "["
    assert L.response_prefix_for("llama", "[") == "["           # prefixo único (texto)
    q = L.format_query(FakeTok(), "P", True, L.response_prefix_for("llama"))
    assert q.endswith("[/INST]Output: [")                        # sem "[" solto após [/INST]


def test_query_format_version():
    assert C.LLM_QUERY_FORMAT == "chat-prefill-v3"
    assert L.ScoreCache().key("llama", "", "e").startswith("chat-prefill-v3|")


@pytest.mark.parametrize("text, expected", [
    ('[0.60, 0.20, 050, "x"]', (0.6 + 0.2 + 0.5) / 3),          # falha observada do Mistral
    ("[0, 0.5, 10]", 0.5),                                       # 0 e 10 não são alterados (10 → 1)
    ("[0.5, 0.5, 0.5]", 0.5),
])
def test_tolerant_parse(text, expected):
    s, ok = L.parse_llm_response(text)
    assert ok and s == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["Inst]", "Assistant]:", 'dim_corr, simp, sim, "feedback"]'])
def test_llama_failures_still_fail(raw):
    """As respostas inválidas observadas continuam sendo falhas (escore neutro), sem falso sucesso."""
    assert L.parse_llm_response("Output: [" + raw) == (0.5, False)


class ScriptedLLM(L.BaseLLM):
    name = "llama"

    def __init__(self, outs, cache=None):
        super().__init__(cache)
        self.response_prefix = L.response_prefix_for(self.name)
        self.outs = list(outs)

    def _generate_batch(self, prompts):
        return [self.outs.pop(0) for _ in prompts]


def test_copy_detection_and_cache(tmp_path):
    path = tmp_path / "c.jsonl"
    llm = ScriptedLLM(['0.95, 0.80, 0.92, "Classic kinematics"]',    # cópia do exemplo 1
                       '0.3, 0.4, 0.5, "ok"]',                        # avaliação própria
                       '[0.05, 0.70, 0.15, "Units mismatch"]'],       # cópia, repetindo o "["
                      cache=L.ScoreCache(str(path)))
    s = llm.score_many([("a", "v = a"), ("b", "v = b"), ("c", "v = c")])
    assert s == pytest.approx([(0.95 + 0.8 + 0.92) / 3, 0.4, 0.3])
    assert llm.n_calls == 3 and llm.n_fail == 0 and llm.n_copy == 2
    again = ScriptedLLM([], cache=L.ScoreCache(str(path)))
    again.score_many([("a", "v = a"), ("b", "v = b"), ("c", "v = c")])
    assert again.n_cached == 3 and again.n_cached_copy == 2


def test_run_one_records_copy_rate():
    cfg = runner.make_config("smoke", n_experiments=6, steps=5, gens=2, pop=10)
    row = runner.run_one(runner.Spec("ball_drop", "deap", "dummy", "A"), cfg, L.DummyLLM())
    assert "llm_copy_rate" in row and row["llm_copy_rate"] == 0.0
    assert "llm_copy_rate" in runner.COLUMNS
