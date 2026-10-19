"""Avaliadores LLM do termo c da loss composta.

"each LLM returns a concise, three-element list, including a score about physical
dimensionality (c1) consistency, syntactic and structural simplicity (c2), and physical realism
(c3). The LLM's overall score is computed as follows: c := 1 − (c1 + c2 + c3)/3" (p.4–5).

Aqui ``score_many`` devolve a média (c1+c2+c3)/3 ∈ [0, 1]; o ``Objective`` (loss.py) faz c = 1 − média.

- PROMPT_TEMPLATE: cópia literal do prompt de src/legacy/llm_models.py (ROLE / METRICS /
  FEW-SHOT / TASK, incluindo o bloco ``Context:\\n{context}``).
- Variantes A–H (LLM prompt knowledge integration experiment, p.6–7): os textos exatos estão no
  material suplementar (indisponível); B e C foram reescritos a partir da descrição do artigo,
  D mantém a redação do código original ("By the way, the right formula is ...").
- HFLLM: transformers + bitsandbytes NF4 4-bit, decodificação gulosa ("we set the sampling
  temperature to zero", p.6), em lote com padding à esquerda, decodificando só os tokens novos.
- ScoreCache: JSONL persistente; como a geração é determinística, a mesma pergunta não é refeita.
  A chave inclui o formato da consulta (LLM_QUERY_FORMAT) e o cache guarda se a resposta foi
  interpretada; respostas recuperadas do cache contam nas estatísticas de falha.
- [AJUSTE pós-piloto] Consulta com o modelo de conversa (chat template) do modelo e resposta
  pré-preenchida com "[" (LLM_CHAT_TEMPLATE, LLM_RESPONSE_PREFIX): no piloto, o LLaMA devolvia
  sequências vazias e o Falcon texto livre em 86–94% das consultas com o prompt em texto puro.
- DummyLLM / PromptAwareLLM: LLMs falsos para testes de integração (sem torch).
"""
import ast
import gc
import hashlib
import json
import os
import re
import time

from constants import (LLM_CHAT_TEMPLATE, LLM_MAX_NEW_TOKENS, LLM_MODELS, LLM_QUERY_FORMAT,
                       LLM_RESPONSE_PREFIX, PROMPT_IDS)

# ---------------------------------------------------------------- prompt (idêntico ao código original)
PROMPT_TEMPLATE = """### ROLE
You are an expert scientific-reasoning assistant.
Return ONLY a Python-style list: [dim_corr, simp, sim, "feedback"]

### METRICS
dim_corr . 0 (wrong) → 1 (perfect)
simp     . 0 (complex) → 1 (simple)
sim      . 0 (unrealistic) → 1 (realistic)

### FEW-SHOT EXAMPLES
#1 Equation: x = v0 * t + 0.5 * g * t^2
   Output:  [0.95, 0.80, 0.92, "Classic kinematics"]
#2 Equation: E = m + c
   Output:  [0.05, 0.70, 0.15, "Units mismatch"]
#3 Equation: y = sin(sin(x))
   Output:  [0.90, 0.10, 0.40, "Needless nesting"]

### TASK
Equation to evaluate:
{equation}

{context_block}

Think step-by-step silently, then output the list only."""


def build_prompt(equation_text, context=None):
    """Prompt completo; reproduz ``{f"Context:\\n{context}" if context else ""}`` do original."""
    block = f"Context:\n{context}" if context else ""
    return PROMPT_TEMPLATE.format(equation=equation_text, context_block=block)


# ---------------------------------------------------------------- variantes A–H (p.6–7)
PROMPT_LABELS = {
    "A": "No context", "B": "Variable descriptions", "C": "Experiment description",
    "D": "Formula at end", "E": "B+C", "F": "B+D", "G": "C+D", "H": "B+C+D",
}
# Composição de cada variante a partir dos blocos B, C e D (mesma estrutura do original).
PROMPT_PARTS = {
    "A": (), "B": ("B",), "C": ("C",), "D": ("D",),
    "E": ("B", "C"), "F": ("B", "D"), "G": ("C", "D"), "H": ("B", "C", "D"),
}
assert tuple(PROMPT_PARTS) == PROMPT_IDS

# B — "data-column descriptions only": as cinco colunas de constants.FEATURES em SI + o alvo.
DESCRIPTION_B = {
    "ball_drop": ("Data columns: m = mass of the ball (kg), L = characteristic length, ball radius (m), "
                  "h = initial drop height (m), b = linear air-drag coefficient (kg/s), "
                  "t = time since release (s)."),
    "shm": ("Data columns: m = oscillating mass (kg), L = characteristic length, spring amplitude (m), "
            "h = initial displacement (m), b = linear damping coefficient (kg/s), t = time (s)."),
    "wave": ("Data columns: m = mass (kg), L = characteristic length, observation position x (m), "
             "h = initial wave amplitude E0, b = damping coefficient alpha (1/s), t = time (s)."),
}
TARGET_DESCRIPTION = {
    "ball_drop": "speed of the ball (m/s)",
    "shm": "displacement of the mass from equilibrium (m)",
    "wave": "wave field magnitude at position x (same units as E0)",
}
# C — "concise, natural-language 'Experiment description'".
DESCRIPTION_C = {
    "ball_drop": ("Experiment description: a ball is released from rest at height h near the Earth's "
                  "surface under constant gravity (g = 9.81 m/s^2) and falls until it hits the ground; "
                  "we measure its speed as a function of time."),
    "shm": ("Experiment description: a mass attached to an ideal spring oscillates without friction "
            "(undamped simple harmonic motion); we measure its displacement from equilibrium over time."),
    "wave": ("Experiment description: a damped plane wave is observed at a fixed position x; the field "
             "oscillates in time and its amplitude decays exponentially with the damping coefficient; "
             "we measure the field magnitude over time."),
}
# D — "appended the GT equation" (redação do código original, run_prompt_sensitivity.py).
DESCRIPTION_D = "By the way, the right formula is {gt}"


def context_for(prompt_id, scenario, gt=None, target=None):
    """Texto de contexto da variante ``prompt_id`` (A–H) para o cenário. ``gt`` é a equação
    verdadeira (data_generation.SCENARIOS) e ``target`` o nome da coluna-alvo; o prefixo
    "Context:" é acrescentado por build_prompt, como no original. A → ""."""
    parts = PROMPT_PARTS[prompt_id]
    if "D" in parts and not gt:
        raise ValueError(f"prompt {prompt_id} exige a equação GT (gt)")
    blocks = {}
    if "B" in parts:
        tgt = TARGET_DESCRIPTION[scenario]
        blocks["B"] = f"{DESCRIPTION_B[scenario]} Target: {target} = {tgt}." if target else \
            f"{DESCRIPTION_B[scenario]} Target: {tgt}."
    if "C" in parts:
        blocks["C"] = DESCRIPTION_C[scenario]
    if "D" in parts:
        blocks["D"] = DESCRIPTION_D.format(gt=gt)
    return "\n".join(blocks[p] for p in parts)


# ---------------------------------------------------------------- resposta
_LIST_RE = re.compile(r"\[.*?\]", flags=re.DOTALL)


def parse_llm_response(text):
    """Primeira lista Python válida com ≥3 elementos numéricos iniciais no texto GERADO →
    (média dos três primeiros recortados em [0, 1], True). Falha → (0.5, False), o fallback
    do código original."""
    for m in _LIST_RE.findall(text or ""):
        try:
            parsed = ast.literal_eval(m)
            if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
                continue
            head = parsed[:3]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in head):
                continue
            vals = [max(0.0, min(1.0, float(v))) for v in head]
            if any(v != v for v in vals):  # NaN
                continue
            return sum(vals) / 3.0, True
        except Exception:
            continue
    return 0.5, False


# ---------------------------------------------------------------- cache persistente
class ScoreCache:
    """Cache (formato, modelo, sha1(contexto)[:12], expressão) → (score, interpretada), em JSONL.

    Entradas gravadas com outro formato de consulta (outra chave) são ignoradas: um cache antigo
    pode ser reaproveitado sem reutilizar os escores neutros das falhas do formato anterior."""

    def __init__(self, path=None, query_format=LLM_QUERY_FORMAT):
        self.path = path
        self.query_format = query_format
        self.mem = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.mem[r["key"]] = (float(r["score"]), bool(r.get("ok", True)))
                    except Exception:  # linha corrompida (escrita interrompida)
                        pass

    def key(self, model_name, context, expr):
        h = hashlib.sha1((context or "").encode("utf-8")).hexdigest()[:12]
        return f"{self.query_format}|{model_name}|{h}|{expr}"

    def get(self, k):
        """(score, interpretada) ou None."""
        return self.mem.get(k)

    def put(self, k, score, ok=True, raw=""):
        self.mem[k] = (float(score), bool(ok))
        if self.path:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "score": float(score), "ok": bool(ok),
                                    "raw": (raw or "")[:300]}) + "\n")

    def __len__(self):
        return len(self.mem)


# ---------------------------------------------------------------- avaliadores
class BaseLLM:
    """Interface comum. Subclasses implementam ``_generate_batch(prompts) -> list[str]``."""
    name = "base"
    response_prefix = ""  # texto pré-preenchido na resposta (devolvido sem ele por _generate_batch)

    def __init__(self, cache=None):
        self.cache = cache if cache is not None else ScoreCache()
        self.reset_stats()

    def reset_stats(self):
        self.n_calls = 0     # perguntas efetivamente enviadas ao LLM
        self.n_cached = 0    # respostas servidas pelo cache
        self.n_fail = 0      # respostas novas sem lista válida (score 0.5)
        self.n_cached_fail = 0  # respostas do cache que também não tinham lista válida
        self.seconds = 0.0   # tempo de geração

    def _generate_batch(self, prompts):
        raise NotImplementedError

    def score_many(self, items, context=""):
        """items: lista de (chave canônica, texto "alvo = expr"). Retorna a média (c1+c2+c3)/3
        de cada item. Chaves repetidas no lote geram uma única pergunta."""
        context = context or ""
        out, todo = {}, {}
        for key, text in items:
            k = self.cache.key(self.name, context, key)
            if k in out or k in todo:
                continue
            v = self.cache.get(k)
            if v is None:
                todo[k] = text
            else:
                out[k] = v[0]
                self.n_cached += 1
                self.n_cached_fail += int(not v[1])
        if todo:
            keys = list(todo)
            t0 = time.time()
            raws = self._generate_batch([build_prompt(todo[k], context) for k in keys])
            self.seconds += time.time() - t0
            for k, raw in zip(keys, raws):
                s, ok = parse_llm_response(self.response_prefix + (raw or ""))
                if not ok and self.response_prefix:  # o modelo repetiu o "[" pré-preenchido
                    s, ok = parse_llm_response(raw)
                self.n_calls += 1
                self.n_fail += int(not ok)
                self.cache.put(k, s, ok, raw)
                out[k] = s
        return [out[self.cache.key(self.name, context, key)] for key, _ in items]

    def evaluate_equation(self, equation_str, context=None):
        """Compatibilidade com o código original: um único score em [0, 1]."""
        return self.score_many([(equation_str, equation_str)], context or "")[0]

    def unload(self):
        pass


class HFLLM(BaseLLM):
    """Modelo causal do Hugging Face, NF4 4-bit (como o original), greedy, em lote."""

    def __init__(self, model_id, name=None, cache=None, batch_size=16,
                 max_new_tokens=LLM_MAX_NEW_TOKENS, chat_template=LLM_CHAT_TEMPLATE,
                 response_prefix=LLM_RESPONSE_PREFIX, quantize=True, cache_dir=None,
                 trust_remote_code=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        super().__init__(cache)
        self.name = name or model_id
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir,
                                                 trust_remote_code=trust_remote_code)
        self.chat_template = chat_template  # o original não usa template de chat
        self.response_prefix = response_prefix or ""
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if quantize:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
                else torch.float16
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            print(f"[LLM] carregando {model_id} (NF4 4-bit, {dtype})...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb, device_map="auto", torch_dtype=dtype,
                cache_dir=cache_dir, trust_remote_code=trust_remote_code)
        else:  # modelos pequenos em testes locais
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
        self.model.eval()

    def _format(self, prompt):
        return format_query(self.tok, prompt, self.chat_template, self.response_prefix)

    def _generate_batch(self, prompts):
        import torch

        outs = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = [self._format(p) for p in prompts[i:i + self.batch_size]]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=not self._templated).to(self.model.device)
            enc.pop("token_type_ids", None)  # Falcon não aceita
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # temperatura zero (p.6)
                    pad_token_id=self.tok.pad_token_id,
                    stop_strings=["]"],  # a resposta útil termina ao fechar a lista
                    tokenizer=self.tok,
                )
            new = gen[:, enc["input_ids"].shape[1]:]  # só os tokens gerados
            outs += self.tok.batch_decode(new, skip_special_tokens=True)
        return outs

    def unload(self):
        import torch

        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    @property
    def _templated(self):
        return bool(self.chat_template and getattr(self.tok, "chat_template", None))


def format_query(tok, prompt, chat_template=True, response_prefix=""):
    """Texto enviado ao modelo. Com ``chat_template`` e um tokenizador que tenha modelo de conversa
    (LLaMA-2-chat, Mistral-Instruct), usa ``apply_chat_template``; sem modelo de conversa no
    tokenizador (Falcon-instruct), usa o formato de instrução "User: ...\nAssistant:". Em seguida
    acrescenta ``response_prefix`` (início da resposta, ex. "[")."""
    if not chat_template:
        return prompt + response_prefix
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
    else:
        text = f"User: {prompt}\nAssistant:" + (" " if response_prefix else "")
    return text + response_prefix


class MistralLLM(HFLLM):
    def __init__(self, model_name=LLM_MODELS["mistral"], **kw):
        super().__init__(model_name, name=kw.pop("name", "mistral"), **kw)


class LlamaLLM(HFLLM):
    def __init__(self, model_name=LLM_MODELS["llama"], **kw):
        super().__init__(model_name, name=kw.pop("name", "llama"), **kw)


class FalconLLM(HFLLM):
    """Falcon é nativo no transformers atual: carregado sem trust_remote_code."""

    def __init__(self, model_name=LLM_MODELS["falcon"], **kw):
        super().__init__(model_name, name=kw.pop("name", "falcon"), **kw)


class DummyLLM(BaseLLM):
    """Sempre 0.5 (DummyLLM do código original)."""

    def __init__(self, name="dummy", cache=None):
        super().__init__(cache)
        self.name = name

    def _generate_batch(self, prompts):
        return ['[0.5, 0.5, 0.5, "dummy"]'] * len(prompts)


class PromptAwareLLM(BaseLLM):
    """LLM falso que "conhece" a GT quando ela aparece no contexto (prompts D, F, G, H):
    score = média de (1 − distância(expr, GT)) e da sobreposição de variáveis com a GT.
    Sem a GT no contexto devolve 0.5. Serve para verificar
    que o termo c altera a equação encontrada."""

    def __init__(self, gt, rename=None, name="prompt_aware", cache=None):
        super().__init__(cache)
        self.name = name
        self.gt = gt
        self.rename = rename

    def score_many(self, items, context=""):
        self._context = context or ""
        return super().score_many(items, context)

    def _overlap(self, eq):
        """Jaccard entre as variáveis da expressão e as da GT (dá gradiente ao escore mesmo
        quando a estrutura ainda está longe da GT)."""
        import re

        def names(txt):
            txt = txt.split("=", 1)[1] if "=" in txt else txt
            found = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", txt))
            found = {(self.rename or {}).get(n, n) for n in found}
            return found - {"sqrt", "exp", "log", "sin", "cos", "neg", "add", "sub", "mul",
                            "div", "pdiv", "plog", "pexp", "protected_div"}

        a, b = names(eq), names(self.gt)
        return len(a & b) / len(a | b) if a | b else 1.0

    def _generate_batch(self, prompts):
        import expression_tree as ET

        outs = []
        for p in prompts:
            if self.gt not in self._context:
                outs.append('[0.5, 0.5, 0.5, "no info"]')
                continue
            eq = p.split("Equation to evaluate:\n", 1)[1].split("\n", 1)[0]
            s = 0.5 * (1.0 - ET.distance(eq, self.gt, rename=self.rename)) + 0.5 * self._overlap(eq)
            outs.append(f'[{s:.6f}, {s:.6f}, {s:.6f}, "aware"]')
        return outs


def make_llm(name, cache=None, **kw):
    """Fábrica: "mistral" | "llama" | "falcon" | "dummy" | "prompt_aware" (exige gt=...)."""
    classes = {"mistral": MistralLLM, "llama": LlamaLLM, "falcon": FalconLLM}
    if name == "dummy":
        return DummyLLM(cache=cache, **kw)
    if name == "prompt_aware":
        return PromptAwareLLM(cache=cache, **kw)
    if name in classes:
        return classes[name](cache=cache, **kw)
    raise ValueError(f"LLM desconhecido: {name}")
