"""Trilha do código legado: roda o código ORIGINAL dos autores (src/legacy/) com patches mínimos.

Os originais não são editados: src/ é copiado para uma pasta de reprodução e os patches são
aplicados por troca de texto, cada uma conferida com assert (se o arquivo não for o esperado,
o patch falha em vez de seguir silenciosamente).
"""
import ast
import os
import re
import shutil
import subprocess
import sys

HELPER = '''
# --- [patch reprodução] utilitários injetados -------------------------------------------
import os as _os, gc as _gc, random as _random
from llm_models import DummyLLM as _DummyLLM


def _env_list(name, default):
    v = _os.environ.get(name)
    return v.split(",") if v else list(default)


_LLM_CACHE = {}


def _get_llm(name, cls):
    """Mantém um único LLM carregado por vez (o original recarregava o 7B a cada config)."""
    if name not in _LLM_CACHE:
        _LLM_CACHE.clear()
        _gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        _LLM_CACHE[name] = _DummyLLM() if _os.environ.get("SRLLM_A_DUMMY") else cls()
    return _LLM_CACHE[name]


_A_GA_POP = int(_os.environ.get("SRLLM_A_GA_POP", 20))
_A_GA_GENS = int(_os.environ.get("SRLLM_A_GA_GENS", 5))
_A_OUTER = int(_os.environ.get("SRLLM_A_OUTER", 20))
# ------------------------------------------------------------------------------------------
'''

SEED = ('if __name__ == "__main__":\n'
        '    _s = int(_os.environ.get("SRLLM_SEED", 0)); np.random.seed(_s); _random.seed(_s)\n')

FILTERS = ('\nSR_CLASSES = {k: v for k, v in SR_CLASSES.items() if k in _env_list("SRLLM_A_SRS", SR_CLASSES)}\n'
           'LLM_CLASSES = {k: v for k, v in LLM_CLASSES.items() if k in _env_list("SRLLM_A_LLMS", LLM_CLASSES)}\n')


def _sub(text, old, new, count=1, regex=False):
    n = len(re.findall(old, text)) if regex else text.count(old)
    assert n == count, f"patch esperava {count} ocorrência(s) de {old[:60]!r}, achou {n}"
    return re.sub(old, new, text) if regex else text.replace(old, new)


def _patch_main_claim(t):
    t = _sub(t, "from expression_tree import ExpressionTree\n",
             "from expression_tree import ExpressionTree\n" + HELPER)
    t = _sub(t, 'LLM_CLASSES = {"mistral": MistralLLM, "llama": LlamaLLM, "falcon": FalconLLM}\n',
             'LLM_CLASSES = {"mistral": MistralLLM, "llama": LlamaLLM, "falcon": FalconLLM}\n' + FILTERS)
    t = _sub(t, 'experiments = ["ball_drop", "shm", "wave"]',
             'experiments = _env_list("SRLLM_A_EXPS", ["ball_drop", "shm", "wave"])')
    old_loop = (
        "        for exp in experiments:\n"
        "            X, y = generate_synthetic_data(exp)\n"
        "            truth = get_right_form(exp)\n"
        "\n"
        "            for sr_name, SRClass in SR_CLASSES.items():\n"
        "                for llm_name, LLMClass in LLM_CLASSES.items():\n"
        "                    sr = SRClass()\n"
        "                    llm = LLMClass()\n")
    new_loop = (
        "        _DATA = {}\n"
        "        for llm_name, LLMClass in LLM_CLASSES.items():\n"
        "            for exp in experiments:\n"
        "                if exp not in _DATA:\n"
        "                    _DATA[exp] = generate_synthetic_data(exp)\n"
        "                X, y = _DATA[exp]\n"
        "                truth = get_right_form(exp)\n"
        "                for sr_name, SRClass in SR_CLASSES.items():\n"
        "                    sr = SRClass()\n"
        "                    llm = _get_llm(llm_name, LLMClass)\n")
    t = _sub(t, old_loop, new_loop)
    t = _sub(t, "pop_size=20,", "pop_size=_A_GA_POP,")
    t = _sub(t, "generations=5,", "generations=_A_GA_GENS,")
    t = _sub(t, "for _ in range(20):", "for _ in range(_A_OUTER):")
    # o parser original não aceita menos unário (ex.: "-0.5*x"): registra NaN em vez de abortar
    t = _sub(t, "                    pred.build_from_string(expr)\n"
                "                    dist = pred.distance(gt)\n",
             "                    try:\n"
             "                        pred.build_from_string(expr)\n"
             "                        dist = pred.distance(gt)\n"
             "                    except Exception:\n"
             "                        dist = float('nan')\n")
    t = _sub(t, 'if __name__ == "__main__":\n', SEED)
    return t


def _patch_prompt_sensitivity(t):
    t = _sub(t, "from expression_tree import ExpressionTree\n",
             "from expression_tree import ExpressionTree\n" + HELPER)
    t = _sub(t, 'LLM_CLASSES = {"mistral": MistralLLM, "llama": LlamaLLM, "falcon": FalconLLM}\n',
             'LLM_CLASSES = {"mistral": MistralLLM, "llama": LlamaLLM, "falcon": FalconLLM}\n' + FILTERS)
    t = _sub(t, 'experiments = ["ball_drop", "shm", "wave"]',
             'experiments = _env_list("SRLLM_A_EXPS", ["ball_drop", "shm", "wave"])')
    old_loop = (
        "        for prompt_id, tmpl in prompt_templates.items():\n"
        "            for sr_name, SRClass in SR_CLASSES.items():\n"
        "                for llm_name, LLMClass in LLM_CLASSES.items():\n")
    new_loop = (
        "        for llm_name, LLMClass in LLM_CLASSES.items():\n"
        "            for sr_name, SRClass in SR_CLASSES.items():\n"
        "                for prompt_id, tmpl in [(k, v) for k, v in prompt_templates.items()\n"
        "                                        if k in _env_list(\"SRLLM_A_PROMPTS\", prompt_templates)]:\n")
    t = _sub(t, old_loop, new_loop)
    # TypeError: generate_synthetic_data devolve (X, y), não um DataFrame
    t = _sub(t, r"(?s)df = generate_synthetic_data\(exp\)\n.*?y = df\[target\]\.values\n",
             "X, y = generate_synthetic_data(exp)\n", regex=True)
    t = _sub(t, "    pred = ExpressionTree(); pred.build_from_string(predicted_expr)\n"
                "    dist = float(pred.distance(gt))\n",
             "    try:\n"
             "        pred = ExpressionTree(); pred.build_from_string(predicted_expr)\n"
             "        dist = float(pred.distance(gt))\n"
             "    except Exception:  # parser original não aceita menos unário\n"
             "        dist = float('nan')\n")
    t = _sub(t, "llm = LLMClass()", "llm = _get_llm(llm_name, LLMClass)")
    t = _sub(t, "pop_size=8,", "pop_size=_A_GA_POP,")
    t = _sub(t, "generations=3,", "generations=_A_GA_GENS,")
    t = _sub(t, 'if __name__ == "__main__":\n', SEED)
    return t


def _patch_llm_models(t, fix_full_text=False):
    # f-string com "\n" dentro de {...}: SyntaxError em Python < 3.12
    t = _sub(t, '{f"Context:\\n{context}" if context else ""}', "{_ctx(context)}", count=4)
    t = _sub(t, "from typing import Optional, Tuple\n",
             "from typing import Optional, Tuple\n\n\n"
             "def _ctx(context):\n    return \"Context:\\n\" + context if context else \"\"\n")
    # max_memory fixo para A100 de 80 GB; device_map='auto' decide sozinho
    t = _sub(t, r'\n\s*max_memory=\{0: "35GiB", "cpu": "80GiB"\},', "", count=4, regex=True)
    if fix_full_text:
        t = _sub(t, "num_return_sequences=1, max_new_tokens=128)",
                 "num_return_sequences=1, max_new_tokens=128, return_full_text=False)", count=4)
    return t


PATCHES = [
    ("ga_weight_optimizer.py", "import 'src.constants' -> 'constants' (ModuleNotFoundError com cwd=src/)",
     lambda t, o: _sub(t, "from src.constants import", "from constants import")),
    ("llm_models.py", "f-string compatível com Python < 3.12; remove max_memory fixo"
                      " (+ opcional: return_full_text=False)",
     lambda t, o: _patch_llm_models(t, o.get("fix_full_text", False))),
    ("run_main_claim.py", "LLM carregado 1x (laço do LLM por fora); distância NaN se o parser falhar; filtros/orçamentos via env; seed",
     lambda t, o: _patch_main_claim(t)),
    ("run_prompt_sensitivity.py", "corrige TypeError (X, y = generate_synthetic_data); LLM 1x; distância NaN se o parser falhar; filtros; seed",
     lambda t, o: _patch_prompt_sensitivity(t)),
]


def prepare(repo_dir, out_dir, fix_full_text=False):
    """Copia o código original (src/legacy/ deste repositório, ou src/ de um clone do upstream
    em 1508a37) para out_dir e aplica os patches mínimos. Retorna a lista do que foi alterado."""
    legacy = os.path.join(repo_dir, "src", "legacy")
    src = legacy if os.path.isdir(legacy) else os.path.join(repo_dir, "src")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(src, out_dir,
                    ignore=shutil.ignore_patterns("__pycache__", "results", "images", "__init__.py"))
    applied = []
    for fname, desc, fn in PATCHES:
        p = os.path.join(out_dir, fname)
        with open(p) as f:
            t = f.read()
        t2 = fn(t, {"fix_full_text": fix_full_text})
        ast.parse(t2)  # o arquivo modificado precisa compilar
        with open(p, "w") as f:
            f.write(t2)
        applied.append((fname, desc))
    return applied


def run_script(script, cwd, env_extra):
    """Roda um script original com saída ao vivo (equivalente a `python -u script`)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", **{k: str(v) for k, v in env_extra.items()})
    proc = subprocess.Popen([sys.executable, "-u", script], cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{script} terminou com código {proc.returncode}")


def _load_generator(repro_dir):
    """Carrega só generate_synthetic_data/get_right_form do run_main_claim original
    (sem importar transformers/torch)."""
    import numpy as np

    src = open(os.path.join(repro_dir, "run_main_claim.py")).read()
    tree = ast.parse(src)
    ns = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("generate_synthetic_data", "get_right_form"):
            exec(compile(ast.Module([node], []), "run_main_claim.py", "exec"), ns)
    return ns["generate_synthetic_data"], ns["get_right_form"]


_SHARED_NAMES = ("constants", "expression_tree", "sr_models", "ga_weight_optimizer",
                 "llm_models", "run_main_claim", "run_prompt_sensitivity")


def c1_check(repro_dir, srs=("deap", "gplearn", "pysr")):
    """Executa o check C1 isolando os módulos legados, que têm os mesmos nomes dos módulos
    novos de src/ (constants, sr_models, ...). Restaura o estado de sys.modules ao final."""
    saved = {n: sys.modules.pop(n) for n in _SHARED_NAMES if n in sys.modules}
    try:
        return _c1_check(repro_dir, srs)
    finally:
        for n in _SHARED_NAMES:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
        if repro_dir in sys.path:
            sys.path.remove(repro_dir)


def _c1_check(repro_dir, srs):
    """C1: com a mesma semente, mudar os pesos (w1,w2,w3) ou o prompt (A x D) muda a equação
    do SR no código original? Usa um LLM falso cujo score depende 100% do prompt."""
    import random
    import warnings

    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sys.path.insert(0, repro_dir)
    import sr_models
    from ga_weight_optimizer import GAWeightOptimizer

    gen, right = _load_generator(repro_dir)

    class PromptAwareLLM:
        def evaluate_equation(self, eq, context=None):
            return 1.0 if (context and "right formula" in context) else 0.0

    np.random.seed(0)
    random.seed(0)
    X, y = gen("ball_drop")
    truth = right("ball_drop")
    classes = {"deap": (sr_models.DEAPModel, {}),
               "gplearn": (sr_models.GplearnModel, {"random_state": 42}),
               "pysr": (sr_models.PySRModel, {"random_state": 42, "deterministic": True,
                                              "parallelism": "serial"})}
    rows = []
    for sr_name in srs:
        cls, kw = classes[sr_name]
        for pid, prompt in (("A", ""), ("D", f"By the way, the right formula is {truth}")):
            for w in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
                np.random.seed(42)
                random.seed(42)
                sr = cls(**kw)
                ga = GAWeightOptimizer(sr, PromptAwareLLM(), X, y, truth, "ball_drop", sr_name, "fake",
                                       results_csv=None, prompt_text=prompt)
                loss = ga._loss_function(np.array(w))
                rows.append({"SR": sr_name, "prompt": pid, "pesos (w1,w2,w3)": w,
                             "loss do GA": round(loss, 3), "equação": sr.get_best_equation()})
    df = pd.DataFrame(rows)
    df["equações distintas p/ este SR"] = df.groupby("SR")["equação"].transform("nunique")
    return df
