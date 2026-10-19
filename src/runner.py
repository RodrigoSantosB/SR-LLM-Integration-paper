"""Execução dos experimentos do artigo, com retomada e sementes pareadas.

Cada execução é uma especificação imutável (cenário, SR, LLM, prompt, modo de ruído, nível,
viés, repetição) identificada por uma chave textual. Os resultados são gravados linha a linha
num CSV; ao reexecutar, as chaves já presentes são puladas.

Sementes: os dados dependem só de (cenário, repetição) e do ruído; o SR depende de
(cenário, SR, repetição). Assim, LLMs, prompts e a linha de base são comparados com os mesmos
dados e a mesma semente do SR. Cada LLM é carregado uma única vez por bloco.

Blocos (seção "Experiment types", p.6–7):
  benchmark — 3 cenários × 3 SR × (linha de base + 3 LLMs) × n  (Tabelas 5 e 6)
  prompts   — 8 prompts × 3 LLMs × 3 SR × 3 cenários × n          (Tabela 2)
  ruído     — cenários × SR × {entradas, alvo, ambos} × 1–5% × n   (Tabelas 3 e 4)

Perfis (constants.PROFILES): "literal" (especificação do artigo) e "adjusted" (variantes
documentadas para as inconsistências da execução piloto). O perfil não entra na chave da
execução; cada perfil grava numa pasta própria (cli.config_from).
"""
import csv
import os
import time
import traceback
import zlib
from dataclasses import asdict, dataclass

import numpy as np

import constants as C
import expression_tree as ET
from data_generation import SCENARIOS, add_noise, generate_clean
from llm_models import DummyLLM, PromptAwareLLM, ScoreCache, context_for, make_llm
from loss import Objective

COLUMNS = [
    "key", "scenario", "sr", "llm", "prompt", "noise_mode", "noise_level", "biased", "rep",
    "MAE", "MSE", "one_minus_R2", "R2", "distance", "distance_literal", "structure_match",
    "size", "plaus", "loss", "equation", "raw_equation", "n_generations", "n_evaluated",
    "sr_seconds", "llm_seconds", "llm_ratio", "llm_calls", "llm_cached", "llm_fail_rate",
    "llm_fail_rate_new", "profile", "timestamp",
]

MODES = {
    # validação rápida em CPU, com LLM falso
    "smoke": dict(n_experiments=40, steps=20, pop=30, gens=4, n_reps=1, llms=["mistral"],
                  scenarios=["ball_drop"], prompts=["A", "D"], noise_levels=[0.01, 0.05],
                  fake_llm=True),
    # piloto com LLM real para medir custo
    "pilot": dict(n_reps=3, llms=["mistral"], srs=["deap"], scenarios=["ball_drop"]),
    # [AJUSTE pós-piloto] verificação curta das correções com LLMs reais, antes do protocolo
    # completo: 1 cenário × 3 SR × Mistral e LLaMA, prompts A e D, n = 1
    "verify": dict(n_reps=1, llms=["mistral", "llama"], scenarios=["ball_drop"],
                   prompts=["A", "D"], noise_levels=[0.01, 0.05]),
    # protocolo completo do artigo
    "full": dict(),
}


def make_config(mode="full", profile="literal", **overrides):
    """Configuração do protocolo do artigo, reduzida conforme o modo, com o perfil indicado."""
    if profile not in C.PROFILES:
        raise ValueError(f"perfil desconhecido: {profile} (use {list(C.PROFILES)})")
    cfg = dict(
        mode=mode, profile=profile, wave_stop=C.WAVE_STOP,
        n_experiments=C.N_EXPERIMENTS, steps=C.STEPS_PER_EXPERIMENT,
        pop=C.POPULATION_SIZE, gens=C.GENERATIONS, n_reps=C.N_REPETITIONS,
        scenarios=list(C.SCENARIOS), srs=list(C.SR_NAMES), llms=list(C.LLM_NAMES),
        prompts=list(C.PROMPT_IDS), noise_levels=list(C.NOISE_LEVELS),
        weights=C.DEFAULT_WEIGHTS, weights_baseline=C.BASELINE_WEIGHTS,
        loss_scaling=C.LOSS_SCALING,
        bench_prompt="A",          # benchmark sem informação adicional (prompt A)
        noise_llm="mistral", noise_prompt="A", noise_baseline=True,  # [DESVIO] não informado no artigo
        fake_llm=False, llm_batch_size=16, max_fit_samples=None,
        cache_path="llm_cache.jsonl",
    )
    if mode not in MODES:
        raise ValueError(f"modo desconhecido: {mode} (use {list(MODES)})")
    cfg.update(C.PROFILES[profile])
    cfg.update(MODES[mode])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


@dataclass(frozen=True)
class Spec:
    scenario: str
    sr: str
    llm: str = "none"           # "none" = linha de base sem LLM
    prompt: str = "-"
    noise_mode: str = "target"  # padrão do artigo: 1% de ruído no alvo
    noise_level: float = C.NOISE_LEVEL
    biased: bool = False
    rep: int = 0

    @property
    def key(self):
        return (f"{self.scenario}|{self.sr}|{self.llm}|{self.prompt}|{self.noise_mode}|"
                f"{self.noise_level:g}|{int(self.biased)}|{self.rep}")


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31 - 1)


def make_data(spec, cfg):
    X, y, meta = generate_clean(spec.scenario, cfg["n_experiments"], cfg["steps"],
                                seed=seed_of("data", spec.scenario, spec.rep),
                                wave_stop=cfg.get("wave_stop"))
    if spec.noise_level > 0:
        X, y = add_noise(X, y, spec.noise_mode, spec.noise_level, spec.biased,
                         seed=seed_of("noise", spec.scenario, spec.rep, spec.noise_mode,
                                      spec.noise_level, spec.biased))
    return X, y


def make_sr(name, cfg, seed):
    import sr_models  # importado aqui: o PySR deve ser carregado antes do torch
    cls = {"deap": sr_models.DEAPModel, "pysr": sr_models.PySRModel,
           "gplearn": sr_models.GplearnModel}[name]
    kw = dict(min_generations=cfg.get("min_generations", 0))
    if cfg.get("extra_ops"):
        kw["extra_unary"] = C.EXTRA_OPS[name]
    if name == "deap" and cfg.get("deap_init_depth"):
        kw["init_depth"] = tuple(cfg["deap_init_depth"])
    return cls(list(C.FEATURES), pop_size=cfg["pop"], generations=cfg["gens"], seed=seed, **kw)


def run_one(spec, cfg, scorer=None):
    sc = SCENARIOS[spec.scenario]
    X, y = make_data(spec, cfg)
    Xf, yf = X, y
    if cfg.get("max_fit_samples") and len(y) > cfg["max_fit_samples"]:
        idx = np.random.default_rng(seed_of("sub", spec.key)).choice(
            len(y), cfg["max_fit_samples"], replace=False)
        Xf, yf = X[idx], y[idx]
    use_llm = spec.llm != "none"
    weights = cfg["weights"] if use_llm else cfg["weights_baseline"]
    context = context_for(spec.prompt, spec.scenario, sc["gt"], sc["target"]) if use_llm else ""
    model = make_sr(spec.sr, cfg, seed_of("sr", spec.scenario, spec.sr, spec.rep))
    stats0 = _stats(scorer)
    obj = Objective(yf, sc["target"], weights, scorer if use_llm else None, context,
                    rename=model.rename, scaling=cfg["loss_scaling"])
    t0 = time.time()
    model.fit(Xf, obj)
    total = time.time() - t0
    with np.errstate(all="ignore"):
        yp = np.asarray(model.predict(X), float).reshape(-1)
    finite = yp.shape == y.shape and bool(np.all(np.isfinite(yp)))
    mae = float(np.mean(np.abs(y - yp))) if finite else np.nan
    mse = float(np.mean((y - yp) ** 2)) if finite else np.nan
    omr2 = mse / float(np.var(y)) if finite else np.nan
    st = {k: v - stats0[k] for k, v in _stats(scorer).items()} if use_llm else _stats(None)
    best = model.best_row or {}
    try:
        tree = ET.parse(model.best_expr, model.rename)
        equation = ET.to_infix(tree)
    except (ET.ParseError, RecursionError):
        equation = ""
    row = dict(asdict(spec))
    asked = st["n_calls"] + st["n_cached"]
    row.update(
        key=spec.key, MAE=mae, MSE=mse, one_minus_R2=omr2,
        R2=(1 - omr2) if finite else np.nan,
        # distância do artigo sobre as formas canônicas (principal) e sobre o texto literal
        distance=ET.canonical_distance(model.best_expr, sc["gt"], model.rename),
        distance_literal=ET.distance(model.best_expr, sc["gt"], model.rename),
        structure_match=ET.structure_match(model.best_expr, sc["gt"], model.rename),
        size=best.get("size", np.nan), plaus=best.get("plaus", np.nan),
        loss=best.get("loss", np.nan), equation=equation, raw_equation=model.best_expr,
        n_generations=model.n_generations, n_evaluated=obj.n_evaluated,
        sr_seconds=total - st["seconds"], llm_seconds=st["seconds"],
        llm_ratio=st["seconds"] / max(total - st["seconds"], 1e-9),
        llm_calls=st["n_calls"], llm_cached=st["n_cached"],
        # falhas sobre todas as consultas (novas e do cache) e só sobre as novas
        llm_fail_rate=((st["n_fail"] + st["n_cached_fail"]) / asked) if asked else np.nan,
        llm_fail_rate_new=(st["n_fail"] / st["n_calls"]) if st["n_calls"] else np.nan,
        profile=cfg.get("profile", "literal"),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return row


def _stats(scorer):
    if scorer is None:
        return dict(n_calls=0, n_cached=0, n_fail=0, n_cached_fail=0, seconds=0.0)
    return dict(n_calls=scorer.n_calls, n_cached=scorer.n_cached, n_fail=scorer.n_fail,
                n_cached_fail=getattr(scorer, "n_cached_fail", 0), seconds=scorer.seconds)


class ResultStore:
    """CSV com uma linha por execução; permite retomar blocos interrompidos."""

    def __init__(self, path):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and list(reader.fieldnames) != COLUMNS:
                    raise ValueError(
                        f"{path} foi gravado por outra versão do pipeline (colunas diferentes). "
                        "Use outra pasta de resultados: as execuções antigas seriam puladas na "
                        "retomada sem os ajustes atuais.")
                self.done = {r["key"] for r in reader}

    def append(self, row):
        new = not os.path.exists(self.path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        self.done.add(row["key"])


_CACHES = {}


def make_scorer(llm, cfg):
    path = cfg["cache_path"]
    if path not in _CACHES:
        _CACHES[path] = ScoreCache(path)
    if cfg.get("fake_llm"):
        return DummyLLM(name=f"dummy-{llm}", cache=_CACHES[path])
    return make_llm(llm, cache=_CACHES[path], batch_size=cfg["llm_batch_size"])


def run_specs(specs, cfg, results_csv, scorer_factory=make_scorer, log=print):
    """Executa as especificações pendentes. Retorna o número de execuções feitas agora."""
    store = ResultStore(results_csv)
    unique = list(dict.fromkeys(specs))
    pending = [s for s in unique if s.key not in store.done]
    log(f"{len(unique)} execuções no bloco, {len(pending)} pendentes -> {results_csv}")
    order = ["none"] + [l for l in dict.fromkeys(s.llm for s in pending) if l != "none"]
    done, t_start = 0, time.time()
    for llm in order:
        group = [s for s in pending if s.llm == llm]
        if not group:
            continue
        scorer = scorer_factory(llm, cfg) if llm != "none" else None
        try:
            for s in group:
                t0 = time.time()
                try:
                    row = run_one(s, cfg, scorer)
                except Exception:
                    log(f"[ERRO] {s.key}\n{traceback.format_exc()}")
                    continue
                store.append(row)
                done += 1
                eta = (time.time() - t_start) / done * (len(pending) - done)
                log(f"[{done}/{len(pending)}] {s.key} dist={row['distance']:.3f} "
                    f"MAE={row['MAE']:.4g} {time.time() - t0:.0f}s (ETA {eta / 3600:.1f} h) "
                    f"eq: {row['equation'][:60]}")
        finally:
            if scorer is not None:
                scorer.unload()
    return done


# ---------------------------------------------------------------- blocos do artigo
def specs_benchmark(cfg):
    """Benchmarking: 27 configurações e a linha de base sem LLM de cada SR (Tabelas 5 e 6)."""
    out = []
    for sc in cfg["scenarios"]:
        for sr in cfg["srs"]:
            for rep in range(cfg["n_reps"]):
                out.append(Spec(sc, sr, rep=rep))
                out += [Spec(sc, sr, llm, cfg["bench_prompt"], rep=rep) for llm in cfg["llms"]]
    return out


def specs_prompts(cfg):
    """LLM prompt knowledge integration experiment: prompts A–H (Tabela 2)."""
    return [Spec(sc, sr, llm, p, rep=rep)
            for p in cfg["prompts"] for llm in cfg["llms"] for sr in cfg["srs"]
            for sc in cfg["scenarios"] for rep in range(cfg["n_reps"])]


def specs_noise(cfg, biased):
    """Noise robustness: entradas, alvo e ambos, 1–5% (Tabela 3: média zero; 4: enviesado)."""
    llms = [cfg["noise_llm"]] + (["none"] if cfg["noise_baseline"] else [])
    out = []
    for sc in cfg["scenarios"]:
        for sr in cfg["srs"]:
            for mode in C.NOISE_MODES:
                for lvl in cfg["noise_levels"]:
                    for llm in llms:
                        for rep in range(cfg["n_reps"]):
                            out.append(Spec(sc, sr, llm, cfg["noise_prompt"] if llm != "none" else "-",
                                            mode, lvl, biased, rep))
    return out


def integration_check(cfg, scenario="ball_drop", srs=("deap", "gplearn", "pysr")):
    """Verifica que o LLM está dentro da busca: mesma semente e mesmos dados, só o prompt muda
    (A × D) com um avaliador falso que conhece a verdade-base quando ela está no prompt."""
    sc = SCENARIOS[scenario]
    rows = []
    for sr in srs:
        for p in ("A", "D"):
            scorer = PromptAwareLLM(sc["gt"])
            r = run_one(Spec(scenario, sr, "prompt_aware", p), cfg, scorer)
            rows.append({"SR": sr, "prompt": p, "equation": r["equation"],
                         "distance": round(r["distance"], 3), "loss": r["loss"],
                         "llm_calls": r["llm_calls"]})
    return rows


def estimate_remaining(cfg, results_csvs, blocks):
    """Tempo médio por execução (por SR, com e sem LLM) × execuções pendentes de cada bloco."""
    import pandas as pd

    frames = [pd.read_csv(p) for p in results_csvs if os.path.exists(p)]
    if not frames:
        return None
    df = pd.concat(frames)
    df["t"] = df["sr_seconds"] + df["llm_seconds"]
    df["com_llm"] = df["llm"] != "none"
    mean_t = df.groupby(["sr", "com_llm"])["t"].mean().to_dict()
    fallback = df["t"].mean()
    rows = []
    for name, specs, path in blocks:
        done = ResultStore(path).done
        pend = [s for s in dict.fromkeys(specs) if s.key not in done]
        secs = sum(mean_t.get((s.sr, s.llm != "none"), fallback) for s in pend)
        rows.append({"bloco": name, "total": len(set(specs)), "pendentes": len(pend),
                     "horas_estimadas": round(secs / 3600, 1)})
    return pd.DataFrame(rows)
