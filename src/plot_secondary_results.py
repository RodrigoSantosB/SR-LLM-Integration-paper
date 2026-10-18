"""Tabelas no formato do artigo (Tabelas 2, 3, 4, 5 e 6), com os valores publicados ao lado,
e figuras. Substitui o plot_secondary_results.py original (preservado em legacy/), que esperava
colunas que nenhum script gerava.

Uso: python plot_secondary_results.py resultados.csv [pasta_saida] [--mode full]
"""

import os

import numpy as np
import pandas as pd

SR_ORDER = ["deap", "pysr", "gplearn"]
TABLE5_LLMS = ["baseline", "llama", "falcon", "mistral"]
LLM_ORDER = ["llama", "falcon", "mistral"]
SC_ORDER = ["ball_drop", "shm", "wave"]
METRICS = ["MAE", "MSE", "one_minus_R2", "distance"]

# ---------------------------------------------------------------- valores do artigo
# Tabela 2: prompt -> llm -> sr -> (MAE, MSE, 1-R2, distância)
_T2_EH = {"llama": [(.040, .004, .01, .00), (.020, .002, .01, .00), (.070, .007, .03, .00)],
          "falcon": [(.060, .006, .02, .00), (.040, .004, .01, .00), (.090, .010, .04, .00)],
          "mistral": [(.030, .003, .01, .00), (.010, .001, .00, .00), (.060, .006, .02, .00)]}
_T2 = {
    "A": {"llama": [(.120, .030, .08, .05), (.100, .020, .06, .02), (.150, .040, .10, .12)],
          "falcon": [(.140, .040, .10, .10), (.120, .030, .08, .07), (.170, .050, .12, .15)],
          "mistral": [(.110, .030, .07, .03), (.090, .020, .05, .01), (.140, .040, .09, .11)]},
    "B": {"llama": [(.110, .028, .07, .04), (.090, .018, .05, .01), (.140, .038, .09, .11)],
          "falcon": [(.130, .038, .09, .08), (.110, .028, .07, .05), (.160, .048, .11, .14)],
          "mistral": [(.100, .028, .06, .02), (.080, .018, .04, .01), (.130, .038, .08, .10)]},
    "C": {"llama": [(.100, .025, .06, .03), (.080, .016, .04, .00), (.130, .035, .08, .09)],
          "falcon": [(.120, .035, .08, .06), (.100, .025, .06, .03), (.150, .045, .10, .13)],
          "mistral": [(.090, .025, .05, .01), (.070, .016, .03, .00), (.120, .035, .07, .08)]},
    "D": {"llama": [(.060, .008, .03, .01), (.040, .005, .02, .00), (.090, .012, .05, .06)],
          "falcon": [(.080, .010, .04, .02), (.060, .007, .03, .01), (.110, .015, .06, .10)],
          "mistral": [(.050, .006, .02, .00), (.030, .004, .01, .00), (.080, .010, .04, .04)]},
    "E": _T2_EH, "F": _T2_EH, "G": _T2_EH, "H": _T2_EH,
}

# Tabelas 3 e 4: (cenário, SR) -> 15 distâncias: features 1–5%, alvo 1–5%, ambos 1–5%
_T3 = {
    ("ball_drop", "deap"): [.17, .22, .26, .29, .30, .29, .37, .40, .42, .44, .46, .52, .55, .56, .57],
    ("ball_drop", "pysr"): [.11, .15, .20, .22, .25, .15, .20, .25, .27, .30, .25, .30, .35, .37, .40],
    ("ball_drop", "gplearn"): [.20, .25, .30, .32, .35, .31, .40, .42, .43, .45, .48, .54, .57, .58, .60],
    ("shm", "deap"): [.12, .17, .22, .24, .25, .22, .27, .30, .33, .35, .29, .34, .37, .39, .40],
    ("shm", "pysr"): [.07, .10, .12, .14, .15, .10, .14, .17, .20, .22, .20, .24, .27, .28, .30],
    ("shm", "gplearn"): [.10, .15, .17, .19, .20, .20, .24, .27, .30, .33, .28, .32, .35, .37, .40],
    ("wave", "deap"): [.10, .15, .19, .20, .22, .25, .30, .32, .34, .35, .35, .40, .42, .44, .45],
    ("wave", "pysr"): [.05, .08, .10, .11, .12, .12, .17, .20, .22, .25, .22, .27, .29, .30, .30],
    ("wave", "gplearn"): [.08, .12, .15, .17, .18, .22, .27, .29, .30, .32, .32, .37, .39, .40, .40],
}
_T4 = {
    ("ball_drop", "deap"): [.20, .26, .31, .35, .37, .36, .45, .49, .52, .54, .55, .62, .66, .68, .70],
    ("ball_drop", "pysr"): [.14, .19, .24, .27, .31, .20, .27, .32, .35, .39, .32, .38, .44, .47, .51],
    ("ball_drop", "gplearn"): [.24, .30, .35, .38, .42, .39, .48, .52, .54, .57, .57, .64, .68, .70, .73],
    ("shm", "deap"): [.15, .21, .27, .30, .32, .28, .34, .38, .41, .44, .36, .42, .46, .49, .52],
    ("shm", "pysr"): [.09, .13, .16, .19, .21, .14, .19, .23, .27, .30, .25, .30, .34, .37, .40],
    ("shm", "gplearn"): [.13, .18, .22, .24, .27, .25, .30, .35, .39, .43, .34, .40, .45, .49, .53],
    ("wave", "deap"): [.13, .19, .24, .26, .28, .31, .37, .40, .43, .46, .42, .48, .52, .55, .58],
    ("wave", "pysr"): [.07, .11, .14, .16, .18, .16, .22, .26, .29, .32, .28, .34, .38, .40, .43],
    ("wave", "gplearn"): [.11, .16, .20, .22, .25, .27, .33, .37, .40, .44, .38, .44, .48, .51, .54],
}

# Tabela 5 (suplementar): cenário -> llm -> sr -> (MAE, MSE, 1-R2, distância)
_T5 = {
    "ball_drop": {
        "baseline": {"deap": (.170, .070, .18, .25), "pysr": (.150, .120, .16, .19), "gplearn": (.180, .080, .20, .10)},
        "llama": {"deap": (.120, .030, .08, .05), "pysr": (.100, .020, .06, .02), "gplearn": (.150, .040, .10, .12)},
        "falcon": {"deap": (.140, .040, .10, .10), "pysr": (.120, .030, .08, .07), "gplearn": (.170, .050, .12, .15)},
        "mistral": {"deap": (.110, .030, .07, .03), "pysr": (.090, .020, .05, .01), "gplearn": (.140, .040, .09, .11)},
    },
    "shm": {
        "baseline": {"deap": (.180, .050, .12, .12), "pysr": (.170, .050, .11, .13), "gplearn": (.180, .060, .11, .18)},
        "llama": {"deap": (.080, .010, .05, .02), "pysr": (.070, .010, .04, .01), "gplearn": (.080, .010, .05, .02)},
        "falcon": {"deap": (.100, .020, .07, .05), "pysr": (.090, .010, .06, .03), "gplearn": (.120, .030, .09, .12)},
        "mistral": {"deap": (.070, .010, .04, .01), "pysr": (.060, .010, .03, .00), "gplearn": (.090, .020, .06, .07)},
    },
    "wave": {
        "baseline": {"deap": (.140, .090, .13, .20), "pysr": (.150, .090, .12, .21), "gplearn": (.170, .070, .15, .24)},
        "llama": {"deap": (.050, .005, .03, .00), "pysr": (.040, .004, .02, .00), "gplearn": (.070, .010, .05, .05)},
        "falcon": {"deap": (.070, .010, .05, .02), "pysr": (.060, .008, .04, .01), "gplearn": (.090, .020, .07, .10)},
        "mistral": {"deap": (.040, .004, .02, .00), "pysr": (.030, .003, .01, .00), "gplearn": (.060, .009, .04, .04)},
    },
}
# Tabela 6 (suplementar): (cenário, sr, llm) -> (Total SR s, Total LLM s, razão, SR/chamada,
# LLM/chamada, SR/iteração, LLM/iteração)
_T6 = {
    ("ball_drop", "deap", "mistral"): (7.8, 55.1, 7.1, 0.3, 2.3, 2.6, 18.4),
    ("ball_drop", "deap", "llama"): (7.3, 141.3, 19.4, 0.3, 5.9, 2.4, 47.1),
    ("ball_drop", "deap", "falcon"): (7.4, 49.3, 0.0, 0.3, 0.0, 2.5, 0.1),
    ("ball_drop", "pysr", "mistral"): (169.9, 56.6, 0.3, 7.1, 2.4, 56.6, 18.9),
    ("ball_drop", "pysr", "llama"): (151.4, 58.5, 0.4, 6.3, 2.4, 50.5, 19.5),
    ("ball_drop", "pysr", "falcon"): (154.9, 47.8, 0.0, 6.5, 0.0, 51.6, 0.1),
    ("ball_drop", "gplearn", "mistral"): (1.7, 54.0, 31.0, 0.1, 2.3, 0.6, 18.0),
    ("ball_drop", "gplearn", "llama"): (1.7, 134.5, 76.9, 0.1, 5.6, 0.6, 44.8),
    ("ball_drop", "gplearn", "falcon"): (1.7, 36.1, 0.1, 0.1, 0.0, 0.6, 0.1),
    ("shm", "deap", "mistral"): (7.2, 54.2, 7.5, 0.3, 2.3, 2.4, 18.1),
    ("shm", "deap", "llama"): (7.6, 140.4, 18.5, 0.3, 5.9, 2.5, 46.8),
    ("shm", "deap", "falcon"): (7.2, 38.2, 0.0, 0.3, 0.0, 2.4, 0.1),
    ("shm", "pysr", "mistral"): (183.0, 56.8, 0.3, 7.6, 2.4, 61.0, 18.9),
    ("shm", "pysr", "llama"): (191.0, 67.6, 0.4, 8.0, 2.8, 63.7, 22.5),
    ("shm", "pysr", "falcon"): (186.2, 41.9, 0.0, 7.8, 0.0, 62.1, 0.1),
    ("shm", "gplearn", "mistral"): (1.7, 43.1, 24.9, 0.1, 1.8, 0.6, 14.4),
    ("shm", "gplearn", "llama"): (1.7, 109.2, 62.9, 0.1, 4.5, 0.6, 36.4),
    ("shm", "gplearn", "falcon"): (1.7, 26.0, 0.1, 0.1, 0.0, 0.6, 0.1),
    ("wave", "deap", "mistral"): (7.3, 48.5, 6.6, 0.3, 2.0, 2.4, 16.2),
    ("wave", "deap", "llama"): (7.2, 72.9, 10.1, 0.3, 3.0, 2.4, 24.3),
    ("wave", "deap", "falcon"): (7.3, 35.6, 0.0, 0.3, 0.0, 2.4, 0.1),
    ("wave", "pysr", "mistral"): (173.0, 55.1, 0.3, 7.2, 2.3, 57.7, 18.4),
    ("wave", "pysr", "llama"): (167.8, 60.8, 0.4, 7.0, 2.5, 55.9, 20.3),
    ("wave", "pysr", "falcon"): (169.9, 30.7, 0.0, 7.1, 0.0, 56.6, 0.1),
    ("wave", "gplearn", "mistral"): (1.8, 53.5, 30.4, 0.1, 2.2, 0.6, 17.8),
    ("wave", "gplearn", "llama"): (1.7, 82.5, 47.5, 0.1, 3.4, 0.6, 27.5),
    ("wave", "gplearn", "falcon"): (1.7, 39.5, 0.1, 0.1, 0.0, 0.6, 0.1),
}
T6_COLS = ["total_sr_s", "total_llm_s", "llm_sr_ratio", "sr_per_call_s", "llm_per_call_s",
           "sr_per_iter_s", "llm_per_iter_s"]

NOISE_COLS = [(m, l) for m in ("features", "target", "both") for l in (0.01, 0.02, 0.03, 0.04, 0.05)]


def paper_table2():
    rows = []
    for p, by_llm in _T2.items():
        for llm, vals in by_llm.items():
            for sr, v in zip(SR_ORDER, vals):
                rows.append({"prompt": p, "llm": llm, "sr": sr, "MAE": v[0], "MSE": v[1],
                             "one_minus_R2": v[2], "distance": v[3]})
    return pd.DataFrame(rows).set_index(["prompt", "llm", "sr"])


def paper_noise(biased):
    src = _T4 if biased else _T3
    idx = pd.MultiIndex.from_tuples(list(src), names=["scenario", "sr"])
    cols = pd.MultiIndex.from_tuples(NOISE_COLS, names=["noise_mode", "noise_level"])
    return pd.DataFrame(list(src.values()), index=idx, columns=cols)


# ---------------------------------------------------------------- nossos resultados
def load(results_csv):
    df = pd.read_csv(results_csv)
    df["noise_level"] = df["noise_level"].round(4)
    return df


def _default_noise(df):
    return df[(df.noise_mode == "target") & (df.noise_level == 0.01) & (df.biased == 0)]


def _mean_sd(g, cols):
    m, s, n = g[cols].mean(), g[cols].std(), g[cols[0]].count()
    out = m.copy()
    for c in cols:
        out[c + "_sd"] = s[c]
    out["n"] = n
    return out


def table5(df, cfg):
    """Benchmark: cenário x SR x LLM (+ baseline 'none'), média ± dp das repetições."""
    d = _default_noise(df)
    d = d[((d.llm == "none") & (d.prompt == "-")) |
          (d.llm.isin(cfg["llms"]) & (d.prompt == cfg["bench_prompt"]))]
    t = d.groupby(["scenario", "sr", "llm"]).apply(_mean_sd, METRICS, include_groups=False)
    return t.round(4)


def table2(df, cfg):
    """Prompts A–H: média sobre cenários e repetições, por prompt x LLM x SR, ao lado do artigo."""
    d = _default_noise(df)
    d = d[d.llm.isin(cfg["llms"]) & d.prompt.isin(list("ABCDEFGH"))]
    ours = d.groupby(["prompt", "llm", "sr"])[METRICS].mean()
    paper = paper_table2().add_suffix("_artigo")
    t = ours.join(paper, how="left")
    cols = [c for m in METRICS for c in (m, m + "_artigo")]
    return t[cols].round(4)


def table_noise(df, cfg, biased, llm=None):
    """Tabela 3 (biased=False) ou 4 (biased=True): distância média por cenário x SR x ruído."""
    llm = llm or cfg["noise_llm"]
    d = df[(df.biased == int(biased)) & (df.llm == llm)]
    if llm != "none":
        d = d[d.prompt == cfg["noise_prompt"]]
    t = d.pivot_table(index=["scenario", "sr"], columns=["noise_mode", "noise_level"],
                      values="distance", aggfunc="mean")
    cols = [c for c in NOISE_COLS if c in t.columns]
    return t.reindex(columns=cols).round(3)


def llm_overhead(df):
    """Razão tempo LLM / tempo SR (o artigo afirma < 0.4) e taxa de falha de parse."""
    d = df[df.llm != "none"]
    return d.groupby(["sr", "llm"]).agg(
        razao_LLM_SR_mediana=("llm_ratio", "median"),
        chamadas_LLM_media=("llm_calls", "mean"),
        falha_parse_media=("llm_fail_rate", "mean"),
        execucoes=("key", "count"),
    ).round(3)


def paper_table5():
    rows = []
    for sc, by_llm in _T5.items():
        for llm, by_sr in by_llm.items():
            for sr, v in by_sr.items():
                rows.append({"scenario": sc, "llm": llm, "sr": sr, "MAE": v[0], "MSE": v[1],
                             "one_minus_R2": v[2], "distance": v[3]})
    return pd.DataFrame(rows).set_index(["scenario", "llm", "sr"])


def paper_table6():
    idx = pd.MultiIndex.from_tuples(list(_T6), names=["scenario", "sr", "llm"])
    return pd.DataFrame(list(_T6.values()), index=idx, columns=T6_COLS)


def table5_vs_paper(df, cfg):
    """Tabela 5 no formato do artigo: cenário × LLM (baseline + 3) × SR, nosso × artigo."""
    t = table5(df, cfg).reset_index()
    t["llm"] = t["llm"].replace({"none": "baseline"})
    ours = t.set_index(["scenario", "llm", "sr"])[METRICS]
    out = ours.join(paper_table5().add_suffix("_artigo"), how="outer")
    cols = [c for m in METRICS for c in (m, m + "_artigo")]
    return out[cols].round(4)


def table6(df):
    """Tabela 6 no formato do artigo: tempos de SR e de LLM (médias por execução)."""
    d = df[df.llm != "none"].copy()
    g = d.groupby(["scenario", "sr", "llm"])
    out = pd.DataFrame({
        "total_sr_s": g["sr_seconds"].mean(),
        "total_llm_s": g["llm_seconds"].mean(),
        "llm_sr_ratio": g["llm_ratio"].mean(),
        "sr_per_call_s": (g["sr_seconds"].sum() / g["n_evaluated"].sum().clip(lower=1)),
        "llm_per_call_s": (g["llm_seconds"].sum() / g["llm_calls"].sum().clip(lower=1)),
        "sr_per_iter_s": (g["sr_seconds"].sum() / g["n_generations"].sum().clip(lower=1)),
        "llm_per_iter_s": (g["llm_seconds"].sum() / g["n_generations"].sum().clip(lower=1)),
    })
    return out.join(paper_table6().add_suffix("_artigo"), how="left").round(3)


def save_all(df, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tabs = {
        "tabela5_benchmark.csv": table5_vs_paper(df, cfg),
        "tabela6_tempos.csv": table6(df),
        "tabela2_prompts.csv": table2(df, cfg),
        "tabela3_ruido_media_zero.csv": table_noise(df, cfg, False),
        "tabela4_ruido_enviesado.csv": table_noise(df, cfg, True),
        "tabela3_baseline_sem_llm.csv": table_noise(df, cfg, False, "none"),
        "tabela4_baseline_sem_llm.csv": table_noise(df, cfg, True, "none"),
        "overhead_llm.csv": llm_overhead(df),
    }
    for name, t in tabs.items():
        t.to_csv(os.path.join(out_dir, name))
    return tabs


# ---------------------------------------------------------------- figuras
def plot_prompts(df, cfg, path=None):
    import matplotlib.pyplot as plt

    t = table2(df, cfg).reset_index()
    fig, axes = plt.subplots(1, len(cfg["srs"]), figsize=(5 * len(cfg["srs"]), 4), sharey=True, squeeze=False)
    for ax, sr in zip(axes[0], cfg["srs"]):
        for llm in cfg["llms"]:
            s = t[(t.sr == sr) & (t.llm == llm)].sort_values("prompt")
            line, = ax.plot(s.prompt, s.distance, marker="o", label=f"{llm} (nosso)")
            ax.plot(s.prompt, s.distance_artigo, ls="--", color=line.get_color(), alpha=.6, label=f"{llm} (artigo)")
        ax.set_title(f"SR = {sr}")
        ax.set_xlabel("Prompt")
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("Distância de árvore")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Tabela 2: distância por prompt (média sobre cenários)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_noise(df, cfg, biased, path=None):
    import matplotlib.pyplot as plt

    ours = table_noise(df, cfg, biased)
    base = table_noise(df, cfg, biased, "none") if cfg.get("noise_baseline") else None
    paper = paper_noise(biased)
    modes = ("features", "target", "both")
    fig, axes = plt.subplots(len(cfg["scenarios"]), 3, figsize=(14, 3.2 * len(cfg["scenarios"])),
                             sharey=True, squeeze=False)
    for i, sc in enumerate(cfg["scenarios"]):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            for sr in cfg["srs"]:
                lv = cfg["noise_levels"]
                get = lambda t: [t.loc[(sc, sr), (mode, l)] if (sc, sr) in t.index and (mode, l) in t.columns else np.nan for l in lv]
                line, = ax.plot(lv, get(ours), marker="o", label=f"{sr} + LLM")
                if base is not None:
                    ax.plot(lv, get(base), marker="x", ls=":", color=line.get_color(), label=f"{sr} sem LLM")
                ax.plot(lv, get(paper), ls="--", color=line.get_color(), alpha=.5, label=f"{sr} (artigo)")
            ax.set_title(f"{sc} | ruído em {mode}")
            ax.grid(alpha=.3)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(f"Tabela {4 if biased else 3}: distância vs nível de ruído ({'enviesado' if biased else 'média zero'})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def main(argv=None):
    import argparse

    import runner

    p = argparse.ArgumentParser(description="Gera as tabelas e figuras no formato do artigo.")
    p.add_argument("results_csv")
    p.add_argument("out_dir", nargs="?", default="tabelas")
    p.add_argument("--mode", default="full", choices=list(runner.MODES))
    a = p.parse_args(argv)
    cfg = runner.make_config(a.mode)
    df = load(a.results_csv)
    cfg["srs"] = [s for s in SR_ORDER if s in set(df.sr)]
    cfg["llms"] = [l for l in LLM_ORDER if l in set(df.llm)]
    cfg["scenarios"] = [s for s in SC_ORDER if s in set(df.scenario)]
    save_all(df, cfg, a.out_dir)
    if set(df.prompt) & set("ABCDEFGH"):
        plot_prompts(df, cfg, os.path.join(a.out_dir, "fig_prompts.png"))
    for biased in (False, True):
        if (df.biased == int(biased)).any() and (df.noise_mode != "target").any():
            plot_noise(df, cfg, biased, os.path.join(a.out_dir, f"fig_ruido_{'enviesado' if biased else 'media_zero'}.png"))
    print(f"tabelas e figuras em {a.out_dir}")


if __name__ == "__main__":
    main()
