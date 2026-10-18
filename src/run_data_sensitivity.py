#!/usr/bin/env python3
"""Noise robustness experiment (p.7) e análise com ruído enviesado (p.8): ruído gaussiano de
1% a 5% do desvio-padrão aplicado às entradas, ao alvo ou a ambos, com média zero (Tabela 3)
e com média igual a 1% da média de cada coluna (Tabela 4).

Substitui o script original de mesmo nome, que era uma cópia do de prompts.
Exemplo: python run_data_sensitivity.py --mode smoke
"""
from cli import config_from, parser, preload_pysr

preload_pysr()

import os  # noqa: E402

import plot_secondary_results as T  # noqa: E402
import runner  # noqa: E402


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--only", choices=["zero_mean", "biased"], help="roda só uma das tabelas")
    args = p.parse_args(argv)
    cfg = config_from(args)
    csv_path = os.path.join(cfg["run_dir"], "resultados_ruido.csv")
    out = os.path.join(cfg["run_dir"], "tabelas")
    os.makedirs(out, exist_ok=True)
    for biased in (False, True):
        if args.only == ("zero_mean" if biased else "biased"):
            continue
        runner.run_specs(runner.specs_noise(cfg, biased), cfg, csv_path)
    df = T.load(csv_path)
    for biased, n in ((False, 3), (True, 4)):
        if not (df.biased == int(biased)).any():
            continue
        T.table_noise(df, cfg, biased).to_csv(os.path.join(out, f"tabela{n}_ruido.csv"))
        if cfg["noise_baseline"]:
            T.table_noise(df, cfg, biased, "none").to_csv(os.path.join(out, f"tabela{n}_ruido_sem_llm.csv"))
        T.plot_noise(df, cfg, biased, os.path.join(out, f"fig_tabela{n}_ruido.png"))
    print(f"[ruído] resultados: {csv_path}; tabelas em {out}")


if __name__ == "__main__":
    main()
