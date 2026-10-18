#!/usr/bin/env python3
"""Benchmarking (seção "Experiment types", p.6): as 27 configurações cenário × SR × LLM,
comparadas à linha de base de cada SR sem LLM, com n repetições (Tabelas 5 e 6).

Exemplo: python run_main_claim.py --mode smoke
"""
from cli import config_from, parser, preload_pysr

preload_pysr()

import os  # noqa: E402

import plot_secondary_results as T  # noqa: E402
import runner  # noqa: E402


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    cfg = config_from(args)
    csv_path = os.path.join(cfg["run_dir"], "resultados_benchmark.csv")
    runner.run_specs(runner.specs_benchmark(cfg), cfg, csv_path)
    df = T.load(csv_path)
    out = os.path.join(cfg["run_dir"], "tabelas")
    os.makedirs(out, exist_ok=True)
    T.table5_vs_paper(df, cfg).to_csv(os.path.join(out, "tabela5_benchmark.csv"))
    T.table6(df).to_csv(os.path.join(out, "tabela6_tempos.csv"))
    print(f"[benchmark] resultados: {csv_path}; tabelas em {out}")


if __name__ == "__main__":
    main()
