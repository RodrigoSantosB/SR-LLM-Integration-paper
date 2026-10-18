#!/usr/bin/env python3
"""Executa todos os experimentos do artigo em sequência, com os mesmos argumentos:
benchmark (Tabelas 5 e 6), prompts A–H (Tabela 2) e ruído (Tabelas 3 e 4). Em seguida
consolida as tabelas e figuras de todos os blocos.

Exemplos:
  python src/run.py --mode smoke            # validação em CPU, LLM falso (minutos)
  python src/run.py --mode pilot            # amostra com LLM real, para estimar custo
  python src/run.py --mode full             # protocolo do artigo (n = 25)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import parser, preload_pysr  # noqa: E402

preload_pysr()

import pandas as pd  # noqa: E402

import plot_secondary_results as T  # noqa: E402
import run_data_sensitivity  # noqa: E402
import run_main_claim  # noqa: E402
import run_prompt_sensitivity  # noqa: E402
from cli import config_from  # noqa: E402

BLOCKS = {"benchmark": run_main_claim, "prompts": run_prompt_sensitivity, "ruido": run_data_sensitivity}


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--blocks", nargs="+", choices=list(BLOCKS), default=list(BLOCKS))
    args, rest = p.parse_known_args(argv)
    argv = list(argv if argv is not None else sys.argv[1:])
    argv = [a for a in argv if a not in ("--blocks", *BLOCKS)]
    for name in args.blocks:
        print(f"\n=== {name} ===")
        BLOCKS[name].main(argv)
    cfg = config_from(args)
    frames = [T.load(os.path.join(cfg["run_dir"], f)) for f in
              ("resultados_benchmark.csv", "resultados_prompts.csv", "resultados_ruido.csv")
              if os.path.exists(os.path.join(cfg["run_dir"], f))]
    if frames:
        df = pd.concat(frames, ignore_index=True).drop_duplicates("key")
        T.save_all(df, cfg, os.path.join(cfg["run_dir"], "tabelas"))
        print(f"\nTabelas consolidadas em {os.path.join(cfg['run_dir'], 'tabelas')}")


if __name__ == "__main__":
    main()
