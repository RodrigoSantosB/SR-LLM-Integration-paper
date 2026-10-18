#!/usr/bin/env python3
"""LLM prompt knowledge integration experiment (p.6–7): prompts A–H para cada combinação
LLM × SR × cenário, com n repetições (Tabela 2).

Exemplo: python run_prompt_sensitivity.py --mode smoke
"""
from cli import config_from, parser, preload_pysr

preload_pysr()

import os  # noqa: E402

import plot_secondary_results as T  # noqa: E402
import runner  # noqa: E402


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    cfg = config_from(args)
    csv_path = os.path.join(cfg["run_dir"], "resultados_prompts.csv")
    runner.run_specs(runner.specs_prompts(cfg), cfg, csv_path)
    df = T.load(csv_path)
    out = os.path.join(cfg["run_dir"], "tabelas")
    os.makedirs(out, exist_ok=True)
    T.table2(df, cfg).to_csv(os.path.join(out, "tabela2_prompts.csv"))
    T.plot_prompts(df, cfg, os.path.join(out, "fig_prompts.png"))
    print(f"[prompts] resultados: {csv_path}; tabelas em {out}")


if __name__ == "__main__":
    main()
