"""Argumentos comuns aos scripts de experimento."""
import argparse
import os


def preload_pysr():
    """O PySR (juliacall) precisa ser importado antes do torch para evitar falhas conhecidas."""
    try:
        import pysr  # noqa: F401
    except ImportError:
        pass


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", default="full", choices=["smoke", "pilot", "verify", "full"],
                   help="smoke: CPU com LLM falso; pilot: amostra com LLM real; verify: verificação "
                        "curta das correções com LLMs reais; full: protocolo do artigo")
    p.add_argument("--n-reps", type=int, help="repetições (artigo: 25)")
    p.add_argument("--scenarios", nargs="+", help="ball_drop shm wave")
    p.add_argument("--srs", nargs="+", help="deap pysr gplearn")
    p.add_argument("--llms", nargs="+", help="mistral llama falcon")
    p.add_argument("--prompts", nargs="+", help="A B C D E F G H")
    p.add_argument("--out", default="results_runs", help="pasta de resultados")
    p.add_argument("--fake-llm", action="store_true", help="usa o avaliador constante (testes)")
    p.add_argument("--max-fit-samples", type=int, help="subamostra para o ajuste (desvio; padrão: todas)")
    p.add_argument("--loss-scaling", choices=["paper", "normalized"])
    p.add_argument("--profile", default="literal", choices=["literal", "adjusted"],
                   help="literal: especificação do artigo; adjusted: variantes pós-piloto (CONFORMIDADE §7)")
    return p


def config_from(args):
    import runner

    cfg = runner.make_config(
        args.mode, profile=args.profile, n_reps=args.n_reps, scenarios=args.scenarios, srs=args.srs, llms=args.llms,
        prompts=args.prompts, max_fit_samples=args.max_fit_samples, loss_scaling=args.loss_scaling,
        fake_llm=True if args.fake_llm else None,
    )
    # uma pasta por modo e perfil: resultados de versões/perfis diferentes nunca se misturam
    run_dir = os.path.join(args.out, f"{args.mode}_{args.profile}")
    os.makedirs(run_dir, exist_ok=True)
    cfg["cache_path"] = os.path.join(run_dir, "llm_cache.jsonl")
    cfg["run_dir"] = run_dir
    return cfg
