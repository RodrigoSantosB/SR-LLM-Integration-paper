# SR-LLM-Integration — implementação fiel ao artigo (branch `paper-faithful`)

> Derivado de [bilgesi/SR-LLM-Integration](https://github.com/bilgesi/SR-LLM-Integration) (Apache-2.0), commit `1508a37`. Branch `main`: código original dos autores; branch `paper-faithful`: esta reimplementação.

Taskin, Xie & Lazebnik, *Knowledge integration for physics-informed symbolic regression using pre-trained large language models*, Scientific Reports 16:1614 (2026).

Este branch reimplementa o pipeline conforme o **texto do artigo**:
- o escore do LLM entra na função de perda da regressão simbólica **durante a busca**;
- hiperparâmetros da Tabela 1;
- gerador de dados da seção *In silico*;
- linha de base sem LLM, n = 25 repetições, ruído (Tabelas 3 e 4) e a distância entre árvores descrita no texto.

O código original está preservado em `src/legacy/`. A correspondência item a item entre artigo e código está em [`docs/CONFORMIDADE.md`](docs/CONFORMIDADE.md).

## Instalação
```bash
pip install -r requirements.txt     # torch vem do ambiente (CUDA); o PySR instala o Julia no 1º uso
huggingface-cli login               # Llama-2 exige licença aceita no Hugging Face
```

## Execução
```bash
python src/run.py --mode smoke      # validação em CPU com LLM falso (minutos)
python src/run.py --mode pilot      # Mistral × DEAP × queda livre, n = 3: mede o custo
python src/run.py --mode full       # protocolo completo do artigo (n = 25)
```
Também é possível executar cada bloco separadamente:
- `src/run_main_claim.py`: Tabelas 5 e 6;
- `src/run_prompt_sensitivity.py`: Tabela 2;
- `src/run_data_sensitivity.py`: Tabelas 3 e 4.

Opções:
- `--n-reps`, `--scenarios`, `--srs`, `--llms`, `--prompts`, `--out`: restringem o escopo da execução;
- `--loss-scaling {paper,normalized}`: escala dos termos da perda;
- `--fake-llm`: troca o LLM por um avaliador constante, para testes.

Os resultados ficam em `results_runs/<modo>/resultados_*.csv`, com uma linha por execução, e a execução pode ser retomada. As tabelas no formato do artigo, com os valores publicados ao lado, ficam em `results_runs/<modo>/tabelas/`.

## Estrutura de `src/`
| Arquivo | Conteúdo |
|---|---|
| `constants.py` | todas as especificações do artigo, com citação |
| `data_generation.py` | cenários, dados e ruído |
| `expression_tree.py` | árvore, tamanho (s) e distância |
| `loss.py` | perda composta (Eq. 1) e early stopping |
| `llm_models.py` | prompt, variantes A–H, Mistral/Llama/Falcon, cache |
| `sr_models.py` | DEAP, gplearn e PySR com o LLM na busca |
| `runner.py` | especificações, sementes pareadas, retomada |
| `plot_secondary_results.py` | Tabelas 2–6 e figuras |

## Testes
```bash
pytest            # inclui testes lentos do PySR (marcador "slow")
```

## Desvios
Os desvios inevitáveis (pesos não informados, o LLM no Hall of Fame do PySR, os textos dos prompts B e C, etc.) e as contradições do artigo estão em `docs/CONFORMIDADE.md`, seções 5 e 6.

## How to cite

If you use this work, please cite:

```bibtex
@article{Taskin2026,
  author  = {Taskin, Bilge and Xie, Wenxiong and Lazebnik, Teddy},
  title   = {Knowledge integration for physics-informed symbolic regression using pre-trained large language models},
  journal = {Scientific Reports},
  year    = {2026},
  volume  = {16},
  number  = {1},
  pages   = {1614},
  doi     = {10.1038/s41598-026-35327-6},
  url     = {https://doi.org/10.1038/s41598-026-35327-6}
}
```


