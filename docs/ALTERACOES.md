# Registro de alterações — branch `paper-faithful`

Referência: código original dos autores, `bilgesi/SR-LLM-Integration@1508a37` (preservado sem alterações em `src/legacy/` e no branch `main`).
Repositório: `RodrigoSantosB/SR-LLM-Integration-paper`, branch `paper-faithful` (padrão).

Este documento descreve **o que mudou, onde e por quê**. Para a correspondência item a item entre o texto do artigo e o código, ver [`CONFORMIDADE.md`](CONFORMIDADE.md).

---

## 1. Visão geral

| Fase | Commits | Objetivo |
|---|---|---|
| Reimplementação | `8a47939`, `73ad278`, `ca07989` | Implementar o método **como descrito no artigo**. No código original, o escore do LLM não entra na busca da regressão simbólica (RS) |
| Controle | `4a0e5ba`, `1e30b5c` | Executar o código original, com patches mínimos, para demonstrar esse defeito |
| Atribuição | `3ae996c` | Indicar a origem e a licença (Apache-2.0) |
| Ajustes pós-piloto | `f6358df` | Corrigir as inconsistências encontradas na execução piloto real (Colab A100, n = 3, 2240 execuções, 18–19/10/2026) |
| Documentação | `6d24239` | Este registro |
| Ajustes pós-`verify` | (seção 6) | Corrigir a consulta ao LLaMA e medir as cópias de exemplos, a partir da execução `verify` de 26–27/10/2026 |

---

## 2. Por que reimplementar: defeitos do código original

| # | Defeito em `1508a37` | Consequência |
|---|---|---|
| C1 | `ga_weight_optimizer.py` otimiza (w₁, w₂, w₃) avaliando a equação **final** de um ajuste já concluído; os pesos nunca chegam ao SR | O LLM e o prompt não alteram a equação encontrada |
| C2 | `expression_tree.py`: o *parser* para no primeiro parêntese; a verdade-base `v(t) = sqrt(2*g*h)` vira o nó `v` | A distância entre árvores não mede proximidade estrutural |
| C3 | DEAP usa só `X[:, 0]` (massa) | DEAP não consegue modelar nenhum dos cenários |
| C4 | 8 gerações; DEAP com cxpb/mutpb 0,6/0,3 | Diverge da Tabela 1 (50 gerações; 0,05/0,01) |
| C5 | `run_data_sensitivity.py` é cópia do script de prompts | Não há experimento de ruído (Tabelas 3 e 4) |
| C6 | Sem linha de base sem LLM, sem repetições | Tabela 5 e n = 25 não reproduzíveis |
| C7 | Erros de execução: f-string inválida em Python < 3.12, `import src.constants`, `TypeError` no script de prompts | O código não roda como publicado |

---

## 3. Reimplementação (commits `8a47939`, `73ad278`, `ca07989`)

### 3.1 Estrutura
- O código original foi **movido sem alterações** para `src/legacy/`.
- Os nomes de arquivo de `src/` foram mantidos; cada módulo foi reescrito.

| Arquivo | Conteúdo |
|---|---|
| `src/constants.py` | Todas as especificações do artigo, com citação de página/seção; desvios marcados com `[DESVIO]` ou `[CONTRADIÇÃO]` |
| `src/data_generation.py` | Gerador da seção *In silico* (500 experimentos × 100 passos, cinco parâmetros em SI) e ruído em entradas/alvo/ambos, de média zero e enviesado |
| `src/expression_tree.py` | Parser via `ast` (notações infixa e prefixa), tamanho `s` e distância entre árvores com as regras do texto |
| `src/loss.py` | Perda composta `L = w₁e + w₂s + w₃c` (`Objective`) e parada antecipada (`EarlyStopper`) |
| `src/llm_models.py` | Prompt idêntico ao original, variantes A–H, Mistral/LLaMA/Falcon em NF4, cache persistente, avaliadores falsos para testes |
| `src/sr_models.py` | DEAP, gplearn e PySR **com a perda composta dentro da busca** |
| `src/runner.py` | Especificação imutável por execução, sementes pareadas, retomada, blocos do artigo (benchmark, prompts, ruído), verificação de integração |
| `src/plot_secondary_results.py` | Tabelas 2–6 no formato do artigo, com os valores publicados ao lado, e figuras |
| `src/cli.py`, `src/run*.py` | Linha de comando e execução dos blocos |
| `docs/CONFORMIDADE.md` | Matriz artigo → código, desvios (D1–D7) e constatações |
| `requirements.txt` | Dependências mínimas; o *freeze* do Colab original virou `requirements-colab-freeze.txt` |
| `tests/` | Testes de conformidade, dados, distância, perda, LLM, SR e *runner* |

### 3.2 Como o LLM entra na busca
- **DEAP:** a perda composta é a aptidão de cada indivíduo, e as expressões novas de cada geração vão ao LLM num único lote.
- **gplearn:** evolução geração a geração (`warm_start`); depois de cada geração, `raw_fitness_`/`fitness_` de toda a população são sobrescritos pela perda composta.
- **PySR:** a evolução interna roda em Julia e não aceita *callback* Python. A cada iteração, a perda composta reordena o Hall of Fame e alimenta a parada antecipada (desvio D3).

### 3.3 Desvios inevitáveis (detalhes em `CONFORMIDADE.md`, seção 5)
D1 pesos fixos (0,5; 0,167; 0,333), linha de base (0,75; 0,25; 0) · D2 constantes do MHS e da onda · D3 PySR via Hall of Fame · D4 gplearn sem parcimônia nativa · D5 textos dos prompts B, C e D · D6 LLM e prompt do experimento de ruído · D7 quantização em 4 bits.

---

## 4. Trilha de controle do código original (commits `4a0e5ba`, `1e30b5c`)

`tools/legacy_track.py` copia `src/legacy/` para uma pasta temporária e aplica **patches mínimos por troca de texto**, cada um conferido com `assert`. Os originais não são editados. Os patches:
- corrigem os erros de execução (C7);
- carregam um LLM por vez;
- permitem filtros, orçamentos e semente via variáveis de ambiente.

O *check* C1 roda o código original com a mesma semente, variando os pesos e o prompt (A × D). Ele mostra que a equação não muda. `1e30b5c` isola os módulos legados que têm o mesmo nome dos módulos novos.

---

## 5. Ajustes pós-piloto (commit `f6358df`)

### 5.1 O que motivou

Constatações da execução piloto real (detalhes em `CONFORMIDADE.md`, seção 7):

| # | Constatação |
|---|---|
| 1 | Distância = 1,0 em 2229 de 2240 execuções |
| 2 | LLaMA e Falcon falharam ao interpretar 86–94% das respostas; com o escore neutro 0,5, as equações coincidiam com as da linha de base |
| 3 | Na escala literal da Eq. 1, o MSE cru domina na queda livre e na onda; no MHS, o termo `s` domina |
| 4 | DEAP parava na 3ª geração e devolvia constantes |
| 5 | Onda com cerca de 10 períodos (≈ 997 passos implícitos); todos os algoritmos devolviam quase-constantes |
| 6 | Tempos de LLM distorcidos por execuções atendidas inteiramente pelo cache |

### 5.2 Correções que valem para os dois perfis

| Ajuste | Arquivos e funções | Antes | Depois |
|---|---|---|---|
| **A1 — Parada da onda** | `constants.WAVE_STOP`; `data_generation.stop_time(…, wave_stop)`, `generate_clean(…, wave_stop)` | Até o envelope cair abaixo de 1%, limitado a 1000 passos | Primeiro estado repetido (um período, 2π/ω = 1 s) ou equilíbrio, o que vier antes: 100 passos, coerente com "around 100 steps". A regra anterior continua disponível como `"equilibrium"` |
| **A2 — Distância** | `expression_tree.canonical`, `canonical_tree`, `canonical_distance`, `structure_match`, `to_sympy`; `data_generation._gt_shm`, `_gt_wave` | Distância sobre o texto literal; GT do MHS com `+ 0.0` e GT da onda com `1.0*L` | Coluna **`distance`**: mesma regra do artigo, aplicada às **formas canônicas (SymPy)**. Variáveis positivas, expoentes reais próximos de frações viram frações (`h**0.5 → sqrt(h)`), constantes inteiras viram inteiros. Coluna **`distance_literal`**: a regra antiga. Coluna **`structure_match`**: 1 se a estrutura coincide com a da GT com constantes livres. GT sem nós neutros |
| **A3 — Consulta ao LLM** | `constants.LLM_CHAT_TEMPLATE`, `LLM_RESPONSE_PREFIX`; `llm_models.format_query`, `HFLLM` | Prompt como texto puro, sem modelo de conversa | Modelo de conversa de cada LLM (`apply_chat_template`; para tokenizadores sem modelo, como o do Falcon-instruct, usa `User: …\nAssistant:`). A resposta é **pré-preenchida com `[`**, para o modelo continuar a lista de escores. O **texto do prompt não muda**. Se o modelo repetir o `[`, a resposta é interpretada sem o prefixo |
| **A4 — Cache e taxa de falha** | `constants.LLM_QUERY_FORMAT`; `llm_models.ScoreCache`, `BaseLLM.score_many`; `runner.run_one` | Chave sem o formato da consulta; cache só com o escore; falhas do cache contadas como sucesso | Chave `formato\|modelo\|sha1(contexto)\|expressão`, para que um cache antigo **não seja reaproveitado**. O cache guarda `(escore, interpretada)`. Colunas `llm_fail_rate` (todas as consultas, inclusive do cache) e `llm_fail_rate_new` (só as novas) |
| **A5 — Tempos** | `plot_secondary_results.table6`, `llm_overhead` | Médias sobre todas as execuções | Tabela 6 e razão LLM/RS só sobre execuções com chamadas novas ao LLM. O *overhead* passa a ter `respostas_cache_media` e `falha_parse_novas_media` |
| **A6 — Separação de resultados** | `cli.config_from`; `runner.ResultStore` | `results_runs/<modo>/` | `results_runs/<modo>_<perfil>/`. O `ResultStore` **recusa um CSV com outras colunas**, para que uma retomada não pule execuções antigas, feitas sem os ajustes |
| **A7 — Modo `verify`** | `runner.MODES["verify"]`; `cli` | — | Verificação curta com LLMs reais: queda livre × 3 SR × {Mistral, LLaMA}, prompts A e D, ruído de 1% e 5%, n = 1 |

### 5.3 Perfil `adjusted` (`--profile adjusted`)

O perfil `literal` (padrão) mantém a especificação do artigo. O `adjusted` aplica variantes documentadas, definidas em `constants.PROFILES`:

| Variante | Valor | Implementação | Motivo |
|---|---|---|---|
| V1 — Escala da perda | `loss_scaling="normalized"`: e = MSE/var(y), s = min(1, nós/30) | `loss.Objective` (já existente) | Termos na mesma escala do termo do LLM (constatação 3) |
| V2 — Operadores extras | `sqrt` protegida nos três SR; `exp` no DEAP | `constants.EXTRA_OPS`; `sr_models.psqrt`; parâmetro `extra_unary` dos três modelos; `"psqrt"` no parser | A GT da queda livre é uma raiz; a Tabela 1 retira `exp` do DEAP |
| V3 — Parada antecipada | `min_generations=10` | `loss.EarlyStopper(min_generations)`; `BaseSR(min_generations)` | DEAP parava na 3ª geração (constatação 4) |
| V4 — Profundidade inicial do DEAP | `deap_init_depth=(1, 3)` | `runner.make_sr` → `DEAPModel(init_depth=…)` | Com 1, o DEAP literal devolvia constantes |

### 5.4 Formato dos resultados
- **Colunas novas** em `resultados_*.csv`: `distance_literal`, `structure_match`, `llm_fail_rate_new`, `profile`.
- **Colunas que mudaram de significado:**
  - `distance`: agora é a distância canônica;
  - `llm_fail_rate`: agora conta todas as consultas, inclusive as do cache.
- **Tabelas novas:** `tabela3_estrutura.csv`, `tabela4_estrutura.csv`.
- **Tabelas com colunas extras:** Tabelas 2 e 5 ganham `distance_literal` e `structure_match`.

---

## 6. Ajustes pós-`verify` (A8)

### 6.1 Diagnóstico da execução `verify`

Colab A100, commit `f6358df`, perfil `literal`. As variáveis de escopo foram preenchidas, o que sobrescreveu o modo: grid completo, n = 1, 423 execuções, ≈ 8 h.

| Observação | Causa | Ação |
|---|---|---|
| Distância 1,0 e `structure_match` 0 em todas as etapas | **Não é defeito da métrica.** As 468 distâncias recalculadas conferem e as formas corretas dão 0. As expressões encontradas não têm a estrutura das verdades-base (`3.91*9.13`, `log(h)/0.077 − 18.22`, `L`, `−log(t)`): sem `sqrt`, parada na ≈ 3ª geração e MSE cru dominante no perfil literal | Executar o perfil `adjusted` |
| LLaMA com 2,7% de respostas interpretadas | O `[` pré-preenchido logo após `[/INST]` foi lido como início de tag: respostas `Inst]`, `Assistant]:` e a linha de formato copiada | Prefixo `"Output: ["`, que repete o formato dos exemplos do prompt |
| Falcon com 78% interpretadas, mas 50,6% delas copiando os escores de um exemplo e 23% em `[0.5, 0.5, 0.5]` | O modelo reproduz os exemplos em vez de avaliar a expressão | Prefixo `"Output: ["` e nova coluna `llm_copy_rate`, para relatar a qualidade das respostas |
| Mistral com 0,1% de falhas (`050`) | Número sem o ponto decimal | Leitura tolerante: `050` → 0,50 |
| 468 execuções no modo `verify` | Listas de escopo explícitas sobrescrevem o modo | Aviso na célula de configuração do notebook; o escopo padrão do `verify` tem 93 execuções |

### 6.2 Alterações

| Arquivo | Alteração |
|---|---|
| `src/constants.py` | `LLM_RESPONSE_PREFIX` passa a ser um dicionário por modelo (`default: "["`, `llama` e `falcon`: `"Output: ["`); `LLM_QUERY_FORMAT = "chat-prefill-v3"` (o cache de `f6358df` não é reaproveitado); `FEWSHOT_SCORES` com os escores dos três exemplos do prompt |
| `src/llm_models.py` | `response_prefix_for(nome)`; `parse_llm_scores` (devolve os três escores) com leitura tolerante de zero à esquerda; `is_example_copy`; o cache grava e lê o indicador `copy`; contadores `n_copy` e `n_cached_copy`; `HFLLM(response_prefix=None)` usa o prefixo do modelo |
| `src/runner.py` | Coluna `llm_copy_rate` (novas e do cache) |
| `src/plot_secondary_results.py` | `overhead_llm.csv` com `copia_exemplo_media` |
| `tests/test_ajustes_verify.py` | 10 testes: prefixo por modelo e consulta resultante, versão do formato, leitura tolerante, as respostas inválidas do LLaMA continuam falhas, detecção de cópias (novas e do cache), coluna no `run_one` |
| `colab/build_notebook.py` (fora do repositório) | Aviso quando `LLMS`, `SRS`, `SCENARIOS` ou `PROMPTS` sobrescrevem o escopo de `smoke`, `pilot` ou `verify`, com o número de execuções previsto |

O texto do prompt não muda. A coluna nova altera o cabeçalho dos CSVs: resultados de `f6358df` não são retomados (use uma pasta nova).

### 6.3 Validação
- `pytest -m "not slow"`: 113 testes passando.
- `run.py --mode smoke --srs deap gplearn`, perfis `literal` e `adjusted`: sem erros.
- Reprocessamento do `llm_cache.jsonl` da execução `verify` com a nova leitura: Mistral 99,9% → 100%; cópias de exemplo: Falcon 50,6%, LLaMA 1,0%, Mistral 0%.
- **Pendente:** o efeito do prefixo `"Output: ["` sobre o LLaMA só pode ser confirmado com o modelo real (modo `verify` no Colab).

---

## 7. Compatibilidade

- **Resultados antigos:** CSVs de versões anteriores não são retomados (erro explícito). Use uma pasta nova; no notebook, `real_literal` ou `real_adjusted`, em vez de `real`.
- **Cache antigo:** pode continuar no disco. As entradas antigas são ignoradas, porque a chave mudou.
- **Leitura de CSVs antigos:** `plot_secondary_results.load` preenche com `NaN` as colunas que faltam, o que permite gerar tabelas da rodada anterior.
- **Reprodução da versão anterior:** commit `3ae996c`, ou `WAVE_STOP="equilibrium"` mais a coluna `distance_literal`.

---

## 8. Uso

```bash
python src/run.py --mode smoke                       # CPU, LLM falso
python src/run.py --mode verify --profile literal    # verificação com LLMs reais
python src/run.py --mode verify --profile adjusted
python src/run.py --mode full   --profile literal    # protocolo do artigo, n = 25
```

**No Colab:** notebook gerado por `colab/build_notebook.py` (fora deste repositório). Variáveis `MODE` (inclui `"verify"`) e `PROFILE`. Para fixar a versão, use `REPO_REF = "f6358df"`.

---

## 9. Validação

| Verificação | Resultado |
|---|---|
| `pytest` | 105 testes passando (85 anteriores + 20 em `tests/test_ajustes.py`) |
| `run.py --mode smoke`, perfis `literal` e `adjusted` | Três blocos completos, sem erros, tabelas e figuras geradas |
| Notebook regerado, executado localmente em `smoke` contra `f6358df` | Todas as células sem erro; 75 execuções |

**Pendente:** a eficácia de A3 (modelo de conversa e resposta pré-preenchida) sobre o LLaMA e o Falcon **não foi verificada com os modelos reais**, porque exige GPU e as licenças dos modelos. Ela deve ser confirmada com o modo `verify` no Colab: a taxa `falha_parse_media` do LLaMA em `overhead_llm.csv` deve ficar bem abaixo dos 86–94% do piloto.
