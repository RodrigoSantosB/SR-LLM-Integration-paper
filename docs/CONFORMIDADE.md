# Conformidade: artigo × implementação

Artigo: Taskin, Xie & Lazebnik, *Scientific Reports* 16:1614 (2026).
Branch `paper-faithful` sobre `bilgesi/SR-LLM-Integration@1508a37`.

O código original foi preservado em `src/legacy/`. As especificações numéricas estão centralizadas em `src/constants.py`, e o teste `tests/test_conformidade.py` verifica que elas coincidem com o texto e a Tabela 1.

**Legenda:**
- **F**: fiel ao texto.
- **I**: interpretação de um texto ambíguo.
- **D**: desvio documentado. O artigo é omisso ou a especificação é impraticável.
- **C**: contradição do próprio artigo. Regra adotada: a Tabela 1 tem precedência.

## 1. Método: função de perda e avaliador LLM

| # | Especificação no artigo | Implementação | Status |
|---|---|---|---|
| M1 | "L = w1 e + w2 s + w3 c", w ∈ [0,1], soma 1 (Eq. 1, p.4) | `loss.Objective`, `loss.check_weights` | F |
| M2 | "e is the mean square error" | `Objective._terms` com `LOSS_SCALING="paper"` (MSE cru); `"normalized"` é opcional | F |
| M3 | "s is the size … in terms of nodes in the expression tree" | `expression_tree.size` sobre a árvore produzida pelo SR, sem simplificação | F |
| M4 | "c := 1 − (c1+c2+c3)/3" (p.5) | `llm_models.parse_llm_response` (média) + `Objective` (1 − média) | F |
| M5 | O LLM integra o processo de busca: "we aim to integrate LLM into the SR's search process itself" (p.4) | DEAP e gplearn: loss composta na aptidão de cada geração. PySR: ver D3. | F (DEAP, gplearn) |
| M6 | Prompt com "### ROLE", "### METRICS", exemplos, lista estrita, "ONLY" (p.5) | `llm_models.PROMPT_TEMPLATE`, idêntico caractere a caractere ao código original (o texto do artigo omite o prompt) | F (ao código) |
| M7 | "temperature to zero to guarantee deterministic outputs" (p.5–6) | `do_sample=False` (decodificação gulosa) | F |
| M8 | Mistral 7B, Llama 2 7B e Falcon 7B, locais (p.6) | `constants.LLM_MODELS`; NF4 4 bits, como no código original | F |
| M9 | Pesos w1, w2, w3 | não informados → `DEFAULT_WEIGHTS = (0,5; 0,167; 0,333)`, a média da Dirichlet(3,2,1) do código original | D1 |
| M10 | Linha de base "in which no LLM model is used" (p.6) | `BASELINE_WEIGHTS = (0,75; 0,25; 0)`; o LLM não é consultado | F/I |

## 2. Dados (seção "In silico physical experiments", p.6)

| # | Especificação | Implementação | Status |
|---|---|---|---|
| A1 | "N = 500 experiments" | `N_EXPERIMENTS` | F |
| A2 | Cinco parâmetros uniformes: massa 0,1–10 kg; comprimento característico 0,01–0,5 m; altura/deslocamento inicial 1–100 m; arrasto/amortecimento 0–1 kg/s; tempo | `data_generation.generate_clean`; matriz X = (m, L, h, b, t) em unidades SI, para todos os cenários | F |
| A3 | "time, sampled randomly between zero and … √(2h/g)" | Tempos uniformes ordenados em [0, T_parada]; na queda livre, T = √(2h/g) | F (queda); I (MHS, onda) |
| A4 | "run until reaching equilibrium or 1000 steps" / "first time a state is observed twice (for period dynamics)" | Queda: impacto. MHS: um período (primeiro estado repetido). Onda: um período 2π/ω ou o equilíbrio, o que vier antes (`WAVE_STOP="period"`, ajuste A1 da seção 8); a regra anterior (envelope < 1%, até 1000 passos) continua disponível como `"equilibrium"` | I. Com `"period"`, 100 passos por experimento, coerente com os "≈100" do texto (a regra anterior implicava ≈ 997) |
| A5 | "around 100 steps … around 50,000 samples" | 100 amostras por experimento → 50 000 linhas | I |
| A6 | Equações 2–4 | Queda: `v = sqrt(2*9.81*h)`. MHS: A = L, com k = 1 e φ = 0 (`x = L*cos(sqrt(1.0/m)*t)`). Onda: E0 = h, α = b, x = L, k = 1, ω = 2π (`E = h*exp(-b*t/2)*cos(L - 6.28…*t)`). Fase nula e k = 1 não entram como nós neutros na verdade-base | D2: constantes não definidas no artigo, com os valores do código original |
| A7 | Ruído "1% of each signal's variability", SNR ≈ 40 dB | σ = 0,01·std(coluna); SNR medido = 40,02 dB | F |
| A8 | "retained in their original SI units … relying on the regression models to manage any required scaling" | Sem padronização (o original padronizava X no PySR) | F |
| A9 | Robustez: 1–5% "separately to the input features and to the target" + colunas "Both" (Tabela 3) | `add_noise(mode ∈ {features, target, both})`, aplicado sobre dados limpos | F |
| A10 | Ruído enviesado: "µ = 1% of the mean value of each feature" (p.8, Tabela 4) | `add_noise(biased=True)`: µ = 0,01·média(coluna) e o mesmo σ | F (entradas); I (alvo) |

## 3. Algoritmos de RS (seção "Symbolic regression models", p.6; Tabela 1, p.7)

| # | Especificação | Implementação | Status |
|---|---|---|---|
| S1 | População 100 e 50 gerações para os três algoritmos | `POPULATION_SIZE`, `GENERATIONS` | F |
| S2 | DEAP: cxpb 0,05, mutpb 0,01, torneio 3, altura máxima 8 | `constants.DEAP`, `sr_models.DEAPModel` | F |
| S3 | DEAP: "Initial tree depth 1" | `genHalfAndHalf(min=max=1)` | F |
| S4 | DEAP: operadores +, −, *, / protegida; neg, log, sin, cos | Tabela 1 literal, sem exp; `DEAP_EXTRA_UNARY=("pexp",)` reativa exp | C1: o texto (p.6) lista exp para os três algoritmos |
| S5 | DEAP usa as variáveis do problema | Todas as colunas (o original usava só `X[:,0]`) | F |
| S6 | PySR: + − * / ^ e neg, exp, log, sin, cos; "Huber loss"; "complexity penalty that softly discourages extra operators" | `PySRModel`: HuberLoss, `parsimony=0.0032` (padrão do PySR; valor não informado) | F/I |
| S7 | PySR: "50 generations (iterations)" | 50 iterações do PySR (uma chamada a `fit` com `warm_start` = uma iteração, com o número padrão de ciclos, como no código original) | I |
| S8 | gplearn: add, sub, mul, div; neg, exp, log, sin, cos; "substantial complexity penalty" | Exp protegida; `parsimony_coefficient=0`, porque o termo s da loss composta exerce a penalização | D4 |
| S9 | Early stopping: "best composite loss improved by less than 0.1% over three generations" | `loss.EarlyStopper`: compara a melhor loss atual com a de 3 gerações antes | F |

## 4. Protocolo e métricas (seções "Experiments" e "Experiment types", p.5–7)

| # | Especificação | Implementação | Status |
|---|---|---|---|
| P1 | "n = 25 repetitions" | `N_REPETITIONS = 25` no modo `full`; o modo `pilot` usa n = 3 | F |
| P2 | Benchmark com 27 configurações vs. linha de base | `runner.specs_benchmark`; Tabela 5 em `plot_secondary_results.table5_vs_paper` | F |
| P3 | Prompts A–H | `llm_models.context_for`, `runner.specs_prompts`; B e C reescritos | D5: material suplementar indisponível |
| P4 | Robustez a ruído (Tabelas 3 e 4) | `runner.specs_noise`; LLM e prompt das tabelas não informados → Mistral, prompt A, mais a linha de base | D6 |
| P5 | MAE, MSE, 1−R² e distância de árvore | `runner.run_one` (sobre todo o conjunto; o artigo não especifica partição) | F/I |
| P6 | Distância: folha igual = 0; variáveis diferentes = 1; números α\|Δ\| limitado a 1; folha × nó = 1; operador diferente = 1; comutativos pelo menor pareamento; não comutativos esquerda-esquerda/direita-direita; soma normalizada pelo nº de comparações | `expression_tree.tree_distance` com as regras do texto e α = 0,1. Coluna `distance`: aplicada às formas canônicas (SymPy) das duas expressões; `distance_literal`: aplicada ao texto produzido pelo SR, sem simplificação; `structure_match`: estrutura igual à da verdade-base com constantes livres (ajuste A2 da seção 8) | F nas regras; I na forma canônica (o artigo não diz como as expressões são escritas antes da comparação) |
| P7 | "zero distance signifies structural equivalence" (p.7) | 0 = equivalente | C2: a p.8 chama 1,00 de "perfect reconstruction"; as tabelas usam 0 como ótimo |
| P8 | Razão de tempo LLM/RS "typically remains below 0.4" (p.8) | Tempos de RS e LLM por execução; Tabela 6 com os valores publicados ao lado | F |

## 5. Desvios inevitáveis

| # | Desvio | Motivo |
|---|---|---|
| D1 | Pesos fixos (0,5; 0,167; 0,333) | Não informados no artigo. O GA de pesos do código original nunca repassava os pesos ao SR. |
| D2 | Constantes k, φ, k_w e ω e o mapeamento das variáveis nas Eqs. 3–4 | O artigo define apenas cinco parâmetros genéricos. |
| D3 | PySR: o LLM reordena o Hall of Fame a cada iteração, sem entrar na seleção interna | A evolução do PySR roda em Julia e não aceita *callback* Python por candidato. |
| D4 | gplearn: parcimônia nativa desligada; a aptidão é sobrescrita pela loss composta a cada geração | O termo s substitui a penalização; é a forma de inserir o LLM na seleção. |
| D5 | Textos dos prompts B, C e D | O material suplementar não está disponível; D mantém a redação do código original. |
| D6 | LLM e prompt das Tabelas 3 e 4 | Não informados; ambos são configuráveis. |
| D7 | Quantização em 4 bits | Herdada do código original; o artigo não a menciona. |

## 6. Constatações da validação (especificação literal)

Execuções com dados no tamanho do artigo (500 × 100), uma repetição e avaliador constante:

1. **Early stopping encerra a busca cedo.** O DEAP para na 3ª geração e o PySR na 4ª iteração; o gplearn segue por cerca de 40 gerações. A Tabela 6 do artigo mostra o mesmo padrão: a razão entre tempo total e tempo por iteração indica cerca de 3 iterações para os três algoritmos.
2. **DEAP com a Tabela 1 literal quase não evolui.** Com profundidade inicial 1, cxpb 0,05 e mutpb 0,01, surgem 3 a 6 indivíduos novos por geração. Na queda livre o melhor indivíduo foi uma constante (1 − R² = 1,39), longe do MAE de 0,17 reportado para o DEAP.
3. **Com a escala literal da Eq. 1, o termo do LLM é pequeno.** O MSE cru é da ordem de 10²–10³ na queda livre, enquanto c ∈ [0, 1] e w3 ≈ 0,33. Por isso a influência do LLM só aparece entre candidatos de erro semelhante: no teste de integração, o prompt D mudou a expressão escolhida pelo gplearn, mas não a do DEAP. Para termos em escala comparável, use `--loss-scaling normalized` (desvio opcional).
4. **Custo.** Na CPU, cada execução do gplearn leva cerca de 100 s e faz cerca de 2 900 consultas ao LLM, das quais muitas são atendidas pelo cache. A estimativa do protocolo completo deve vir do modo `pilot` (`runner.estimate_remaining`).

## 7. Constatações da execução piloto real (18–19/10/2026)

Colab A100, grid completo (3 cenários × 3 SR × {linha de base, Mistral, LLaMA, Falcon}, prompts A–H, ruído de média zero e enviesado), n = 3, perfil literal, commit `3ae996c`; 2240 execuções, ≈ 17,5 h. Resultados e relatório: `relatorio_reproducao_SR-LLM.md` (repositório do grupo).

1. **Distância saturada:** 1,0 em 2229 das 2240 execuções (operador de topo diferente vale 1; `sqrt` ausente dos operadores; formas equivalentes como `(19.62*h)**0.5` recebiam 1; nó neutro `+ 0.0` na verdade-base do MHS).
2. **Interpretação das respostas:** LLaMA e Falcon falharam em 86–94% das consultas novas (LLaMA: sequências vazias; Falcon: texto livre). Com o escore neutro 0,5 em todas as candidatas, L = ⅔·L_base + ⅙ e a seleção coincide com a da linha de base. A taxa de falha era subestimada (o cache contava como sucesso) e os escores neutros ficavam no cache.
3. **Escala da perda:** na queda livre e na onda o MSE cru (10²–10³) domina; no MHS (var(y) = 0,04) o termo s domina e o escore do Mistral passou a selecionar expressões piores.
4. **Busca interrompida:** DEAP para na 3ª geração em 758 de 792 execuções e devolve constantes; PySR para em 4–10 iterações.
5. **Onda:** com a regra anterior de parada (≈ 10 períodos), var(y) ≈ 466 e todos os algoritmos devolveram expressões quase constantes.
6. **Tempos:** médias por execução distorcidas pelas execuções atendidas inteiramente pelo cache.

## 8. Ajustes pós-piloto

Correções que valem para os dois perfis (defeitos):

| # | Ajuste | Implementação |
|---|---|---|
| A1 | Parada da onda: primeiro estado repetido (um período) ou equilíbrio | `constants.WAVE_STOP="period"`, `data_generation.stop_time` |
| A2 | Distância sobre formas canônicas + estrutura com constantes livres; verdade-base sem nós neutros | `expression_tree.canonical_distance`, `structure_match`, `_gt_shm`, `_gt_wave`; colunas `distance`, `distance_literal`, `structure_match` |
| A3 | Consulta com o modelo de conversa de cada LLM e resposta pré-preenchida com "[" (texto do prompt inalterado) | `constants.LLM_CHAT_TEMPLATE`, `LLM_RESPONSE_PREFIX`; `llm_models.format_query` (Falcon-instruct, sem modelo de conversa no tokenizador: formato "User: …\nAssistant:") |
| A4 | Cache versionado pelo formato da consulta e com o indicador de resposta interpretada; falhas do cache contam nas estatísticas | `LLM_QUERY_FORMAT` na chave; `ScoreCache.get → (score, ok)`; colunas `llm_fail_rate` (todas as consultas) e `llm_fail_rate_new` (só as novas) |
| A5 | Tempos de LLM só das execuções com chamadas novas | `plot_secondary_results.table6`, `llm_overhead` |
| A6 | Resultados de versões ou perfis diferentes nunca se misturam | pasta `results_runs/<modo>_<perfil>/`; `ResultStore` recusa um CSV com outras colunas |
| A7 | Verificação curta das correções com LLMs reais | modo `verify`: queda livre × 3 SR × {Mistral, LLaMA}, prompts A e D, ruído 1% e 5%, n = 1 |

Variantes do perfil `adjusted` (`--profile adjusted`; o perfil `literal` mantém a especificação do artigo):

| # | Variante | Valor | Motivo |
|---|---|---|---|
| V1 | `loss_scaling="normalized"` | e = MSE/var(y), s = min(1, nós/30) | Termos em escala comparável a c (constatação 3) |
| V2 | `extra_ops=True` | `sqrt` protegida nos três SR; `exp` no DEAP | A verdade-base da queda livre é uma raiz; a Tabela 1 retira exp do DEAP (C1) |
| V3 | `min_generations=10` | a parada antecipada só dispara após 10 gerações | Constatação 4 |
| V4 | `deap_init_depth=(1, 3)` | profundidade inicial do DEAP | Com 1, o DEAP literal devolvia constantes |

## 9. Código legado

`src/legacy/` contém o código original, sem alterações. Nele, o LLM avalia apenas a equação final e os pesos otimizados nunca chegam ao SR (ver `relatorio_review_SR-LLM.md`).
