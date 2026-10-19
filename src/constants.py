"""Especificações do artigo — fonte única para todo o pipeline.

Taskin, Xie & Lazebnik, "Knowledge integration for physics-informed symbolic regression using
pre-trained large language models", Scientific Reports 16:1614 (2026).

Cada constante cita a seção/página do artigo de onde vem. Quando o artigo é omisso ou
contraditório, a escolha aparece marcada com [DESVIO] ou [CONTRADIÇÃO] e está registrada em
docs/CONFORMIDADE.md. Regra adotada para contradições: Tabela 1 literal (decisão do grupo).
"""

# ---------------------------------------------------------------- física
GRAVITY = 9.81  # m/s² — "where g = 9.81 m/s²" (In silico physical experiments, p.5)

# ---------------------------------------------------------------- dados (In silico physical experiments, p.6)
N_EXPERIMENTS = 500          # "we conducted N = 500 experiments"
STEPS_PER_EXPERIMENT = 100   # "On average, this process took around 100 steps in time"
MAX_STEPS = 1000             # "Each experiment is run until reaching equilibrium or 1000 steps in time"
TARGET_SAMPLES = 50_000      # "resulting in around 50,000 samples for each case" (informativo)
# Faixas uniformes dos cinco parâmetros independentes:
MASS_RANGE = (0.1, 10.0)        # "(1) mass, randomly selected from 0.1 to 10 kg"
LENGTH_RANGE = (0.01, 0.5)      # "(2) a characteristic length (e.g. drop height, spring amplitude, or radius) between 0.01 and 0.5 m"
HEIGHT_RANGE = (1.0, 100.0)     # "(3) initial height/displacement, varying from 1 to 100 m"
DAMPING_RANGE = (0.0, 1.0)      # "(4) a linear drag/damping coefficient from 0 to 1 kg s^-1"
# "(5) time, sampled randomly between zero and the physically realistic maximum of sqrt(2h/g)"
FEATURES = ("m", "L", "h", "b", "t")  # matriz preditora X com os cinco parâmetros, em unidades SI
# Constantes que o artigo não define para MHS e onda [DESVIO]: valores do código original
# dos autores (src/legacy/run_main_claim.py: A,k,phi = 1,1,0 ; k_w, omega = 1, 2*pi).
SHM_SPRING_K = 1.0           # N/m
SHM_PHASE = 0.0              # rad
WAVE_NUMBER_K = 1.0          # rad/m
WAVE_OMEGA = 6.283185307179586  # rad/s (2*pi)
EQUILIBRIUM_FRACTION = 0.01  # [DESVIO] critério de equilíbrio da onda: envelope < 1% de E0
# Parada da onda. "until an equilibrium is achieved or the first time a state is observed twice
# (for period dynamics). On average, this process took around 100 steps" (p.6).
# "period" [AJUSTE pós-piloto, padrão]: a onda é periódica no tempo; o experimento termina no
#   primeiro estado repetido (um período 2π/ω) ou no equilíbrio, o que vier antes — 100 passos de
#   dt = período/100, coerente com "around 100 steps" (como a MHS, que já usa um período).
# "equilibrium": versão anterior (até o envelope cair abaixo de 1%, limitado a 1000 passos —
#   ≈ 10 períodos e ≈ 997 passos implícitos, incompatível com "around 100 steps").
WAVE_STOP = "period"

NOISE_LEVEL = 0.01           # "The noise level was set to 1% of each signal's variability" (p.6)
NOISE_LEVELS = (0.01, 0.02, 0.03, 0.04, 0.05)  # "five levels— 1%, 2%, 3%, 4%, and 5%" (Noise robustness, p.7)
NOISE_MODES = ("features", "target", "both")   # Tabelas 3 e 4 (p.10)
BIAS_FRACTION = 0.01         # "ε ~ N(µ, σ²) with µ = 1% of the mean value of each feature" (Results, p.8)

# ---------------------------------------------------------------- loss (LLM-based knowledge integration, p.4–5)
# L = w1*e + w2*s + w3*c, w ∈ [0,1], w1+w2+w3 = 1 (Eq. 1). Valores não informados [DESVIO]:
# média da Dirichlet(3,2,1) usada pelo GA de pesos do código original.
DEFAULT_WEIGHTS = (0.5, 1.0 / 6.0, 1.0 / 3.0)
BASELINE_WEIGHTS = (0.75, 0.25, 0.0)  # "baseline configuration ... in which no LLM model is used" (p.6)
# "e is the mean square error", "s is the size of the analytical equation in terms of nodes".
# LOSS_SCALING = "paper" usa MSE e nº de nós crus (literal). "normalized" usa MSE/var(y) e
# min(1, nós/S_MAX) [DESVIO opcional, para termos em escala comparável].
LOSS_SCALING = "paper"
S_MAX = 30
FAIL_LOSS = 1e12

# ---------------------------------------------------------------- early stopping (Symbolic regression models, p.6)
EARLY_STOP_TOLERANCE = 0.001  # "improved by less than 0.1% ..."
EARLY_STOP_WINDOW = 3         # "... over three generations"

# ---------------------------------------------------------------- SR (Symbolic regression models, p.6; Tabela 1, p.7)
POPULATION_SIZE = 100   # "A uniform population size of 100"
GENERATIONS = 50        # "and 50 generations (iterations) are used for all three SR models"
DEAP = dict(
    cxpb=0.05,          # "crossover probability of 0.05"
    mutpb=0.01,         # "a mutation probability of 0.01"
    tournament=3,       # "a tournament selection size of 3"
    max_height=8,       # "a maximum tree depth of 8"
    init_depth=(1, 1),  # Tabela 1: "Initial tree depth 1"
    binary=("add", "sub", "mul", "pdiv"),   # Tabela 1: "+, –, *, /, protected div"
    unary=("neg", "plog", "sin", "cos"),    # Tabela 1: "neg, log, sin, cos" (sem exp)
)
# [CONTRADIÇÃO] o texto (p.6) diz "+, −, ×, ÷, exp, log, sin, and cos" para os três SR;
# a Tabela 1 retira exp do DEAP. Seguimos a Tabela 1; DEAP_EXTRA_UNARY=("pexp",) reativa exp.
DEAP_EXTRA_UNARY = ()
PYSR = dict(
    binary=("+", "-", "*", "/", "^"),            # Tabela 1
    unary=("neg", "exp", "log", "sin", "cos"),   # Tabela 1
    loss="HuberLoss()",                          # "we used the Huber loss function"
    parsimony=0.0032,  # "enabled a complexity penalty that softly discourages extra operators" [valor: padrão do PySR]
    populations=1,     # uma única população de 100 indivíduos
)
GPLEARN = dict(
    binary=("add", "sub", "mul", "div"),            # Tabela 1
    unary=("neg", "exp", "log", "sin", "cos"),      # Tabela 1 (exp protegido)
    parsimony=0.0,  # [DESVIO] "substantial complexity penalty" é exercida pelo termo s da loss composta
)

# ---------------------------------------------------------------- LLMs (Large language models, p.6)
LLM_MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "falcon": "tiiuae/falcon-7b-instruct",
}
LLM_MAX_NEW_TOKENS = 128
# "we set the sampling temperature to zero" (p.6) → decodificação gulosa (do_sample=False).
# [AJUSTE pós-piloto] Na execução piloto, LLaMA e Falcon falharam em 86–94% das respostas (texto
# puro sem modelo de conversa: LLaMA devolvia sequências vazias, Falcon texto livre). A consulta
# passa a usar o modelo de conversa (chat template) de cada modelo e a pré-preencher a resposta
# com "[", de modo que o modelo continue diretamente a lista de escores. O texto do prompt
# (PROMPT_TEMPLATE) não muda. LLM_QUERY_FORMAT entra na chave do cache: respostas obtidas com
# outro formato (inclusive os escores neutros das falhas) não são reaproveitadas.
LLM_CHAT_TEMPLATE = True
LLM_RESPONSE_PREFIX = "["
LLM_QUERY_FORMAT = "chat-prefill-v2"

# ---------------------------------------------------------------- distância (Expression tree distance metric, p.7)
DISTANCE_ALPHA = 0.1  # "the distance is α|v1 − v2| for α ∈ [0, 1], capped at one" [valor de α não informado]

# ---------------------------------------------------------------- protocolo (Experiments, p.5)
N_REPETITIONS = 25  # "the results are shown as the mean value of n = 25 repetitions"
SCENARIOS = ("ball_drop", "shm", "wave")
SR_NAMES = ("deap", "pysr", "gplearn")
LLM_NAMES = ("mistral", "llama", "falcon")
PROMPT_IDS = ("A", "B", "C", "D", "E", "F", "G", "H")

# ---------------------------------------------------------------- perfis de execução [AJUSTE pós-piloto]
# "literal": especificação do artigo (Tabela 1 e Eq. 1 literais), com as correções de defeitos
#            que valem para os dois perfis (consulta ao LLM, cache, distância, parada da onda).
# "adjusted": variantes documentadas para as inconsistências do piloto (docs/CONFORMIDADE.md, §7):
#   loss_scaling   — termos e e s em escala comparável a c (o MSE cru dominava ou era dominado);
#   extra_ops      — sqrt nos três SR e exp no DEAP (sqrt ausente da Tabela 1; a GT da queda livre é √);
#   min_generations— a parada antecipada só pode disparar após este nº de gerações (DEAP parava na 3ª);
#   deap_init_depth— profundidade inicial do DEAP (com 1, o DEAP literal devolvia constantes).
PROFILES = {
    "literal": dict(loss_scaling=LOSS_SCALING, extra_ops=False, min_generations=0,
                    deap_init_depth=DEAP["init_depth"]),
    "adjusted": dict(loss_scaling="normalized", extra_ops=True, min_generations=10,
                     deap_init_depth=(1, 3)),
}
EXTRA_OPS = dict(deap=("pexp", "psqrt"), pysr=("sqrt",), gplearn=("sqrt",))
