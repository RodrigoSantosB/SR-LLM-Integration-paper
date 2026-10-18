"""Geração dos dados sintéticos (seção "In silico physical experiments", p.5–6) e ruído
("Noise robustness experiment", p.7; Tabelas 3 e 4, p.10).

Frases do artigo implementadas literalmente:
  - "we conducted N = 500 experiments, uniformly sampling five independent parameters for their
    initial condition, including: (1) mass ... 0.1 to 10 kg; (2) a characteristic length (e.g. drop
    height, spring amplitude, or radius) between 0.01 and 0.5 m; (3) initial height/displacement,
    varying from 1 to 100 m; (4) a linear drag/damping coefficient from 0 to 1 kg s^-1; and (5) time,
    sampled randomly between zero and the physically realistic maximum of sqrt(2h/g)".
  - "Each experiment is run until reaching equilibrium or 1000 steps in time."
  - "these calculations occur until an equilibrium is achieved or the first time a state is
    observed twice (for period dynamics). On average, this process took around 100 steps in time,
    resulting in around 50,000 samples for each case."
  - "we added zero-mean Gaussian noise to our ideal signals. The noise level was set to 1% of each
    signal's variability ... resulting in an average signal-to-noise ratio (SNR) of approximately 40 dB".
  - "With the five independent parameters defined, we constructed a predictor matrix X. The
    corresponding noisy signals were assigned as the target vector y. The data were retained in
    their original SI units".
  - Ruído (p.7): "we first followed the regular data generation procedure ... but without adding
    noise. We then added zero-mean Gaussian noise at five levels— 1%, ..., 5% of each signal's
    standard deviation—applying it separately to the input features and to the target variable."
  - Tabela 4 (p.8): "ε ∼ N(µ, σ²) with µ = 1% of the mean value of each feature of the data".

Interpretações (o artigo é omisso; ver docs/CONFORMIDADE.md):
  1. X tem sempre as colunas FEATURES = (m, L, h, b, t) nos três cenários, sem escala; cada GT usa
     apenas um subconjunto e as demais colunas são distratoras.
  2. GT da MHS: A = L ("spring amplitude"), k e φ fixos (SHM_SPRING_K, SHM_PHASE) [DESVIO].
     GT da onda: E0 = h ("initial displacement"), α = b ("damping coefficient"), posição x = L
     ("radius"/comprimento), k e ω fixos (WAVE_NUMBER_K, WAVE_OMEGA) [DESVIO].
  3. Cada experimento tem STEPS_PER_EXPERIMENT instantes sorteados uniformemente (ordenados) em
     [0, T_parada] — "around 100 steps" tomado como exatamente 100, o que dá 500·100 = 50.000 linhas.
  4. T_parada por cenário:
       ball_drop: impacto no solo, sqrt(2h/g) (equilíbrio; é o máximo citado pelo artigo);
       shm:       primeiro estado repetido = um período 2π·sqrt(m/k);
       wave:      envelope e^{−bt/2} < EQUILIBRIUM_FRACTION ⇒ T = 2·ln(1/EQUILIBRIUM_FRACTION)/b
                  [DESVIO no limiar], limitado a MAX_STEPS passos de dt = (2π/ω)/STEPS_PER_EXPERIMENT;
                  b = 0 nunca atinge equilíbrio ⇒ usa o limite de MAX_STEPS passos.
     O nº de passos implícito min(MAX_STEPS, ceil(T/dt)) fica em meta["implied_steps"] (documental).
  5. Ruído com σ = nível·std(coluna); no modo "features" o ruído é aplicado a todas as colunas de X
     (inclusive t). Ruído enviesado: µ = BIAS_FRACTION·média(coluna), mesmo σ.
"""
import math

import numpy as np

from constants import (BIAS_FRACTION, DAMPING_RANGE, EQUILIBRIUM_FRACTION, FEATURES, GRAVITY,
                       HEIGHT_RANGE, LENGTH_RANGE, MASS_RANGE, MAX_STEPS, N_EXPERIMENTS, NOISE_LEVEL,
                       NOISE_MODES, SHM_PHASE, SHM_SPRING_K, STEPS_PER_EXPERIMENT, WAVE_NUMBER_K,
                       WAVE_OMEGA)

_NOISE_STREAM = 0x501  # separa o gerador do ruído do gerador dos parâmetros com a mesma semente

# dt da onda: um período de oscilação dividido em STEPS_PER_EXPERIMENT passos
WAVE_DT = (2.0 * math.pi / WAVE_OMEGA) / STEPS_PER_EXPERIMENT

SCENARIOS = {
    "ball_drop": {
        "target": "v",
        "gt": f"v = sqrt(2*{GRAVITY}*h)",
        "features": FEATURES,
        "label": "Queda livre",
        "description": ("Uma bola cai a partir do repouso de uma altura h sob gravidade constante "
                        "e mede-se a sua velocidade v."),
    },
    "shm": {
        "target": "x",
        "gt": f"x = L*cos(sqrt({SHM_SPRING_K}/m)*t + {SHM_PHASE})",
        "features": FEATURES,
        "label": "Movimento harmônico simples",
        "description": ("Uma massa m presa a uma mola oscila com amplitude L e mede-se o seu "
                        "deslocamento x ao longo do tempo t."),
    },
    "wave": {
        "target": "E",
        "gt": f"E = h*exp(-b*t/2)*cos({WAVE_NUMBER_K}*L - {WAVE_OMEGA}*t)",
        "features": FEATURES,
        "label": "Onda eletromagnética amortecida",
        "description": ("Uma onda plana com amplitude inicial h e coeficiente de amortecimento b "
                        "propaga-se e mede-se o campo E na posição fixa L ao longo do tempo t."),
    },
}


def _target(scenario, m, L, h, b, t):
    """Eqs. 2–4 com as mesmas operações e constantes da string GT."""
    if scenario == "ball_drop":
        return np.sqrt(2 * GRAVITY * h)
    if scenario == "shm":
        return L * np.cos(np.sqrt(SHM_SPRING_K / m) * t + SHM_PHASE)
    if scenario == "wave":
        return h * np.exp(-b * t / 2) * np.cos(WAVE_NUMBER_K * L - WAVE_OMEGA * t)
    raise ValueError(f"cenário desconhecido: {scenario!r}")


def stop_time(scenario, m, h, b):
    """Instante de parada do experimento (equilíbrio, estado repetido ou MAX_STEPS)."""
    if scenario == "ball_drop":
        return math.sqrt(2 * h / GRAVITY)                     # impacto no solo
    if scenario == "shm":
        return 2 * math.pi * math.sqrt(m / SHM_SPRING_K)      # primeiro estado repetido
    if scenario == "wave":
        cap = MAX_STEPS * WAVE_DT
        if b <= 0:
            return cap                                        # sem amortecimento: nunca equilibra
        return min(cap, 2 * math.log(1 / EQUILIBRIUM_FRACTION) / b)
    raise ValueError(f"cenário desconhecido: {scenario!r}")


def generate_clean(scenario, n_experiments=N_EXPERIMENTS, steps=STEPS_PER_EXPERIMENT, seed=0):
    """Dados ideais (sem ruído). Retorna X (n·steps, 5) nas colunas FEATURES, y (n·steps,) e meta."""
    if scenario not in SCENARIOS:
        raise ValueError(f"cenário desconhecido: {scenario!r}")
    rng = np.random.default_rng(seed)
    blocks, t_stop, implied = [], [], []
    for _ in range(n_experiments):
        m = rng.uniform(*MASS_RANGE)
        L = rng.uniform(*LENGTH_RANGE)
        h = rng.uniform(*HEIGHT_RANGE)
        b = rng.uniform(*DAMPING_RANGE)
        T = stop_time(scenario, m, h, b)
        t = np.sort(rng.uniform(0.0, T, steps))
        const = np.ones(steps)
        blocks.append(np.column_stack([m * const, L * const, h * const, b * const, t]))
        t_stop.append(T)
        implied.append(min(MAX_STEPS, math.ceil(T / WAVE_DT)) if scenario == "wave" else steps)
    X = np.vstack(blocks)
    y = _target(scenario, *X.T)
    meta = {
        "scenario": scenario, "features": FEATURES, "target": SCENARIOS[scenario]["target"],
        "gt": SCENARIOS[scenario]["gt"], "n_experiments": n_experiments, "steps": steps,
        "seed": seed, "t_stop": np.array(t_stop), "implied_steps": np.array(implied),
        "experiment": np.repeat(np.arange(n_experiments), steps),
    }
    return X, y, meta


def _noise(a, level, biased, rng):
    """ε ~ N(µ, σ²) por coluna: σ = level·std(coluna); µ = BIAS_FRACTION·média(coluna) se biased."""
    sigma = level * a.std(axis=0)
    mu = BIAS_FRACTION * a.mean(axis=0) if biased else 0.0
    return a + rng.normal(size=a.shape) * sigma + mu


def add_noise(X, y, mode="target", level=NOISE_LEVEL, biased=False, seed=0):
    """Ruído gaussiano em X ("features"), y ("target") ou ambos ("both"). Não altera as entradas."""
    if mode not in NOISE_MODES:
        raise ValueError(f"modo de ruído inválido: {mode!r} (use {NOISE_MODES})")
    rng = np.random.default_rng((_NOISE_STREAM, seed))  # fluxo independente do de generate_clean
    X = np.array(X, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    if mode in ("features", "both"):
        X = _noise(X, level, biased, rng)
    if mode in ("target", "both"):
        y = _noise(y, level, biased, rng)
    return X, y


def snr_db(clean, noisy):
    """SNR em dB: 10·log10(var(sinal)/var(ruído))."""
    clean = np.asarray(clean, dtype=float)
    noise = np.asarray(noisy, dtype=float) - clean
    return float(10 * np.log10(clean.var() / noise.var()))


def make_dataset(scenario, noise_mode="target", noise_level=NOISE_LEVEL, biased=False,
                 n_experiments=N_EXPERIMENTS, steps=STEPS_PER_EXPERIMENT, seed=0, noise_seed=None):
    """Procedimento do artigo: gera os dados ideais e só depois adiciona o ruído."""
    X, y, meta = generate_clean(scenario, n_experiments=n_experiments, steps=steps, seed=seed)
    noise_seed = seed if noise_seed is None else noise_seed
    if noise_level:
        X, y = add_noise(X, y, mode=noise_mode, level=noise_level, biased=biased, seed=noise_seed)
    meta.update(noise_mode=noise_mode, noise_level=noise_level, biased=biased, noise_seed=noise_seed)
    return X, y, meta
