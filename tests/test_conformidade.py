"""As constantes do pipeline devem bater com o que o artigo declara (Tabela 1 e texto)."""
import constants as C


def test_table1_general():
    assert C.POPULATION_SIZE == 100 and C.GENERATIONS == 50


def test_table1_deap():
    d = C.DEAP
    assert (d["cxpb"], d["mutpb"], d["tournament"], d["max_height"]) == (0.05, 0.01, 3, 8)
    assert d["init_depth"] == (1, 1)
    assert set(d["binary"]) == {"add", "sub", "mul", "pdiv"}
    assert set(d["unary"]) == {"neg", "plog", "sin", "cos"}     # Tabela 1: sem exp
    assert C.DEAP_EXTRA_UNARY == ()


def test_table1_pysr_and_gplearn():
    assert set(C.PYSR["binary"]) == {"+", "-", "*", "/", "^"}
    assert set(C.PYSR["unary"]) == {"neg", "exp", "log", "sin", "cos"}
    assert C.PYSR["loss"] == "HuberLoss()" and C.PYSR["parsimony"] > 0
    assert set(C.GPLEARN["binary"]) == {"add", "sub", "mul", "div"}
    assert set(C.GPLEARN["unary"]) == {"neg", "exp", "log", "sin", "cos"}


def test_data_protocol():
    assert C.N_EXPERIMENTS == 500 and C.MAX_STEPS == 1000 and C.STEPS_PER_EXPERIMENT == 100
    assert C.MASS_RANGE == (0.1, 10.0) and C.LENGTH_RANGE == (0.01, 0.5)
    assert C.HEIGHT_RANGE == (1.0, 100.0) and C.DAMPING_RANGE == (0.0, 1.0)
    assert C.NOISE_LEVEL == 0.01 and C.NOISE_LEVELS == (0.01, 0.02, 0.03, 0.04, 0.05)
    assert C.BIAS_FRACTION == 0.01 and set(C.NOISE_MODES) == {"features", "target", "both"}
    assert C.GRAVITY == 9.81


def test_method_and_protocol():
    assert abs(sum(C.DEFAULT_WEIGHTS) - 1) < 1e-9 and C.BASELINE_WEIGHTS[2] == 0
    assert C.EARLY_STOP_TOLERANCE == 0.001 and C.EARLY_STOP_WINDOW == 3
    assert C.N_REPETITIONS == 25
    assert len(C.PROMPT_IDS) == 8 and set(C.LLM_NAMES) == {"mistral", "llama", "falcon"}
    assert 0 <= C.DISTANCE_ALPHA <= 1


def test_paper_tables_embedded():
    import plot_secondary_results as T
    assert T.paper_table2().shape == (72, 4)
    assert T.paper_table5().shape == (36, 4)
    assert T.paper_table6().shape == (27, 7)
    assert T.paper_noise(False).shape == (9, 15) and T.paper_noise(True).shape == (9, 15)
