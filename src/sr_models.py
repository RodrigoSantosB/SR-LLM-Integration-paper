"""Modelos de regressão simbólica com a loss composta (Eq. 1) DENTRO da busca.

Seção "Symbolic regression models" (p.6) e Tabela 1 (p.7): DEAP, PySR e gplearn, população
100, 50 gerações, early stopping "if the best composite loss improved by less than 0.1% over
three generations". Seção "LLM-based knowledge integration to SR" (p.4–5): o score do LLM é um
termo da loss do SR, avaliado durante a busca.

Interface comum
---------------
    m = Model(feature_names, pop_size=POPULATION_SIZE, generations=GENERATIONS, seed=0, **overrides)
    obj = loss.Objective(y, target, weights, scorer, rename=m.rename)
    m.fit(X, obj); m.predict(X)
    m.best_expr (texto nativo), m.best_row (linha da Objective), m.history (por geração:
    gen, best_loss, best_mse, n_new, llm_seconds), m.n_generations, m.sr_seconds,
    m.llm_seconds, m.get_best_equation().

Em cada geração, todas as expressões novas da população vão para UMA chamada de
`objective.evaluate` (um lote do LLM). Resultados são memorizados por texto da expressão.

Tempos: `llm_seconds` = soma dos deltas de `objective.scorer.seconds` (se existir);
`sr_seconds` = tempo de parede do fit menos `llm_seconds`; `total_seconds` = tempo de parede.

Contagem de gerações: DEAP e gplearn registram a geração 0 (população inicial) e mais
`generations` gerações evoluídas (como `eaSimple`); PySR registra as iterações 1..`generations`.
"""
import functools
import operator
import random
import shutil
import tempfile
import time
import warnings

import numpy as np
from deap import base, creator, gp, tools

import expression_tree as ET
from constants import DEAP, DEAP_EXTRA_UNARY, GENERATIONS, GPLEARN, POPULATION_SIZE, PYSR
from loss import EarlyStopper

# [DESVIO] não especificados pelo artigo:
EPHEMERAL_RANGE = (-10.0, 10.0)  # constantes efêmeras do DEAP ~ U(−10, 10)
EPHEMERAL_DIGITS = 2             # ... arredondadas a 2 casas (expressões legíveis para o LLM)
PROTECT_EPS = 1e-9               # limiar da divisão/log protegidos
EXP_CLIP = 50.0                  # exp protegido: argumento limitado a ±50
DEAP_MUT_DEPTH = (0, 2)          # subárvore da mutação uniforme (genFull), como em eaSimple/exemplos DEAP
PYSR_CYCLES_PER_ITER = None      # padrão do PySR (como no código original); "iteração" do artigo = iteração do PySR
                                  # (Tabela 6: ~50–60 s por iteração do PySR, compatível com o padrão)
PYSR_CONSTRAINTS = {"^": (-1, 1)}         # código original dos autores (legacy/sr_models.py)
PYSR_OPERATOR_COMPLEXITY = {"^": 4}       # idem


# ---------------------------------------------------------------- operadores protegidos
def pdiv(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.ones_like(a)
    ok = np.abs(b) > PROTECT_EPS
    out[ok] = a[ok] / b[ok]
    return out


def plog(a):
    a = np.abs(np.asarray(a, float))
    return np.where(a > PROTECT_EPS, np.log(np.maximum(a, PROTECT_EPS)), 0.0)


def psqrt(a):
    """Raiz protegida: sqrt(|a|) [AJUSTE pós-piloto, operador extra do perfil "adjusted"]."""
    return np.sqrt(np.abs(np.asarray(a, float)))


def pexp(a):
    with np.errstate(over="ignore"):
        return np.exp(np.clip(a, -EXP_CLIP, EXP_CLIP))


def _rand_const():
    return round(random.uniform(*EPHEMERAL_RANGE), EPHEMERAL_DIGITS)


DEAP_PRIMITIVES = {
    "add": (np.add, 2), "sub": (np.subtract, 2), "mul": (np.multiply, 2), "pdiv": (pdiv, 2),
    "neg": (np.negative, 1), "plog": (plog, 1), "sin": (np.sin, 1), "cos": (np.cos, 1),
    "pexp": (pexp, 1), "psqrt": (psqrt, 1),
}


def _as_vector(out, n):
    return np.broadcast_to(np.asarray(out, float), (n,)).copy()


# ---------------------------------------------------------------- base
class BaseSR:
    """Estado e utilidades comuns: avaliação em lote com memória, relógio do LLM, histórico."""
    name = "base"

    def __init__(self, feature_names, pop_size=POPULATION_SIZE, generations=GENERATIONS, seed=0,
                 verbose=False, min_generations=0):
        self.feature_names = list(feature_names)
        self.min_generations = int(min_generations)
        self.pop_size, self.generations = int(pop_size), int(generations)
        self.seed = seed
        self.verbose = verbose
        self.rename = {n: n for n in self.feature_names}
        self.best_expr, self.best_row = None, None
        self.history = []
        self.n_generations = 0
        self.sr_seconds = self.llm_seconds = self.total_seconds = 0.0

    # -- tempo gasto dentro do avaliador LLM
    @staticmethod
    def _llm_clock(objective):
        return float(getattr(getattr(objective, "scorer", None), "seconds", 0.0) or 0.0)

    def _start(self, objective):
        self._cache = {}
        self._stopper = EarlyStopper(min_generations=self.min_generations)
        self.history = []
        self.best_expr, self.best_row = None, None
        self.llm_seconds = 0.0
        self._t0 = time.time()

    def _evaluate(self, objective, items):
        """items: lista de (texto, y_pred_callable). Avalia só os textos ainda não vistos, em
        UM lote. Retorna (linhas na ordem de items, nº de novos, segundos de LLM)."""
        new = {}
        for k, pred in items:
            if k not in self._cache and k not in new:
                try:
                    with np.errstate(all="ignore"):
                        new[k] = pred()
                except Exception:  # expressão inválida → loss de falha na Objective
                    new[k] = np.full(objective.y.shape, np.nan)
        c0 = self._llm_clock(objective)
        if new:
            keys = list(new)
            for k, r in zip(keys, objective.evaluate([(k, new[k]) for k in keys])):
                self._cache[k] = r
        dt = self._llm_clock(objective) - c0
        self.llm_seconds += dt
        return [self._cache[k] for k, _ in items], len(new), dt

    def _record(self, gen, n_new, dt):
        """Registra a geração e devolve True se o early stopping disparar."""
        best = self.best_row["loss"]
        self.history.append(dict(gen=gen, best_loss=best, best_mse=self.best_row["mse"],
                                 n_new=n_new, llm_seconds=dt))
        self.n_generations = gen
        if self.verbose:
            print(f"[{self.name}] gen {gen} best={best:.4g} new={n_new}")
        return self._stopper.update(best)

    def _finish(self):
        self.total_seconds = time.time() - self._t0
        self.sr_seconds = self.total_seconds - self.llm_seconds

    @property
    def best_infix(self):
        """Melhor expressão em notação infixa com nomes físicos."""
        return ET.to_infix(ET.parse(self.best_expr, self.rename))

    def get_best_equation(self):  # compatibilidade com legacy/sr_models.py
        return self.best_expr


# ---------------------------------------------------------------- DEAP
# nomes de classe únicos para não colidir com legacy/ e colab/
if not hasattr(creator, "FitnessMinPaper"):
    creator.create("FitnessMinPaper", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualPaper"):
    creator.create("IndividualPaper", gp.PrimitiveTree, fitness=creator.FitnessMinPaper)


class DEAPModel(BaseSR):
    """DEAP-GP (Tabela 1): torneio 3, cxpb 0.05, mutpb 0.01, altura máx. 8, profundidade
    inicial 1, operadores binários add/sub/mul/pdiv e unários neg/plog/sin/cos (+ DEAP_EXTRA_UNARY).
    Usa TODAS as colunas de X (o original usava só X[:, 0]); argumentos com nomes físicos."""
    name = "deap"

    def __init__(self, feature_names, pop_size=POPULATION_SIZE, generations=GENERATIONS, seed=0,
                 verbose=False, extra_unary=DEAP_EXTRA_UNARY, min_generations=0, **overrides):
        super().__init__(feature_names, pop_size, generations, seed, verbose, min_generations)
        unknown = set(overrides) - set(DEAP)
        if unknown:
            raise TypeError(f"parâmetros DEAP desconhecidos: {sorted(unknown)}")
        self.params = {**DEAP, **overrides}
        self.primitive_names = tuple(dict.fromkeys(
            tuple(self.params["binary"]) + tuple(self.params["unary"]) + tuple(extra_unary)))
        self.pset = self._build_pset()

    def _build_pset(self):
        pset = gp.PrimitiveSet("MAIN", len(self.feature_names))
        for nm in self.primitive_names:
            f, arity = DEAP_PRIMITIVES[nm]
            pset.addPrimitive(f, arity, name=nm)
        pset.addEphemeralConstant("rconst_paper", _rand_const)
        pset.renameArguments(**{f"ARG{i}": n for i, n in enumerate(self.feature_names)})
        return pset

    def _toolbox(self):
        p, pset = self.params, self.pset
        tb = base.Toolbox()
        tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=p["init_depth"][0], max_=p["init_depth"][1])
        tb.register("individual", tools.initIterate, creator.IndividualPaper, tb.expr)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("compile", gp.compile, pset=pset)
        tb.register("select", tools.selTournament, tournsize=p["tournament"])
        tb.register("mate", gp.cxOnePoint)
        tb.register("expr_mut", gp.genFull, min_=DEAP_MUT_DEPTH[0], max_=DEAP_MUT_DEPTH[1])
        tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
        limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=p["max_height"])
        tb.decorate("mate", limit)
        tb.decorate("mutate", limit)
        return tb

    def _func(self, ind, X):
        f = self._tb.compile(expr=ind)
        return _as_vector(f(*[X[:, i] for i in range(X.shape[1])]), X.shape[0])

    def _eval_pop(self, inds, X, objective):
        """Atribui a loss composta a cada indivíduo (lote único por geração)."""
        rows, n_new, dt = self._evaluate(
            objective, [(str(i), functools.partial(self._func, i, X)) for i in inds])
        for ind, r in zip(inds, rows):
            ind.fitness.values = (r["loss"],)
        return n_new, dt

    def fit(self, X, objective):
        X = np.asarray(X, float)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._tb = tb = self._toolbox()
        cxpb, mutpb = self.params["cxpb"], self.params["mutpb"]
        self._start(objective)
        hof = tools.HallOfFame(1)

        pop = tb.population(n=self.pop_size)
        n_new, dt = self._eval_pop(pop, X, objective)
        hof.update(pop)
        self._set_best(hof)
        stop = self._record(0, n_new, dt)
        gen = 1
        while not stop and gen <= self.generations:  # eaSimple: seleção → varAnd → avaliação
            off = [tb.clone(i) for i in tb.select(pop, len(pop))]
            for i in range(1, len(off), 2):
                if random.random() < cxpb:
                    off[i - 1], off[i] = tb.mate(off[i - 1], off[i])
                    del off[i - 1].fitness.values, off[i].fitness.values
            for i in range(len(off)):
                if random.random() < mutpb:
                    (off[i],) = tb.mutate(off[i])
                    del off[i].fitness.values
            invalid = [i for i in off if not i.fitness.valid]
            n_new, dt = self._eval_pop(invalid, X, objective)
            pop[:] = off
            hof.update(pop)
            self._set_best(hof)
            stop = self._record(gen, n_new, dt)
            gen += 1
        self.best_ = hof[0]
        self._finish()
        return self

    def _set_best(self, hof):
        self.best_expr = str(hof[0])
        self.best_row = self._cache[self.best_expr]

    def predict(self, X):
        with np.errstate(all="ignore"):
            return self._func(self.best_, np.asarray(X, float))


# ---------------------------------------------------------------- gplearn
def _gplearn_exp():
    from gplearn.functions import make_function
    return make_function(function=pexp, name="exp", arity=1)


class GplearnModel(BaseSR):
    """gplearn (Tabela 1): add, sub, mul, div, neg, exp (protegido), log, sin, cos.

    `make_fitness` do gplearn só vê (y, y_pred), não o programa; por isso a evolução avança
    uma geração por vez (warm_start) e, após cada geração, `raw_fitness_`/`fitness_` de todos
    os programas são sobrescritos pela loss composta. O torneio da geração seguinte lê
    `parents[i].fitness_` e, com metric="mse" (greater_is_better=False), escolhe o mínimo.
    """
    name = "gplearn"

    def __init__(self, feature_names, pop_size=POPULATION_SIZE, generations=GENERATIONS, seed=0,
                 verbose=False, extra_unary=(), min_generations=0, **overrides):
        super().__init__(feature_names, pop_size, generations, seed, verbose, min_generations)
        p = {**GPLEARN, **overrides}
        p["unary"] = tuple(dict.fromkeys(tuple(p["unary"]) + tuple(extra_unary)))
        self.params = p
        self.rename = {f"X{i}": n for i, n in enumerate(self.feature_names)}
        self.function_names = tuple(p["binary"]) + tuple(p["unary"])
        self.extra_params = {k: v for k, v in overrides.items() if k not in GPLEARN}

    def _function_set(self):
        return tuple(_gplearn_exp() if f == "exp" else f for f in self.function_names)

    def _estimator(self):
        from gplearn.genetic import SymbolicRegressor
        est = SymbolicRegressor(
            population_size=self.pop_size, generations=1, function_set=self._function_set(),
            metric="mse", parsimony_coefficient=self.params["parsimony"],
            stopping_criteria=-np.inf,  # sem parada interna; o early stopping é o do artigo
            warm_start=True, low_memory=True, n_jobs=1, random_state=self.seed,
            **self.extra_params)
        return est

    def fit(self, X, objective):
        X = np.asarray(X, float)
        est = self._estimator()
        self._start(objective)
        best_prog = None
        gen = 0
        while gen <= self.generations:
            est.set_params(generations=gen + 1)  # gen 0 = população inicial
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X, objective.y)
            assert not est._metric.greater_is_better
            population = est._programs[-1]
            rows, n_new, dt = self._evaluate(
                objective, [(str(p), functools.partial(p.execute, X)) for p in population])
            for p, r in zip(population, rows):
                p.raw_fitness_ = p.fitness_ = r["loss"]
                if self.best_row is None or r["loss"] < self.best_row["loss"]:
                    best_prog, self.best_row, self.best_expr = p, r, str(p)
            if self._record(gen, n_new, dt):
                break
            gen += 1
        self.best_ = best_prog
        self.estimator_ = est
        self._finish()
        return self

    def predict(self, X):
        with np.errstate(all="ignore"):
            return _as_vector(self.best_.execute(np.asarray(X, float)), len(X))


# ---------------------------------------------------------------- PySR
class PySRModel(BaseSR):
    """PySR (Tabela 1): + − * / ^ e neg, exp, log, sin, cos; Huber loss; penalidade de
    parcimônia; dados em unidades SI (sem padronização).

    [DESVIO DOCUMENTADO] a evolução do PySR roda em Julia e não aceita callback Python por
    candidato. A busca avança uma iteração por chamada de fit() (warm_start, niterations=1,
    uma chamada de fit() = uma iteração do PySR, com o nº padrão de ciclos); após cada chamada, o Hall of Fame
    (`equations_`) é avaliado pela loss composta (MSE + tamanho + LLM) em um lote e a equação
    de menor loss composta é selecionada. Ou seja, o LLM re-ranqueia o Hall of Fame e alimenta
    o early stopping, mas não a seleção interna do Julia.
    """
    name = "pysr"

    def __init__(self, feature_names, pop_size=POPULATION_SIZE, generations=GENERATIONS, seed=0,
                 verbose=False, extra_unary=(), min_generations=0, **overrides):
        super().__init__(feature_names, pop_size, generations, seed, verbose, min_generations)
        self.binary_operators = list(PYSR["binary"])
        self.unary_operators = list(dict.fromkeys(list(PYSR["unary"]) + list(extra_unary)))
        self.pysr_params = dict(
            populations=PYSR["populations"], population_size=self.pop_size,
            niterations=1, warm_start=True,
            **({"ncycles_per_iteration": PYSR_CYCLES_PER_ITER} if PYSR_CYCLES_PER_ITER else {}),
            binary_operators=self.binary_operators, unary_operators=self.unary_operators,
            elementwise_loss=PYSR["loss"], parsimony=PYSR["parsimony"],
            constraints=dict(PYSR_CONSTRAINTS), complexity_of_operators=dict(PYSR_OPERATOR_COMPLEXITY),
            deterministic=True, parallelism="serial", random_state=seed,
            verbosity=0, progress=False,
        )
        self.pysr_params.update(overrides)

    def fit(self, X, objective):
        from pysr import PySRRegressor  # import tardio: juliacall antes de torch

        X = np.asarray(X, float)
        tmp = tempfile.mkdtemp(prefix="pysr_")
        model = PySRRegressor(output_directory=tmp, **self.pysr_params)
        self._start(objective)
        self._best_fn = None
        try:
            for gen in range(1, self.generations + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X, objective.y, variable_names=self.feature_names)
                eqs = model.equations_.reset_index(drop=True)
                items = [(str(row["equation"]), functools.partial(self._pred, model, X, i))
                         for i, row in eqs.iterrows()]
                rows, n_new, dt = self._evaluate(objective, items)
                for i, r in enumerate(rows):
                    if self.best_row is None or r["loss"] < self.best_row["loss"]:
                        self.best_row, self.best_expr = r, items[i][0]
                        self.best_index = i
                        self._best_fn = eqs.loc[i, "lambda_format"]
                if self._record(gen, n_new, dt):
                    break
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.model_ = model
        self._finish()
        return self

    @staticmethod
    def _pred(model, X, i):
        return _as_vector(model.predict(X, index=i), X.shape[0])

    def predict(self, X):
        """Usa a equação selecionada pela loss composta (pode ser de uma iteração anterior)."""
        X = np.asarray(X, float)
        with np.errstate(all="ignore"):
            return _as_vector(self._best_fn(X), X.shape[0])


# ---------------------------------------------------------------- Dummy
class DummySR(BaseSR):
    """SR trivial para testes: devolve sempre a constante 0."""
    name = "dummy"

    def __init__(self, feature_names=(), pop_size=POPULATION_SIZE, generations=GENERATIONS, seed=0,
                 verbose=False, **overrides):
        super().__init__(feature_names, pop_size, generations, seed, verbose)

    def fit(self, X, objective):
        n = len(X)
        self._start(objective)
        rows, n_new, dt = self._evaluate(objective, [("0", lambda: np.zeros(n))])
        self.best_expr, self.best_row = "0", rows[0]
        self._record(0, n_new, dt)
        self._finish()
        return self

    def predict(self, X):
        return np.zeros(len(X))
