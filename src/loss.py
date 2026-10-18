"""Função de perda composta do artigo e critério de parada.

Eq. 1 (p.4):  L = w1·e + w2·s + w3·c,  w1, w2, w3 ∈ [0, 1],  w1 + w2 + w3 = 1
  e = erro quadrático médio entre a predição e os dados;
  s = tamanho da equação em número de nós da árvore de expressão;
  c = 1 − (c1 + c2 + c3)/3, com c1, c2, c3 ∈ [0, 1] devolvidos pelo LLM (p.5).

O termo c entra na avaliação de cada expressão candidata durante a busca (o SR chama
`Objective.evaluate` a cada geração). Com w3 = 0 (linha de base) o LLM não é consultado.

Substitui o `ga_weight_optimizer.py` original (preservado em `legacy/`), no qual os pesos
nunca chegavam ao SR.
"""
import math

import numpy as np

import expression_tree as ET
from constants import (EARLY_STOP_TOLERANCE, EARLY_STOP_WINDOW, FAIL_LOSS, LOSS_SCALING,
                       S_MAX)


def check_weights(weights):
    w = tuple(float(v) for v in weights)
    if len(w) != 3 or any(v < 0 or v > 1 for v in w) or not math.isclose(sum(w), 1.0, abs_tol=1e-6):
        raise ValueError(f"pesos inválidos {w}: exige w1,w2,w3 ∈ [0,1] e soma 1 (Eq. 1)")
    return w


class Objective:
    """Avalia expressões candidatas com a loss composta.

    Parâmetros
    ----------
    y : alvo observado.
    target : nome da variável-alvo (o LLM recebe ``"alvo = expressão"``).
    weights : (w1, w2, w3).
    scorer : avaliador LLM com ``score_many([(chave, texto)], contexto) -> [c_médio]``; None = sem LLM.
    context : texto adicional do prompt (variantes A–H).
    rename : mapeia nomes de variáveis do SR (x0, X0, ARG0) para os nomes físicos.
    scaling : "paper" (MSE e nº de nós crus, literal) ou "normalized" (MSE/var(y), min(1, nós/S_MAX)).
    """

    def __init__(self, y, target, weights, scorer=None, context="", rename=None,
                 scaling=LOSS_SCALING, s_max=S_MAX):
        self.y = np.asarray(y, dtype=float)
        self.var_y = float(np.var(self.y)) + 1e-12
        self.target = target
        self.w = check_weights(weights)
        self.scorer = scorer
        self.use_llm = scorer is not None and self.w[2] > 0
        self.context = context
        self.rename = rename or {}
        if scaling not in ("paper", "normalized"):
            raise ValueError(f"scaling desconhecido: {scaling}")
        self.scaling = scaling
        self.s_max = s_max
        self._trees = {}
        self.n_evaluated = 0

    def tree(self, expr_str):
        t = self._trees.get(expr_str)
        if t is None:
            try:
                t = ET.parse(expr_str, self.rename)
            except (ET.ParseError, RecursionError):
                t = False
            self._trees[expr_str] = t
        return t

    def llm_text(self, expr_str):
        t = self.tree(expr_str)
        return f"{self.target} = {ET.to_infix(t)}" if t is not False else None

    def _terms(self, mse, size):
        if self.scaling == "paper":
            return mse, float(size)
        return min(mse / self.var_y, 1e6), min(1.0, size / self.s_max)

    def evaluate(self, items):
        """items: lista de (expr_str, y_pred). Retorna, para cada item, um dict com
        loss, mse, e, size, s, plaus (média c1..c3) e c. Todas as expressões novas vão ao
        LLM num único lote."""
        rows, ask = [], []
        for expr_str, y_pred in items:
            self.n_evaluated += 1
            t = self.tree(expr_str)
            y_pred = np.asarray(y_pred, dtype=float)
            if y_pred.ndim == 0:
                y_pred = np.full_like(self.y, float(y_pred))
            ok = t is not False and y_pred.shape == self.y.shape and np.all(np.isfinite(y_pred))
            mse = float(np.mean((self.y - y_pred) ** 2)) if ok else math.inf
            row = {"expr": expr_str, "mse": mse, "size": np.nan, "plaus": np.nan}
            if not ok or not math.isfinite(mse):
                row.update(loss=FAIL_LOSS, e=math.inf, s=np.nan, c=np.nan)
            else:
                row["size"] = ET.size(t)
                if self.use_llm:
                    text = f"{self.target} = {ET.to_infix(t)}"
                    ask.append((len(rows), text, text))
            rows.append(row)
        if ask:
            scores = self.scorer.score_many([(k, txt) for _, k, txt in ask], self.context)
            for (i, _, _), sc in zip(ask, scores):
                rows[i]["plaus"] = float(sc)
        w1, w2, w3 = self.w
        for r in rows:
            if "loss" in r:
                continue
            e, s = self._terms(r["mse"], r["size"])
            c = (1.0 - r["plaus"]) if self.use_llm else 0.0
            r.update(e=e, s=s, c=c, loss=w1 * e + w2 * s + w3 * c)
        return rows


class EarlyStopper:
    """"Optimization stopped if the best composite loss improved by less than 0.1% over three
    generations" (p.6): compara a melhor loss atual com a de três gerações antes."""

    def __init__(self, tol=EARLY_STOP_TOLERANCE, window=EARLY_STOP_WINDOW):
        self.tol, self.window = tol, window
        self.history = []

    def update(self, best_loss):
        best = min(best_loss, self.history[-1]) if self.history else best_loss
        self.history.append(best)
        if len(self.history) <= self.window:
            return False
        old = self.history[-1 - self.window]
        if not math.isfinite(old):
            return False
        improvement = (old - best) / (abs(old) + 1e-12)
        return improvement < self.tol
