"""Árvores de expressão, tamanho (termo s) e distância entre árvores do artigo.

Seção "Expression tree distance metric" (p.7):
  - folhas idênticas: 0; variáveis diferentes: 1;
  - números diferentes: α|v1 − v2|, limitado a 1;
  - folha × nó interno: 1;
  - operadores diferentes em nós pareados: 1;
  - operadores comutativos (+, *): escolhe o pareamento (direto ou cruzado) de menor distância total;
  - não comutativos (−, /, ^): esquerda-esquerda e direita-direita, somando as subdistâncias;
  - as distâncias locais são somadas e normalizadas pelo número de comparações de nós → [0, 1].

A árvore é construída diretamente do texto produzido pelo SR (sem simplificação simbólica),
com o analisador sintático do Python (`ast`). Assim, "−" e "/" permanecem como nós binários
não comutativos e o tamanho reflete a expressão que o SR de fato produziu. Aceita a notação
infixa (PySR, verdade-base) e a prefixa (DEAP, gplearn: ``add(mul(x0, x1), 2.0)``).

Representação: folha = str (variável) ou float (constante); nó = (op, filho) ou (op, esq, dir).
"""
import ast
import math

from constants import DISTANCE_ALPHA

COMMUTATIVE = {"+", "*"}

# Nomes de primitivas dos três SR → operador canônico da árvore.
_BINARY_CALLS = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "pdiv": "/", "protected_div": "/",
    "pow": "^",
}
_UNARY_CALLS = {
    "neg": "neg", "sin": "sin", "cos": "cos", "exp": "exp", "pexp": "exp", "log": "log",
    "plog": "log", "sqrt": "sqrt", "abs": "abs", "inv": "inv", "square": "square", "cube": "cube",
}
_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}


class ParseError(ValueError):
    pass


def _build(node, rename):
    if isinstance(node, ast.Expression):
        return _build(node.body, rename)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return rename.get(node.id, node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        child = _build(node.operand, rename)
        return -child if isinstance(child, float) else ("neg", child)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _build(node.operand, rename)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return (_BINOPS[type(node.op)], _build(node.left, rename), _build(node.right, rename))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [_build(a, rename) for a in node.args]
        if name in _BINARY_CALLS and len(args) == 2:
            return (_BINARY_CALLS[name], args[0], args[1])
        if name in _UNARY_CALLS and len(args) == 1:
            return (_UNARY_CALLS[name], args[0])
        raise ParseError(f"função não suportada: {name}/{len(args)}")
    raise ParseError(f"nó não suportado: {ast.dump(node)[:80]}")


def parse(expr_str, rename=None):
    """Texto (infixo ou prefixo) → árvore. `rename` mapeia nomes do SR (x0, X0, ARG0) para os
    nomes físicos. Aceita "^" como potência e ignora um prefixo "alvo =".
    """
    if not isinstance(expr_str, str) or not expr_str.strip():
        raise ParseError("expressão vazia")
    s = expr_str.split("=", 1)[1] if "=" in expr_str else expr_str
    s = s.replace("^", "**")
    try:
        tree = ast.parse(s.strip(), mode="eval")
    except SyntaxError as exc:
        raise ParseError(str(exc)) from exc
    return _build(tree, rename or {})


def fold_constants(t):
    """Reduz subárvores só numéricas a uma constante (ex.: 2*9.81 → 19.62), para que a mesma
    expressão escrita com constantes agrupadas de forma diferente não seja penalizada."""
    if not isinstance(t, tuple):
        return t
    kids = [fold_constants(c) for c in t[1:]]
    if all(isinstance(k, float) for k in kids):
        try:
            v = _apply(t[0], kids)
            if math.isfinite(v):
                return float(v)
        except (ArithmeticError, ValueError, OverflowError):
            pass
    return (t[0], *kids)


def _apply(op, a):
    if len(a) == 1:
        x = a[0]
        return {"neg": lambda: -x, "sin": lambda: math.sin(x), "cos": lambda: math.cos(x),
                "exp": lambda: math.exp(x), "log": lambda: math.log(x),
                "sqrt": lambda: math.sqrt(x), "abs": lambda: abs(x), "inv": lambda: 1 / x,
                "square": lambda: x * x, "cube": lambda: x ** 3}[op]()
    x, y = a
    return {"+": x + y, "-": x - y, "*": x * y, "/": x / y, "^": x ** y}[op]


def size(t):
    """Termo s: número de nós da árvore de expressão (operadores, variáveis e constantes)."""
    if not isinstance(t, tuple):
        return 1
    return 1 + sum(size(c) for c in t[1:])


def to_infix(t, digits=4):
    """Árvore → texto infixo legível (enviado ao LLM e usado como chave de cache)."""
    if isinstance(t, float):
        return f"{t:.{digits}g}"
    if isinstance(t, str):
        return t
    op = t[0]
    if len(t) == 2:
        inner = to_infix(t[1], digits)
        return f"-({inner})" if op == "neg" else f"{op}({inner})"
    a, b = to_infix(t[1], digits), to_infix(t[2], digits)
    sym = "**" if op == "^" else op
    return f"({a} {sym} {b})"


def _leaf(t):
    return not isinstance(t, tuple)


def _dist(a, b, alpha):
    """Retorna (soma das distâncias locais, número de comparações de nós)."""
    if _leaf(a) and _leaf(b):
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) or math.isnan(b):
                return 1.0, 1
            return min(1.0, alpha * abs(a - b)), 1
        return (0.0 if a == b else 1.0), 1
    if _leaf(a) or _leaf(b):
        return 1.0, 1                      # folha × nó interno
    if a[0] != b[0] or len(a) != len(b):
        return 1.0, 1                      # operadores diferentes
    if len(a) == 2:
        s, n = _dist(a[1], b[1], alpha)
        return s, n + 1
    l, r = _dist(a[1], b[1], alpha), _dist(a[2], b[2], alpha)
    best = (l[0] + r[0], l[1] + r[1])
    if a[0] in COMMUTATIVE:
        lc, rc = _dist(a[1], b[2], alpha), _dist(a[2], b[1], alpha)
        cross = (lc[0] + rc[0], lc[1] + rc[1])
        if cross[0] < best[0]:
            best = cross
    return best[0], best[1] + 1           # +1: a comparação do próprio nó (distância 0)


def tree_distance(t1, t2, alpha=DISTANCE_ALPHA, fold=True):
    """Distância normalizada em [0, 1] entre duas árvores."""
    if fold:
        t1, t2 = fold_constants(t1), fold_constants(t2)
    s, n = _dist(t1, t2, alpha)
    return s / n


def distance(pred_str, gt_str, rename=None, alpha=DISTANCE_ALPHA):
    """Distância entre a expressão do SR e a verdade-base, ambas em texto. Expressões que não
    podem ser analisadas recebem a distância máxima (1)."""
    try:
        return tree_distance(parse(pred_str, rename), parse(gt_str), alpha)
    except (ParseError, RecursionError):
        return 1.0


class ExpressionTree:
    """Interface compatível com o código original (build_from_string / distance /
    compute_complexity), agora sobre o parser e a distância do artigo."""

    def __init__(self, root=None):
        self.root = root

    def build_from_string(self, expression_string, rename=None):
        self.root = parse(expression_string, rename)
        return self

    def distance(self, other):
        return tree_distance(self.root, other.root)

    def compute_complexity(self):
        return size(self.root)

    def __str__(self):
        return to_infix(self.root)
