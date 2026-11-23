"""Adjoint algorithmic differentiation (reverse mode) on numpy arrays, for Monte Carlo Greeks.

Each operation records its inputs and a vector-Jacobian product; backward() walks the tape in
reverse once, so every sensitivity of a price (all deltas, all vegas, rho...) costs about one
extra pricing, whatever the number of inputs. This is the idea in Savine (2018) and Capriotti &
Giles (2010), written with whole-array nodes so that Python stays fast.
"""
from __future__ import annotations

import numpy as np


class Var:
    __array_priority__ = 1000

    def __init__(self, value, parents=()):
        self.value = np.asarray(value, dtype=float)
        self.parents = parents            # tuple of (Var, vjp)
        self.adj = None

    # -- graph ops
    @staticmethod
    def _wrap(o):
        return o if isinstance(o, Var) else Var(o)

    def __add__(self, o):
        o = Var._wrap(o)
        return Var(self.value + o.value, ((self, lambda g: g), (o, lambda g: g)))

    __radd__ = __add__

    def __sub__(self, o):
        o = Var._wrap(o)
        return Var(self.value - o.value, ((self, lambda g: g), (o, lambda g: -g)))

    def __rsub__(self, o):
        return Var._wrap(o) - self

    def __mul__(self, o):
        o = Var._wrap(o)
        a, b = self.value, o.value
        return Var(a * b, ((self, lambda g: g * b), (o, lambda g: g * a)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Var._wrap(o)
        a, b = self.value, o.value
        return Var(a / b, ((self, lambda g: g / b), (o, lambda g: -g * a / (b * b))))

    def __rtruediv__(self, o):
        return Var._wrap(o) / self

    def __neg__(self):
        return Var(-self.value, ((self, lambda g: -g),))

    def __pow__(self, p):
        a = self.value
        return Var(a ** p, ((self, lambda g: g * p * a ** (p - 1)),))

    def __getitem__(self, idx):
        shape = self.value.shape
        def vjp(g):
            out = np.zeros(shape)
            np.add.at(out, idx, g)
            return out
        return Var(self.value[idx], ((self, vjp),))

    # -- reverse sweep
    def backward(self):
        order, seen = [], set()
        stack = [(self, False)]
        while stack:                      # iterative topological sort
            v, done = stack.pop()
            if done:
                order.append(v)
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            for p, _ in v.parents:
                if id(p) not in seen:
                    stack.append((p, False))
        for v in order:
            v.adj = None
        self.adj = np.ones_like(self.value)
        for v in reversed(order):
            if v.adj is None:
                continue
            for p, vjp in v.parents:
                g = _unbroadcast(vjp(v.adj), p.value.shape)
                p.adj = g if p.adj is None else p.adj + g


def _unbroadcast(g, shape):
    g = np.asarray(g, dtype=float)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


def exp(x):
    e = np.exp(x.value)
    return Var(e, ((x, lambda g: g * e),))


def log(x):
    a = x.value
    return Var(np.log(a), ((x, lambda g: g / a),))


def sqrt(x):
    s = np.sqrt(x.value)
    return Var(s, ((x, lambda g: g / (2 * s)),))


def relu(x, k=0.0):
    """max(x - k, 0); pathwise derivative is the indicator (fine for Lipschitz payoffs)."""
    a = x.value - k
    return Var(np.maximum(a, 0), ((x, lambda g: g * (a > 0)),))


def mean(x, axis=None):
    a = x.value
    n = a.size if axis is None else a.shape[axis]
    def vjp(g):
        g = np.asarray(g)
        if axis is not None:
            g = np.expand_dims(g, axis)
        return np.broadcast_to(g, a.shape) / n
    return Var(a.mean(axis=axis), ((x, vjp),))


def sum_(x, axis=None):
    a = x.value
    def vjp(g):
        g = np.asarray(g)
        if axis is not None:
            g = np.expand_dims(g, axis)
        return np.broadcast_to(g, a.shape)
    return Var(a.sum(axis=axis), ((x, vjp),))


# ---------------------------------------------------------------- Monte Carlo pricers on the tape
def basket_call(S0, sigma, r, w, K, T, corr, n_paths=200_000, seed=0, adjoint=True):
    """European call on sum_i w_i S_i(T), correlated GBM. Returns price and, with adjoint=True,
    the gradient dict {delta: dV/dS0_i, vega: dV/dsigma_i, rho: dV/dr} from one backward pass."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_paths, len(S0))) @ L.T
    s0, sg, rr = Var(np.asarray(S0, float)), Var(np.asarray(sigma, float)), Var(np.asarray(r, float))
    ST = s0 * exp((rr - 0.5 * sg * sg) * T + sg * np.sqrt(T) * z)
    B = sum_(ST * np.asarray(w, float), axis=1)
    V = exp(-rr * T) * mean(relu(B, K))
    if not adjoint:
        return float(V.value)
    V.backward()
    return float(V.value), {"delta": s0.adj, "vega": sg.adj, "rho": float(rr.adj)}


def asian_call(S0, sigma, r, K, T, n_steps=52, n_paths=100_000, seed=0):
    """Arithmetic-average Asian call, one Var per time step: a long tape, one backward pass."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    s0, sg, rr = Var(float(S0)), Var(float(sigma)), Var(float(r))
    S, acc = s0 * 1.0, None
    for k in range(n_steps):
        z = rng.standard_normal(n_paths)
        S = S * exp((rr - 0.5 * sg * sg) * dt + sg * np.sqrt(dt) * z)
        acc = S if acc is None else acc + S
    V = exp(-rr * T) * mean(relu(acc / n_steps, K))
    V.backward()
    return float(V.value), {"delta": float(s0.adj), "vega": float(sg.adj), "rho": float(rr.adj)}


if __name__ == "__main__":
    import time
    n = 10
    rho = 0.5
    corr = np.full((n, n), rho) + (1 - rho) * np.eye(n)
    S0, sig, w = np.full(n, 100.0), np.linspace(0.15, 0.35, n), np.full(n, 1 / n)
    t0 = time.perf_counter()
    v = basket_call(S0, sig, 0.03, w, 100, 1.0, corr, adjoint=False)
    t1 = time.perf_counter()
    v, g = basket_call(S0, sig, 0.03, w, 100, 1.0, corr)
    t2 = time.perf_counter()
    print(f"10-asset basket call {v:.4f}; price {1e3 * (t1 - t0):.0f} ms, price + all 21 Greeks {1e3 * (t2 - t1):.0f} ms")
    print("  deltas", np.round(g["delta"], 4))
    print("  vegas ", np.round(g["vega"], 3))
    print(f"  rho {g['rho']:.3f}")


# ---------------------------------------------------------------- neural-network ops (used by deephedge)
def matmul(x, W):
    a, b = x.value, W.value
    return Var(a @ b, ((x, lambda g: g @ b.T), (W, lambda g: a.T @ g)))


def tanh(x):
    t = np.tanh(x.value)
    return Var(t, ((x, lambda g: g * (1 - t * t)),))


def sigmoid(x):
    s = 1 / (1 + np.exp(-x.value))
    return Var(s, ((x, lambda g: g * s * (1 - s)),))


def abs_(x):
    a = x.value
    return Var(np.abs(a), ((x, lambda g: g * np.sign(a)),))


def concat(xs, axis=-1):
    vals = [x.value for x in xs]
    sizes = np.cumsum([v.shape[axis] for v in vals])[:-1]
    def make(i):
        return lambda g: np.split(g, sizes, axis=axis)[i]
    return Var(np.concatenate(vals, axis=axis), tuple((x, make(i)) for i, x in enumerate(xs)))
