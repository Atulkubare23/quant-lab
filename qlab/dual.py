"""Forward-mode automatic differentiation with dual numbers carrying a gradient vector.

Dual(x, g) represents x + g.eps where eps is a vector of infinitesimals: every arithmetic step
propagates the exact gradient, so a curve solver gets its Jacobian for free (rateslib does the
same with its Dual type).
"""
from __future__ import annotations

import math

import numpy as np


class Dual:
    __slots__ = ("x", "g")
    __array_priority__ = 1000

    def __init__(self, x, g):
        self.x = float(x)
        self.g = np.asarray(g, dtype=float)

    @staticmethod
    def variables(values):
        n = len(values)
        return [Dual(v, np.eye(n)[i]) for i, v in enumerate(values)]

    def _lift(self, o):
        return o if isinstance(o, Dual) else Dual(o, np.zeros_like(self.g))

    def __add__(self, o):
        o = self._lift(o)
        return Dual(self.x + o.x, self.g + o.g)

    __radd__ = __add__

    def __sub__(self, o):
        o = self._lift(o)
        return Dual(self.x - o.x, self.g - o.g)

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        o = self._lift(o)
        return Dual(self.x * o.x, self.g * o.x + o.g * self.x)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._lift(o)
        return Dual(self.x / o.x, (self.g * o.x - o.g * self.x) / (o.x * o.x))

    def __rtruediv__(self, o):
        return self._lift(o) / self

    def __neg__(self):
        return Dual(-self.x, -self.g)

    def __pow__(self, p):
        if isinstance(p, Dual):
            return exp(p * log(self))
        return Dual(self.x ** p, p * self.x ** (p - 1) * self.g)

    def __float__(self):
        return self.x

    def __lt__(self, o):
        return self.x < float(o)

    def __gt__(self, o):
        return self.x > float(o)

    def __repr__(self):
        return f"Dual({self.x:.10g}, {np.array2string(self.g, precision=4)})"


def value(v):
    return v.x if isinstance(v, Dual) else float(v)


def grad(v, n):
    return v.g if isinstance(v, Dual) else np.zeros(n)


def exp(v):
    if isinstance(v, Dual):
        e = math.exp(v.x)
        return Dual(e, e * v.g)
    return math.exp(v)


def log(v):
    if isinstance(v, Dual):
        return Dual(math.log(v.x), v.g / v.x)
    return math.log(v)


def sqrt(v):
    if isinstance(v, Dual):
        s = math.sqrt(v.x)
        return Dual(s, v.g / (2 * s))
    return math.sqrt(v)
