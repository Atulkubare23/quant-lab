"""Shared aggregation for sensitivity-based capital (FRTB-SA and ISDA SIMM share the same algebra).

    K_b  = sqrt(max(0, sum_k WS_k^2 + sum_{k != l} rho_kl WS_k WS_l))
    S_b  = sum_k WS_k   (FRTB alternative: clipped to [-K_b, K_b] when the total is negative)
    risk = sqrt(sum_b K_b^2 + sum_{b != c} gamma_bc S_b S_c)
"""
from __future__ import annotations

from math import sqrt

import numpy as np


def within(ws, rho):
    ws = np.asarray(ws, float)
    rho = np.asarray(rho, float).copy()
    np.fill_diagonal(rho, 1.0)
    return sqrt(max(0.0, float(ws @ rho @ ws)))


def across(K, S, gamma, clip=True):
    """K, S per bucket; gamma matrix. If the variance is negative, retry with S clipped to [-K, K]."""
    K, S = np.asarray(K, float), np.asarray(S, float)
    g = np.asarray(gamma, float).copy()
    np.fill_diagonal(g, 0.0)
    v = float(np.sum(K * K) + S @ g @ S)
    if v < 0 and clip:
        S = np.clip(S, -K, K)
        v = float(np.sum(K * K) + S @ g @ S)
    return sqrt(max(v, 0.0))
