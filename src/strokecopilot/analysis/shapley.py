"""Exact Shapley values by enumeration (DEC-35), for games with few players.

A game is n players and a value function on coalitions. Every one of the 2**n coalitions is evaluated once, so the
values have no sampling error. A coalition is an int bitmask: bit i set means player i is present.

    phi_i = sum over S not containing i of |S|! (n - |S| - 1)! / n! * (v(S + i) - v(S))

Values satisfy efficiency: sum_i phi_i = v(all) - v(none). The value function may return a number or a 1-D array
(several outputs scored on the same coalitions, for example Dice and volume).
"""

from math import factorial

import numpy as np

MAX_PLAYERS = 16  # 65,536 coalitions; above this, enumeration is not the right tool


def coalition_values(n, value):
    """v for every coalition, as an array indexed by bitmask: shape (2**n,) or (2**n, k)."""
    if not 0 < n <= MAX_PLAYERS:
        raise ValueError(f"exact enumeration needs 1 to {MAX_PLAYERS} players, got {n}")
    return np.asarray([value(mask) for mask in range(1 << n)], dtype=float)


def shapley_from_values(v):
    """Exact Shapley values from the full table of coalition values. Returns shape (n,) or (n, k)."""
    v = np.asarray(v, dtype=float)
    size = v.shape[0]
    n = size.bit_length() - 1
    if size != 1 << n:
        raise ValueError("the table must hold 2**n coalition values")
    masks = np.arange(size)
    popcount = np.array([bin(m).count("1") for m in masks])
    weight = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) if s < n else 0.0 for s in range(n + 1)])
    phi = np.zeros((n,) + v.shape[1:])
    for i in range(n):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        w = weight[popcount[without]]
        diff = v[without | bit] - v[without]
        phi[i] = np.tensordot(w, diff, axes=(0, 0))
    return phi


def shapley(n, value):
    """Exact Shapley values of an n-player game: (phi, v_all, v_none)."""
    v = coalition_values(n, value)
    return shapley_from_values(v), v[-1], v[0]


def members(mask, n):
    """The player indices present in a coalition."""
    return [i for i in range(n) if mask >> i & 1]
