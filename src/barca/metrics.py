"""Metric helpers (week 2 adds the rest)."""
import numpy as np


def gini(x):
    """Gini coefficient of non-negative values (0 = equal, ->1 = concentrated)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
