"""Is a metric difference real or noise? Per-user paired bootstrap.

Resample USERS (not items) with replacement, recompute the mean difference each time.
Pairing matters: both systems are scored on the same users, so user difficulty cancels out.
"""
import numpy as np


def per_user(recs: dict, truth: dict, fn, users, k=10):
    return np.array([fn(recs[int(u)], truth[int(u)], k) for u in users])


def bootstrap_ci(x, n_boot=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def paired_bootstrap(a, b, n_boot=5000, alpha=0.05, seed=0):
    """Difference b - a. Returns mean diff, (lo, hi) CI, two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b) - np.asarray(a)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()),
            "ci95": tuple(map(float, np.quantile(boots, [alpha / 2, 1 - alpha / 2]))),
            "p": float(min(p, 1.0))}
