"""Offline ranking metrics + beyond-accuracy metrics (coverage, diversity)."""
import numpy as np


def precision_at_k(rec, truth, k):
    return len(set(rec[:k]) & truth) / k


def recall_at_k(rec, truth, k):
    return len(set(rec[:k]) & truth) / max(len(truth), 1)


def ndcg_at_k(rec, truth, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(rec[:k]) if it in truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(rec, E):
    """Mean pairwise cosine distance inside one list (E rows are L2-normalised)."""
    X = E[list(rec)]
    sim = X @ X.T
    n = len(rec)
    return float(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))


def category_entropy(rec, item_cat):
    _, counts = np.unique(item_cat[list(rec)], return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def evaluate(recs: dict, truth: dict, E, item_cat, n_items, k=10) -> dict:
    users = list(recs)
    out = {
        f"P@{k}": np.mean([precision_at_k(recs[u], truth[u], k) for u in users]),
        f"R@{k}": np.mean([recall_at_k(recs[u], truth[u], k) for u in users]),
        f"NDCG@{k}": np.mean([ndcg_at_k(recs[u], truth[u], k) for u in users]),
        "coverage": len({i for u in users for i in recs[u][:k]}) / n_items,
        "diversity": np.mean([intra_list_diversity(recs[u][:k], E) for u in users]),
        "cat_entropy": np.mean([category_entropy(recs[u][:k], item_cat) for u in users]),
    }
    return {m: float(v) for m, v in out.items()}
