"""Synthetic content-platform data with known ground-truth preferences.

Why synthetic: with known user/item latent factors we can simulate click behaviour
and run an honest offline A/B test. Swap `generate()` for a real loader (e.g. MovieLens)
and the rest of the pipeline stays the same.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

CATEGORIES = ["komedi", "teknoloji", "spor", "muzik", "oyun", "belgesel",
              "dizi", "haber", "mutfak", "seyahat", "bilim", "moda"]
GENERIC = [f"genel_{i}" for i in range(60)]


@dataclass
class Dataset:
    items: pd.DataFrame          # item_id, category, title
    item_cat: np.ndarray         # category index per item
    U: np.ndarray                # true user latent (ground truth, used only for simulation)
    V: np.ndarray                # true item latent
    pop_bias: np.ndarray
    fav_cats: np.ndarray         # each user's 2 favourite categories (= onboarding choice)
    base: np.ndarray             # (user, item) pairs used for training
    val: np.ndarray              # pairs used to train the ranker (labels)
    test: np.ndarray             # held-out pairs for evaluation
    cold_users: np.ndarray       # users with NO training history
    warm_users: np.ndarray
    n_users: int
    n_items: int

    def true_logit(self, users, items):
        """Ground-truth preference logit, used only by the click simulator."""
        k = self.U.shape[1]
        aff = np.einsum("ij,ij->i", self.U[users], self.V[items]) / np.sqrt(k)
        return 2.0 * aff + 0.5 * self.pop_bias[items]


def generate(n_users=2000, n_items=1500, k=8, n_cold=150, seed=7) -> Dataset:
    rng = np.random.default_rng(seed)
    nc = len(CATEGORIES)
    centers = rng.normal(size=(nc, k))
    item_cat = rng.integers(0, nc, n_items)
    V = centers[item_cat] + 0.6 * rng.normal(size=(n_items, k))
    pop_bias = np.log(rng.lognormal(0, 1, n_items))
    fav = np.array([rng.choice(nc, 2, replace=False) for _ in range(n_users)])
    U = centers[fav].mean(1) + 0.5 * rng.normal(size=(n_users, k))

    logits = 2.0 * (U @ V.T) / np.sqrt(k) + 0.5 * pop_bias
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)

    users = np.arange(n_users)
    cold_users = users[-n_cold:]
    warm_users = users[:-n_cold]

    base, val, test = [], [], []
    for u in users:
        n_i = int(rng.integers(25, 60))
        its = rng.choice(n_items, n_i, replace=False, p=probs[u])
        rng.shuffle(its)
        pairs = np.column_stack([np.full(n_i, u), its])
        if u in set(cold_users.tolist()):
            test.append(pairs)                       # cold: everything is held out
        else:
            a, b = int(n_i * 0.7), int(n_i * 0.8)
            base.append(pairs[:a]); val.append(pairs[a:b]); test.append(pairs[b:])

    titles = []
    for c in item_cat:
        words = list(rng.choice([f"{CATEGORIES[c]}_{i}" for i in range(25)], 3, replace=False))
        words += list(rng.choice(GENERIC, 2, replace=False))
        titles.append(" ".join(words))
    items = pd.DataFrame({"item_id": np.arange(n_items),
                          "category": [CATEGORIES[c] for c in item_cat],
                          "title": titles})
    return Dataset(items, item_cat, U, V, pop_bias, fav,
                   np.vstack(base), np.vstack(val), np.vstack(test),
                   cold_users, warm_users, n_users, n_items)
