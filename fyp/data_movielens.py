"""MovieLens loader. Returns the same Dataset object as fyp.data.generate(), so the whole
pipeline (RecSys, metrics) runs unchanged.

Expects the folder of `ml-latest-small` (ratings.csv, movies.csv).
Differences from the synthetic data: there is NO ground-truth preference model,
so U/V/pop_bias are None and the click-simulating A/B test cannot be run here.
"""
import os
import numpy as np
import pandas as pd
from .data import Dataset


def load_movielens(path, min_rating=4.0, min_pos=10, cold_frac=0.1, seed=7) -> Dataset:
    rng = np.random.default_rng(seed)
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
    movies = pd.read_csv(os.path.join(path, "movies.csv"))

    pos = ratings[ratings["rating"] >= min_rating]                 # implicit "liked" signal
    counts = pos.groupby("userId")["movieId"].transform("count")
    pos = pos[counts >= min_pos]

    movies = movies[movies["movieId"].isin(pos["movieId"].unique())].reset_index(drop=True)
    item_idx = {m: i for i, m in enumerate(movies["movieId"])}
    user_ids = np.sort(pos["userId"].unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    n_users, n_items = len(user_ids), len(movies)

    first_genre = movies["genres"].str.split("|").str[0]
    cats = sorted(first_genre.unique())
    item_cat = first_genre.map({c: i for i, c in enumerate(cats)}).to_numpy()
    titles = (movies["title"].str.replace(r"\(\d{4}\)", "", regex=True) + " "
              + movies["genres"].str.replace("|", " ", regex=False))
    items = pd.DataFrame({"item_id": np.arange(n_items), "category": first_genre.to_numpy(),
                          "title": titles.to_numpy()})

    pos = pos.assign(u=pos["userId"].map(user_idx), i=pos["movieId"].map(item_idx))
    pos = pos.sort_values(["u", "timestamp"])                      # temporal order per user

    cold_users = np.sort(rng.choice(n_users, max(1, int(n_users * cold_frac)), replace=False))
    cold_set = set(cold_users.tolist())
    warm_users = np.array([u for u in range(n_users) if u not in cold_set])

    base, val, test = [], [], []
    fav = np.zeros((n_users, 2), dtype=int)
    for u, g in pos.groupby("u"):
        its = g["i"].to_numpy()
        pairs = np.column_stack([np.full(len(its), u), its])
        top = np.bincount(item_cat[its], minlength=len(cats)).argsort()[::-1][:2]
        fav[u] = top                                               # "onboarding" = user's top-2 genres
        if u in cold_set:
            test.append(pairs)
        else:
            a, b = int(len(its) * 0.7), int(len(its) * 0.8)        # oldest 70% / next 10% / newest 20%
            base.append(pairs[:a]); val.append(pairs[a:b]); test.append(pairs[b:])
    return Dataset(items, item_cat, None, None, None, fav,
                   np.vstack(base), np.vstack(val), np.vstack(test),
                   cold_users, warm_users, n_users, n_items)
