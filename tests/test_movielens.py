import os, tempfile
import numpy as np, pandas as pd
from fyp.data_movielens import load_movielens
from fyp.pipeline import RecSys


def make_fixture(d, n_users=120, n_movies=300, seed=0):
    rng = np.random.default_rng(seed)
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror"]
    movies = pd.DataFrame({"movieId": np.arange(1, n_movies + 1) * 3,
                           "title": [f"Movie {i} ({1990 + i % 30})" for i in range(n_movies)],
                           "genres": ["|".join(rng.choice(genres, rng.integers(1, 3), replace=False)) for _ in range(n_movies)]})
    rows = []
    for u in range(1, n_users + 1):
        fav = rng.choice(genres, 2, replace=False)
        for m in rng.choice(n_movies, 40, replace=False):
            g = movies.loc[m, "genres"].split("|")
            r = rng.choice([4.0, 4.5, 5.0]) if set(g) & set(fav) else rng.choice([1.0, 2.0, 3.0, 4.0])
            rows.append((u, movies.loc[m, "movieId"], r, 1_000_000 + len(rows)))
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(os.path.join(d, "ratings.csv"), index=False)
    movies.to_csv(os.path.join(d, "movies.csv"), index=False)


def test_loader_split_is_temporal_and_disjoint():
    with tempfile.TemporaryDirectory() as d:
        make_fixture(d)
        ds = load_movielens(d)
        assert ds.n_users > 0 and ds.n_items > 0
        assert not set(map(tuple, ds.base)) & set(map(tuple, ds.test))
        assert not set(ds.cold_users) & set(ds.warm_users)
        assert not set(ds.cold_users) & set(ds.base[:, 0])          # cold users have no training history
        assert ds.base.max() < max(ds.n_users, ds.n_items)

def test_pipeline_runs_on_movielens_format():
    with tempfile.TemporaryDirectory() as d:
        make_fixture(d)
        ds = load_movielens(d)
        rs = RecSys(ds, n_cand=40).fit()
        recs = rs.recommend_ranked(ds.warm_users[:10], 10, mmr_lambda=0.7)
        assert all(len(v) == 10 for v in recs.values())
        assert len(rs.recommend_cold(ds.fav_cats[ds.cold_users[0]], 10)) == 10
