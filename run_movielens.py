"""Run the pipeline on MovieLens:  python run_movielens.py data/raw/ml-latest-small"""
import sys
from fyp.data_movielens import load_movielens
from fyp.pipeline import RecSys
from fyp.metrics import evaluate

K = 10
path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ml-latest-small"
ds = load_movielens(path)
rs = RecSys(ds, n_cand=100).fit()
warm = ds.warm_users
truth = {}
for u, i in ds.test:
    truth.setdefault(int(u), set()).add(int(i))

def ev(recs, users):
    return evaluate(recs, {int(u): truth[int(u)] for u in users}, rs.E, ds.item_cat, ds.n_items, K)

systems = {
    "Popularity": rs.recommend_scores(warm, "pop", K),
    "Content-based": rs.recommend_scores(warm, "content", K),
    "Collaborative (SVD)": rs.recommend_scores(warm, "mf", K),
    "Two-stage ranker": rs.recommend_ranked(warm, K),
    "Ranker + MMR (0.7)": rs.recommend_ranked(warm, K, mmr_lambda=0.7),
}
rows = {n: ev(r, warm) for n, r in systems.items()}
cold = ds.cold_users
cold_rows = {
    "Global popularity": ev({int(u): rs.recommend_cold(ds.fav_cats[u], K, use_onboarding=False) for u in cold}, cold),
    "Onboarding + explore": ev({int(u): rs.recommend_cold(ds.fav_cats[u], K) for u in cold}, cold),
}
cols = [f"P@{K}", f"R@{K}", f"NDCG@{K}", "coverage", "diversity", "cat_entropy"]
def table(d):
    h = "| System | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    return h + "\n".join(f"| {n} | " + " | ".join(f"{v[c]:.4f}" for c in cols) + " |" for n, v in d.items())

md = (f"# MovieLens results\nUsers: {ds.n_users} ({len(warm)} warm, {len(cold)} cold) | Items: {ds.n_items} | "
      f"Train pairs: {len(ds.base)}\nSplit: per-user temporal (oldest 70% train, 10% ranker, newest 20% test)\n\n"
      f"## Warm users (K={K})\n{table(rows)}\n\n## Cold-start users\n{table(cold_rows)}\n")
open("results_movielens.md", "w", encoding="utf-8").write(md)
print(md)
