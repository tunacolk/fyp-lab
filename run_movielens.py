"""Run the pipeline on MovieLens:  python run_movielens.py data/raw/ml-latest-small"""
import sys
from fyp.data_movielens import load_movielens
from fyp.pipeline import RecSys
from fyp.metrics import evaluate, ndcg_at_k, intra_list_diversity
from fyp.significance import per_user, paired_bootstrap, bootstrap_ci
import numpy as np

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
# ---- significance: per-user paired bootstrap (users resampled, 5000 draws)
wt = {int(u): truth[int(u)] for u in warm}
nd = {n: per_user(r, wt, ndcg_at_k, warm, K) for n, r in systems.items()}
div = {n: np.array([intra_list_diversity(r[int(u)][:K], rs.E) for u in warm]) for n, r in systems.items()}
ci_lines = []
for n, v in nd.items():
    m, (lo, hi) = bootstrap_ci(v)
    ci_lines.append(f"| {n} | {m:.4f} | [{lo:.4f}, {hi:.4f}] |")
pairs = [("Collaborative (SVD)", "Popularity", nd), ("Two-stage ranker", "Collaborative (SVD)", nd),
         ("Ranker + MMR (0.7)", "Two-stage ranker", nd), ("Ranker + MMR (0.7)", "Two-stage ranker", div)]
cmp_lines = []
for new, old, src in pairs:
    r = paired_bootstrap(src[old], src[new])
    metric = "NDCG@10" if src is nd else "diversity"
    verdict = "significant" if r["ci95"][0] > 0 or r["ci95"][1] < 0 else "not significant"
    cmp_lines.append(f"| {new} vs {old} | {metric} | {r['diff']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['p']:.4f} | {verdict} |")
md += ("\n## NDCG@10 with 95% bootstrap CI (warm users)\n| System | mean | 95% CI |\n|---|---|---|\n" + "\n".join(ci_lines)
       + "\n\n## Paired comparisons (new minus old, same users)\n| Comparison | Metric | Diff | 95% CI | p | Verdict |\n|---|---|---|---|---|---|\n"
       + "\n".join(cmp_lines)
       + "\n\nCaveat: 4 comparisons on one split; no multiple-testing correction. Treat p-values near 0.05 as weak evidence.\n")
open("results_movielens.md", "w", encoding="utf-8").write(md)
print(md)
