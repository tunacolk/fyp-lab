"""Run the whole experiment and write results.md"""
import json, time
import numpy as np
from fyp.data import generate
from fyp.pipeline import RecSys
from fyp.metrics import evaluate
from fyp.abtest import run_ab, aa_false_positive_rate
from fyp.safety import evaluate_guardrail

K = 10
ds = generate()
rs = RecSys(ds).fit()
warm = ds.warm_users
truth = {}
for u, i in ds.test:
    truth.setdefault(int(u), set()).add(int(i))
wt = {int(u): truth[int(u)] for u in warm}

def ev(recs):
    return evaluate(recs, wt, rs.E, ds.item_cat, ds.n_items, K)

systems = {
    "Popularity": rs.recommend_scores(warm, "pop", K),
    "Content-based": rs.recommend_scores(warm, "content", K),
    "Collaborative (SVD)": rs.recommend_scores(warm, "mf", K),
    "Two-stage ranker": rs.recommend_ranked(warm, K),
    "Ranker + MMR (λ=0.7)": rs.recommend_ranked(warm, K, mmr_lambda=0.7),
}
t0 = time.perf_counter(); rs.recommend_ranked(warm[:200], K, mmr_lambda=0.7)
latency_ms = (time.perf_counter() - t0) / 200 * 1000
rows = {name: ev(r) for name, r in systems.items()}

# MMR lambda sweep: accuracy vs diversity trade-off
sweep = {f"lambda={l}": ev(rs.recommend_ranked(warm, K, mmr_lambda=l)) for l in (0.9, 0.7, 0.5, 0.3)}

# cold start
cold = ds.cold_users
ct = {int(u): truth[int(u)] for u in cold}
cold_pop = {int(u): rs.recommend_cold(ds.fav_cats[u], K, use_onboarding=False) for u in cold}
cold_onb = {int(u): rs.recommend_cold(ds.fav_cats[u], K, explore=2) for u in cold}
cold_rows = {"Global popularity": evaluate(cold_pop, ct, rs.E, ds.item_cat, ds.n_items, K),
             "Onboarding + explore": evaluate(cold_onb, ct, rs.E, ds.item_cat, ds.n_items, K)}

# A/B tests: control = current CF system, treatment = ranker + MMR
ab = run_ab(ds, systems["Collaborative (SVD)"], systems["Ranker + MMR (λ=0.7)"], seed=3)
ab_pop = run_ab(ds, systems["Popularity"], systems["Collaborative (SVD)"], seed=3)
aa = aa_false_positive_rate(ds, systems["Collaborative (SVD)"], n_sims=200)
guard = evaluate_guardrail()

def table(d, cols):
    h = "| System | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    return h + "\n".join(f"| {n} | " + " | ".join(f"{v[c]:.4f}" for c in cols) + " |" for n, v in d.items())

cols = [f"P@{K}", f"R@{K}", f"NDCG@{K}", "coverage", "diversity", "cat_entropy"]
def abrow(r):
    p, g = r["primary_ctr"], r["guardrail_categories"]
    return (f"CTR {p['control']:.4f} -> {p['treatment']:.4f} (lift {p['rel_lift']*100:+.1f}%, "
            f"95% CI diff [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}], p={p['p_value']:.4f}); "
            f"guardrail categories/list {g['control']:.2f} -> {g['treatment']:.2f} (p={g['p_value']:.4f}); "
            f"n={r['n_control']}/{r['n_treatment']}")

md = f"""# FYP Lab results
Users: {ds.n_users} ({len(warm)} warm, {len(cold)} cold) | Items: {ds.n_items} | Train pairs: {len(ds.base)}

## Offline ranking (warm users, K={K})
{table(rows, cols)}

Serving latency (ranker + MMR, single user, CPU): {latency_ms:.2f} ms

## MMR lambda sweep (lower lambda = more diversity)\n{table(sweep, cols)}\n\n## Cold start ({len(cold)} users, zero history)
{table(cold_rows, cols)}

## Offline A/B simulation
- Test 1, control=SVD vs treatment=Ranker+MMR: {abrow(ab)}
- Test 2, control=Popularity vs treatment=SVD: {abrow(ab_pop)}
- A/A false-positive rate (200 sims, alpha=0.05): {aa:.3f}
- Users needed per arm to detect an absolute CTR change of {ab['mde_abs']} (80% power): {ab['required_n_per_arm']}

## Guardrail (rule-based baseline, {guard['n']} labelled examples)
precision={guard['precision']:.2f} recall={guard['recall']:.2f}  (TP={guard['tp']} FP={guard['fp']} FN={guard['fn']} TN={guard['tn']})
Errors: {guard['errors']}
Note: rules and eval set written by the same person, so these numbers are optimistic; use an independent set before trusting them.
"""
open("results.md", "w", encoding="utf-8").write(md)
print(md)
