"""Offline A/B test simulation: assignment, click simulation, Welch t-test, CI, sample size, A/A check."""
import numpy as np
from scipy import stats


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate_user_ctr(ds, recs: dict, rng, offset=3.5, pos_decay=0.12):
    """Click ~ Bernoulli(sigmoid(true_logit - offset) * position_exam). Returns per-user CTR and category count."""
    ctr, ncat = {}, {}
    for u, items in recs.items():
        items = np.array(items)
        logit = ds.true_logit(np.full(len(items), u), items)
        exam = 1.0 / (1.0 + pos_decay * np.arange(len(items)))
        clicks = rng.random(len(items)) < sigmoid(logit - offset) * exam
        ctr[u] = clicks.mean()
        ncat[u] = len(set(ds.item_cat[items]))
    return ctr, ncat


def welch(a, b, alpha=0.05):
    a, b = np.asarray(a), np.asarray(b)
    t, p = stats.ttest_ind(b, a, equal_var=False)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"control": a.mean(), "treatment": b.mean(), "abs_diff": diff,
            "rel_lift": diff / a.mean() if a.mean() else float("nan"),
            "ci95": (diff - z * se, diff + z * se), "p_value": float(p)}


def sample_size_per_arm(std, mde_abs, alpha=0.05, power=0.8):
    """n per arm to detect an absolute difference `mde_abs` of a mean metric."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (za + zb) ** 2 * std ** 2 / mde_abs ** 2))


def run_ab(ds, recs_control, recs_treatment, seed=0, alpha=0.05, mde_abs=0.02):
    """Randomise users 50/50, simulate clicks on the arm each user is assigned to."""
    rng = np.random.default_rng(seed)
    users = np.array(sorted(recs_control))
    arm = rng.random(len(users)) < 0.5
    ctrl_users, treat_users = users[~arm], users[arm]
    c_ctr, c_cat = simulate_user_ctr(ds, {u: recs_control[u] for u in ctrl_users}, rng)
    t_ctr, t_cat = simulate_user_ctr(ds, {u: recs_treatment[u] for u in treat_users}, rng)
    primary = welch(list(c_ctr.values()), list(t_ctr.values()), alpha)
    guardrail = welch(list(c_cat.values()), list(t_cat.values()), alpha)   # categories per list
    return {"n_control": len(ctrl_users), "n_treatment": len(treat_users),
            "primary_ctr": primary, "guardrail_categories": guardrail,
            "required_n_per_arm": sample_size_per_arm(np.std(list(c_ctr.values()), ddof=1), mde_abs),
            "mde_abs": mde_abs}


def aa_false_positive_rate(ds, recs, n_sims=300, alpha=0.05, seed=1):
    """Sanity check: both arms get the SAME system; ~alpha of tests should still 'win' by chance."""
    rng = np.random.default_rng(seed)
    users = np.array(sorted(recs))
    hits = 0
    for _ in range(n_sims):
        arm = rng.random(len(users)) < 0.5
        a, _ = simulate_user_ctr(ds, {u: recs[u] for u in users[~arm]}, rng)
        b, _ = simulate_user_ctr(ds, {u: recs[u] for u in users[arm]}, rng)
        hits += welch(list(a.values()), list(b.values()), alpha)["p_value"] < alpha
    return hits / n_sims
