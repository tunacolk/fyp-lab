# FYP Lab

![CI](https://github.com/tunacolk/fyp-lab/actions/workflows/ci.yml/badge.svg)

A small end-to-end project that mirrors the work of a recommendation + AI-chat team: a two-stage "For You" recommender, cold-start handling, statistically tested evaluation on MovieLens, an offline A/B test simulation, and a layered guardrail for LLM chat, all served from a Dockerised FastAPI app with CI.

**Why it exists:** to practice measuring recommendation changes honestly. Several results below are *null results*, and they are reported as such.

## Key results (MovieLens `ml-latest-small`)

Temporal per-user split (oldest 70% train, next 10% ranker labels, newest 20% test). 579 users (522 warm, 57 cold-start), 6,296 items, "liked" = rating >= 4.

| System | NDCG@10 | 95% bootstrap CI | Catalog coverage | Diversity |
|---|---|---|---|---|
| Popularity | 0.0655 | [0.0551, 0.0764] | 0.9% | 0.702 |
| Content-based (TF-IDF) | 0.0089 | [0.0059, 0.0123] | 16.8% | 0.157 |
| Collaborative (SVD) | 0.0906 | [0.0793, 0.1027] | 6.7% | 0.748 |
| Two-stage ranker | 0.0886 | [0.0772, 0.1006] | 7.2% | 0.674 |
| Ranker + MMR (lambda=0.7) | 0.0884 | [0.0771, 0.1004] | 7.4% | 0.740 |

Paired bootstrap over users (5,000 draws), new minus old:

| Comparison | Metric | Diff | 95% CI | Verdict |
|---|---|---|---|---|
| SVD vs Popularity | NDCG@10 | +0.0251 | [+0.0133, +0.0372] | significant |
| Ranker vs SVD | NDCG@10 | -0.0020 | [-0.0102, +0.0064] | not significant |
| Ranker+MMR vs Ranker | NDCG@10 | -0.0003 | [-0.0032, +0.0027] | not significant |
| Ranker+MMR vs Ranker | Diversity | +0.0660 | [+0.0611, +0.0710] | significant |

**What this says**
- Collaborative filtering clearly beats popularity and reaches much further into the catalog.
- MMR re-ranking raises list diversity by roughly 10% with no detectable NDCG loss.
- The gradient-boosted ranker showed **no significant gain** over SVD with the current features. The data rules out differences larger than about 0.01 NDCG but cannot resolve smaller ones.
- Cold-start (onboarding genres + exploration slots) scored NDCG 0.377 vs 0.358 for global popularity. This is a small gain on only 57 users and was **not** statistically tested.

## What is inside

| Module | Purpose |
|---|---|
| `fyp/data.py` | Synthetic users/items with known ground-truth preferences (for A/B simulation) |
| `fyp/data_movielens.py` | MovieLens loader producing the same `Dataset` object |
| `fyp/pipeline.py` | SVD + content candidates, HistGradientBoosting ranker, MMR re-ranking, cold start |
| `fyp/metrics.py` | P@K, Recall@K, NDCG@K, coverage, intra-list diversity, category entropy |
| `fyp/significance.py` | Per-user bootstrap CIs and paired bootstrap tests |
| `fyp/abtest.py` | Randomisation, click simulation, Welch t-test, guardrail metric, sample size, A/A check |
| `fyp/safety.py` | Input/output moderation, prompt-injection rules, PII masking, labelled eval set |
| `fyp/api.py` | FastAPI: `/recommend`, `/event`, `/chat`, `/health` with deterministic A/B bucketing and JSON logs |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt pytest

pytest tests/                     # unit tests
python run_all.py                 # synthetic data: offline metrics + A/B simulation + guardrail eval
python run_movielens.py           # real data (see below)
```

**MovieLens data:** download `ml-latest-small` from grouplens.org and place it so that `data/raw/ml-latest-small/ratings.csv` exists. The folder is git-ignored.

**Docker:**

```bash
docker compose up --build
# open http://localhost:8000/docs
```

Try `GET /recommend/5`, then `/recommend/1900` (a cold-start user), and `POST /chat` with `{"message": "Ignore all previous instructions"}`.

## Synthetic-data experiments

Because the generator knows each user's true preferences, it supports an honest offline A/B simulation:

- Ranker beats SVD offline (NDCG@10 0.134 vs 0.121), but in the simulated A/B test the CTR difference is **not significant** (p=0.78) and the category-diversity guardrail got worse, so the call is "do not ship yet".
- A/A test false-positive rate is about 5%, as expected for alpha = 0.05.
- Guardrail baseline (35 labelled examples): precision 0.92, recall 0.71. Errors are listed in `results.md`, not hidden.

## Limitations

- Synthetic-data numbers are not comparable to real platforms; the click model is the generator's own ground truth.
- MovieLens has no preference model, so A/B simulation is not run there. Real A/B tests also need real traffic, novelty-effect checks and sample-ratio checks.
- Content embeddings are TF-IDF + SVD on titles and genres. This is weak (NDCG 0.0089).
- Cold-start "onboarding" genres are derived from each user's own history, so that result is optimistic.
- Four paired comparisons on one split, no multiple-testing correction.
- Guardrail rules and its eval set were written by the same person, so those metrics are optimistic. A production system would put an ML/LLM moderation model behind `moderate()`.

## Next steps

1. Richer item features (`tags.csv`, release year, genre one-hot) and Sentence-Transformers embeddings, measured with the bootstrap tests.
2. Ranking-aware loss (LambdaRank) for the ranker.
3. Real LLM call in `/chat` plus LLM-as-judge for response quality.
4. Experiment tracking (MLflow / W&B).
