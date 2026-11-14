# FYP Lab: mini "For You" feed, A/B testing and safe-chat guardrail

A small end-to-end project that mirrors the work of a recommendation + AI-chat team:
two-stage recommender (candidate generation -> learned ranker -> MMR diversity re-rank),
cold-start handling, offline A/B test simulation, event logging API and a layered LLM guardrail.

## Run
```bash
pip install -r requirements.txt
python run_all.py        # trains everything, prints + writes results.md
python run_tests.py      # 8 tests (pytest-compatible: `pytest tests/`)
uvicorn fyp.api:app --reload   # /recommend/{user_id}, /event, /chat, /health
```

## What is inside
| Module | What it shows |
|---|---|
| `fyp/data.py` | Synthetic users/items with known ground-truth preferences (swap in MovieLens) |
| `fyp/pipeline.py` | SVD collaborative + TF-IDF/SVD content candidates, HistGradientBoosting ranker, MMR, cold start |
| `fyp/metrics.py` | P@K, Recall@K, NDCG@K, catalog coverage, intra-list diversity, category entropy |
| `fyp/abtest.py` | Randomisation, click simulation, Welch t-test, 95% CI, guardrail metric, sample-size calc, A/A check |
| `fyp/safety.py` | Input/output moderation, prompt-injection rules, PII masking, labelled eval set with P/R |
| `fyp/api.py` | FastAPI service with deterministic A/B bucketing and structured impression/click logs |

## Results (see results.md)
- Two-stage ranker: NDCG@10 0.134 vs 0.121 for SVD, 0.044 for popularity.
- MMR lambda sweep shows the accuracy/diversity trade-off explicitly.
- Cold start: onboarding + exploration P@10 0.231 vs 0.153 for global popularity.
- A/B: Popularity -> SVD is a significant win; SVD -> Ranker+MMR is NOT significant (CTR diff CI includes 0)
  and the diversity guardrail got worse, so the honest call is "do not ship yet, tune lambda / run longer".
- A/A test false-positive rate ~5% (as expected for alpha=0.05).
- Guardrail baseline: precision 0.92, recall 0.71 on a 35-example set; errors are listed, not hidden.

## Limitations (be upfront about these)
- Data is synthetic, so absolute numbers are not comparable to real platforms.
- Click simulator uses the generator's ground truth; real A/B tests need real traffic, novelty-effect checks and SRM checks.
- Content embeddings are TF-IDF+SVD; production would use Sentence-Transformers.
- Guardrail rules and eval set were written by the same person, so metrics are optimistic. Next step: ML/LLM moderation behind `moderate()`.

## Next steps
Real dataset (MovieLens), Sentence-Transformers embeddings, FAISS retrieval, LLM-as-judge for chat quality, Next.js demo UI.
