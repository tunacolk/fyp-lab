# MovieLens results
Users: 579 (522 warm, 57 cold) | Items: 6296 | Train pairs: 30806
Split: per-user temporal (oldest 70% train, 10% ranker, newest 20% test)

## Warm users (K=10)
| System | P@10 | R@10 | NDCG@10 | coverage | diversity | cat_entropy |
|---|---|---|---|---|---|---|
| Popularity | 0.0494 | 0.0448 | 0.0655 | 0.0086 | 0.7018 | 1.5223 |
| Content-based | 0.0067 | 0.0067 | 0.0089 | 0.1682 | 0.1565 | 0.3961 |
| Collaborative (SVD) | 0.0699 | 0.0697 | 0.0906 | 0.0670 | 0.7477 | 1.9306 |
| Two-stage ranker | 0.0699 | 0.0691 | 0.0886 | 0.0718 | 0.6738 | 1.7189 |
| Ranker + MMR (0.7) | 0.0707 | 0.0675 | 0.0884 | 0.0742 | 0.7398 | 1.9131 |

## Cold-start users
| System | P@10 | R@10 | NDCG@10 | coverage | diversity | cat_entropy |
|---|---|---|---|---|---|---|
| Global popularity | 0.3386 | 0.0747 | 0.3581 | 0.0016 | 0.7136 | 1.4855 |
| Onboarding + explore | 0.3421 | 0.0825 | 0.3773 | 0.0038 | 0.7437 | 1.6482 |
