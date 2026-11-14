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

## NDCG@10 with 95% bootstrap CI (warm users)
| System | mean | 95% CI |
|---|---|---|
| Popularity | 0.0655 | [0.0551, 0.0764] |
| Content-based | 0.0089 | [0.0059, 0.0123] |
| Collaborative (SVD) | 0.0906 | [0.0793, 0.1027] |
| Two-stage ranker | 0.0886 | [0.0772, 0.1006] |
| Ranker + MMR (0.7) | 0.0884 | [0.0771, 0.1004] |

## Paired comparisons (new minus old, same users)
| Comparison | Metric | Diff | 95% CI | p | Verdict |
|---|---|---|---|---|---|
| Collaborative (SVD) vs Popularity | NDCG@10 | +0.0251 | [+0.0133, +0.0372] | 0.0000 | significant |
| Two-stage ranker vs Collaborative (SVD) | NDCG@10 | -0.0020 | [-0.0102, +0.0064] | 0.6444 | not significant |
| Ranker + MMR (0.7) vs Two-stage ranker | NDCG@10 | -0.0003 | [-0.0032, +0.0027] | 0.8252 | not significant |
| Ranker + MMR (0.7) vs Two-stage ranker | diversity | +0.0660 | [+0.0611, +0.0710] | 0.0000 | significant |

Caveat: 4 comparisons on one split; no multiple-testing correction. Treat p-values near 0.05 as weak evidence.
