# Weighted Cosine Similarity Benchmark Summary

## Summary Table
| Strategy | Precision@5 | Hit-Rate@5 | NDCG@5 | Coverage | Diversity | Top 10 Share | Latency (sec) |
|---|---|---|---|---|---|---|---|
| Plain Cosine | 0.0460 | 0.2300 | 0.1728 | 1.0000 | 0.6891 | 0.4279 | 0.012526 |
| IDF-Weighted Cosine | 0.0472 | 0.2360 | 0.1720 | 0.9900 | 0.6974 | 0.4700 | 0.026863 |
| IDF + Popularity Damping | 0.0332 | 0.1660 | 0.1086 | 1.0000 | 0.5922 | 0.1664 | 0.028536 |

## Concrete Test User Case Study (User ID: 1912)
- **Plain Cosine Recommendations**: [4, 67, 10, 11, 26]
- **IDF-Weighted Damped Recommendations**: [67, 4, 84, 51, 70]

### Rationale & Interpretation
IDF weighting down-weights globally ubiquitous movies that frequently co-occur with many users.
Popularity damping ($lpha=0.5$) penalizes dominant blockbusters, boosting novel niche recommendations and catalog coverage.