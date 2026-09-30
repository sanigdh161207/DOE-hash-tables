# Week 7 — Heuristic Recommendation Quality Evaluation Report

**Source Data**: `results/quality_baseline_vs_cosine.csv`  
**Rubric Evaluator**: `score_rubric.py`

---

## 1. Transparent 1–10 Quality Rubric

The system calculates an automated 1–10 heuristic quality score based on actual empirical metrics measured during the leave-one-out evaluation protocol across 500 structured synthetic user profiles.

### Metric Weights & Target Normalization Bounds

| Metric Dimension | Weight | Target Upper Bound (10/10) | Formula / Calculation |
|---|---|---|---|
| **Precision@5** | 25% | 0.40 | $\min\left(10, \frac{\text{Measured P@5}}{0.40} \times 10\right)$ |
| **Hit-Rate@5** | 20% | 0.70 | $\min\left(10, \frac{\text{Measured HR@5}}{0.70} \times 10\right)$ |
| **Catalog Coverage** | 20% | 0.80 | $\min\left(10, \frac{\text{Measured Cov}}{0.80} \times 10\right)$ |
| **Intra-List Diversity** | 15% | 0.80 | $\min\left(10, \frac{\text{Measured Div}}{0.80} \times 10\right)$ |
| **Recommendation Latency** | 20% | $\le 0.001$ sec (1 ms) | 10 pts if $\le 1$ ms; 8.5 pts if $\le 5$ ms; 7 pts if $\le 10$ ms |

---

## 2. Concrete Measured Observations

All values are drawn directly from empirical execution saved under `results/`:

1. **Observation 1 (Recommendation Quality & Hit-Rate)**:
   - *Metric*: Precision@5 & Hit-Rate@5
   - *Measured Values*: Cosine Recommender achieved Hit-Rate@5 of **0.7840** and Precision@5 of **0.1568**, compared to Random Baseline Hit-Rate@5 of **0.2360** and Precision@5 of **0.0472**.
   - *Interpretation*: User-based collaborative filtering using cosine similarity achieves over **3.3x improvement** in hit-rate over random guessing by capturing latent genre co-occurrence.

2. **Observation 2 (Catalog Coverage vs Popularity Bias)**:
   - *Metric*: Catalog Coverage & Top 10 Share
   - *Measured Values*: Plain Cosine achieved **0.8200** catalog coverage, while IDF-weighted cosine with popularity damping increased catalog coverage to **0.9400** while reducing top-10 popular movie share from **0.4210** to **0.1830**.
   - *Interpretation*: IDF weighting and popularity exponent damping successfully prevent popular items from monopolizing recommendations, distributing recommendations across niche catalog items.

3. **Observation 3 (User Lookup Latency & Hash Table Overhead)**:
   - *Metric*: Average Recommendation Latency
   - *Measured Values*: Average recommendation query latency is **0.000342 seconds** (342 microseconds) for the hash-table backed cosine recommender.
   - *Interpretation*: The $O(1)$ expected hash table user preference lookup layer keeps total query latency well below the 1 ms interactive threshold.
