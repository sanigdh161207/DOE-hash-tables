# Week 7 — Analysis of Two Biggest Logical Flaws

**Project**: Hash Table Recommender System (`DOE-hash-tables`)

---

## Flaw 1: Cold-Start Vulnerability for Unknown User IDs

### Code & Experimental Evidence
In the baseline implementation of `RecommenderSystem.recommend_movies`:
```python
user_movies, trace = self.get_user_preferences(user_id)
if not user_movies:
    return [], trace
```
When an un-indexed user ID is passed, the hash table search returns `None`, resulting in an empty recommendation list `[]` (`Hit-Rate@5 = 0.0`).

### Consequence
New or anonymous users receive no recommendations, degrading user engagement and product experience.

### AI / Data Structure Solution
Implement a **hybrid multi-stage cold-start engine** (`recommend_movies_hybrid`):
1. **Scenario A (No seed items)**: Fall back to globally popular unseen items ranked by interaction count.
2. **Scenario B (Transient seed items)**: Compute a temporary preference vector from seed selections and calculate IDF-weighted cosine similarity against indexed users without writing to the main hash table.
3. **Scenario C (Known user with sparse recommendations)**: Fill remaining top-N positions with globally popular items.

### Measured Before / After Results (Source: `results/cold_start.csv`)
- **Before (Unchecked Cold-Start)**: Hit-Rate@5 = **0.0000** for unknown users.
- **After (Popularity Fallback)**: Hit-Rate@5 = **0.6120** for unknown users.
- **After (Seed-Based Cosine, 2 seed items)**: Hit-Rate@5 = **0.7480** for unknown users.

---

## Flaw 2: Popularity Bias & Co-Occurrence Dominance

### Code & Experimental Evidence
Standard cosine similarity treats all item matches equally regardless of item frequency:
$$\text{sim}(a, b) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$
Under uniform weighting, ubiquitous items present in 80% of user profiles produce high cosine similarity across unrelated users, causing top recommendations to be dominated by the top 10% most popular items (`top_10_share = 0.4210`).

### Consequence
Recommendations lack personalization and diversity, creating filter bubbles and ignoring long-tail catalog items.

### AI / Data Structure Solution
Implement **Inverse Document Frequency (IDF) Weighting & Popularity Damping**:
1. **Smoothed IDF Weighting**: $IDF(i) = \log\left(\frac{1 + U}{1 + df_i}\right) + 1$, down-weighting ubiquitous items during cosine calculation.
2. **Popularity Exponent Damping**: $\text{score}_{\text{damped}}(m) = \frac{\text{score}(m)}{\text{popularity}(m)^\alpha}$ ($\alpha = 0.5$).

### Measured Before / After Results (Source: `results/weighted_vs_plain_cosine.csv`)
- **Before (Plain Cosine)**: Catalog Coverage = **0.8200**, Top 10 Share = **0.4210**, Diversity = **0.6120**.
- **After (IDF + Popularity Damping)**: Catalog Coverage = **0.9400**, Top 10 Share = **0.1830**, Diversity = **0.7840**.
