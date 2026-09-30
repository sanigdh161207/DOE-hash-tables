# Week 8 — Implemented Extensions Before vs. After Report

**Source Data**: `results/quality_baseline_vs_cosine.csv`, `results/weighted_vs_plain_cosine.csv`, `results/sparse_experiment.csv`, `results/extendible_vs_rehash.csv`

---

## 1. Extension 1: IDF-Weighted Cosine & Popularity Damping

- **What Changed**: Added smoothed IDF weighting ($IDF(i) = \log\left(\frac{1 + U}{1 + df_i}\right) + 1$) and popularity exponent damping ($\text{score} / \text{pop}^{0.5}$) in `recommend_movies_weighted_cosine`.
- **Why**: Plain cosine similarity recommended top blockbuster movies to all users regardless of preference.
- **Before (Plain Cosine)**: Catalog Coverage = **0.8200**, Top 10 Popular Share = **0.4210**, Diversity = **0.6120**.
- **After (IDF-Weighted + Damping)**: Catalog Coverage = **0.9400**, Top 10 Popular Share = **0.1830**, Diversity = **0.7840**.
- **Interpretation**: Popularity damping reduced popular item concentration by **56.5%** while increasing catalog coverage by **14.6%**.

---

## 2. Extension 2: Cold-Start Hybrid Fallback Engine

- **What Changed**: Implemented multi-stage hybrid engine (`recommend_movies_hybrid`) routing un-indexed users to popularity fallbacks or transient seed-vector matching.
- **Why**: Un-indexed users received empty recommendation lists (`[]`).
- **Before (Unchecked Cold-Start)**: Hit-Rate@5 = **0.0000**.
- **After (Popularity Fallback)**: Hit-Rate@5 = **0.6120**.
- **After (Seed-Based Cosine, 2 seed items)**: Hit-Rate@5 = **0.7480**.
- **Interpretation**: Prevents user drop-off for new/anonymous visitors.

---

## 3. Extension 3: Sparse CSR Matrix Representation (`SparseRecommender`)

- **What Changed**: Implemented `scipy.sparse.csr_matrix` row-normalized dot product vectorization with `HashTableChaining` mapped from `user_id -> row_index`.
- **Why**: Dense 2D float64 NumPy matrices consume excessive memory for large user/item scales.
- **Before (Dense Float64 Matrix, N=1,000, M=1,000)**: Memory Footprint = **8.0 MB** (745.06 GiB at 1M x 100k scale).
- **After (Sparse CSR Matrix, N=1,000, M=1,000)**: Memory Footprint = **0.12 MB** (124 MB at 1M x 100k scale, 0.01% density).
- **Interpretation**: Achieved **98.5% memory reduction** for 1,000 users / 1,000 items while maintaining mathematically identical recommendation outputs.

---

## 4. Extension 4: Extendible Hashing (`ExtendibleHashTable`)

- **What Changed**: Implemented bitwise directory depth expansion and bucket splitting without full-table rehashing.
- **Why**: Standard chaining/probing requires re-allocating and re-hashing all keys when load factor exceeds threshold.
- **Before (Full Table Rehashing, N=5,000)**: Rehash Entries Moved = **10,000** total keys moved during array resizes.
- **After (Extendible Hashing, N=5,000)**: Extendible Entries Moved = **1,248** total keys moved during incremental bucket splits.
- **Interpretation**: Reduced entry movement during table growth by **87.5%**, eliminating resize latency spikes.
