# Week 7 — Data Structures Comparison Report

**Source Benchmarks**: `results/structures_comparison.csv`  
**Evaluated Structures**: `HashTableChaining`, `HashTableLinearProbing`, Python `dict`, `BST` (Binary Search Tree), `SkipList`

---

## 1. Empirical Latency Benchmark Summary

Average lookup latency in microseconds ($\mu s$) across dataset sizes:

| Dataset Size (Keys) | HashTable (Chaining) | HashTable (Probing) | Python dict (C-Hash) | BST (Unbalanced) | Skip List |
|---|---|---|---|---|---|
| **10** | 0.42 $\mu s$ | 0.38 $\mu s$ | 0.08 $\mu s$ | 0.65 $\mu s$ | 0.95 $\mu s$ |
| **100** | 0.45 $\mu s$ | 0.41 $\mu s$ | 0.09 $\mu s$ | 1.12 $\mu s$ | 1.42 $\mu s$ |
| **500** | 0.48 $\mu s$ | 0.46 $\mu s$ | 0.09 $\mu s$ | 1.85 $\mu s$ | 2.15 $\mu s$ |
| **1000** | 0.51 $\mu s$ | 0.49 $\mu s$ | 0.10 $\mu s$ | 2.45 $\mu s$ | 2.80 $\mu s$ |
| **5000** | 0.55 $\mu s$ | 0.53 $\mu s$ | 0.11 $\mu s$ | 4.10 $\mu s$ | 4.25 $\mu s$ |

---

## 2. Qualitative & Trade-off Analysis

### Average vs Worst-Case Time Complexity
- **Hash Tables (Chaining & Probing)**: $O(1)$ average search, insertion, and deletion. Worst-case $O(N)$ when all keys hash to a single index.
- **Python dict**: Optimized C-level open addressing hash table ($O(1)$ average).
- **BST**: $O(\log N)$ average search/insert for balanced trees, but degenerates to $O(N)$ for sequential insertion sequences. Supports ordered key iteration.
- **Skip List**: Probabilistic $O(\log N)$ average search/insert/delete using multi-level pointer forward links.

### Memory & Cache Locality
- **Linear Probing**: High CPU cache locality due to contiguous array memory, but susceptible to primary clustering at high load factors ($\ge 0.75$).
- **Separate Chaining**: Flexible bucket pointer lists without table fill limits, but higher pointer memory overhead.
- **BST / Skip List**: Node pointer overhead per element ($2-16$ pointers per node), lower cache locality.

### Relevance to Recommender System
Hash Tables provide optimal $O(1)$ user profile lookup by user ID during similarity computations. Ordered structures (BST/Skip List) are unnecessary for unordered integer user ID key lookups, confirming Hash Tables as the ideal user lookup layer.
