# Week 7 — Gap Analysis Report

**Project**: Hash Table Recommender System (`DOE-hash-tables`)  
**Scope**: Technical Gap Evaluation, System Requirements & AI/DS Solution Strategy

---

## 1. Formal Gap Analysis Table

| # | What We Have | What We Need | The Gap | Our Plan | AI/DS Solution | Status |
|---|---|---|---|---|---|---|
| 1 | Baseline random movie selection | Meaningful recommendation evaluation | No latent preference structure in uniform random data | Implement structured synthetic generator with genre preferences | Zipf-like genre preference model (`generate_structured_data`) | Done |
| 2 | Unchecked cold-start users | Deterministic cold-start handling | Unknown user IDs yield empty recommendations | Implement cold-start hybrid fallback engine | Hybrid popularity & seed-vector cosine fallback (`recommend_movies_hybrid`) | Done |
| 3 | Raw cosine similarity | IDF-weighted similarity with popularity damping | Popular movies dominate recommendations | Implement smoothed IDF weighting and popularity exponent damping | IDF-weighted cosine score with $\alpha$-damping (`recommend_movies_weighted_cosine`) | Done |
| 4 | Step-by-step trace logging | Trace-free performance benchmarking | Operation trace dicts distort execution latency measurements | Add optional `record_trace=False` parameter to core data structures | High-precision timing without memory trace allocation | Done |
| 5 | Full dense array calculations | Memory-efficient sparse matrix representations | Dense $U \times M$ matrices consume massive memory for large domains | Build CSR matrix recommender with `scipy.sparse.csr_matrix` | Sparse matrix vectorization (`SparseRecommender`) | Done |
| 6 | Fixed table size resizing | Dynamic bitwise bucket splitting | Full-table rehashing causes temporary latency spikes during resize | Implement Extendible Hashing with global/local depth tracking | Dynamic Extendible Hash Table (`ExtendibleHashTable`) | Done |
| 7 | Hand-rolled benchmark loops | End-to-end data pipeline & CSV export | Missing standardized ETL pipeline for batch processing | Build Pandas + NumPy pipeline with CLI integration | Modular Pandas/NumPy ETL pipeline (`pipeline.py`) | Done |
| 8 | Manual verification script | Automated unit & integration testing suite | Need comprehensive test coverage for all features | Implement `pytest` test suite in `tests/` | Pytest suite covering >26 test scenarios | Done |
| 9 | Basic GUI layout | Modern multi-tab interactive UI | GUI lacks dedicated recommender lab and chart embedding | Redesign CustomTkinter GUI with dark theme, tabs, and real-time cards | Multi-tab GUI (`app.py`) | Done |
| 10 | Scattered CLI commands | Unified application entry point | Commands split across scripts | Build central CLI dispatcher with `argparse` | Single entry point (`main.py`) | Done |
| 11 | Qualitatively guessed accuracy | Automated 1–10 quality rubric | No empirical quantitative score calculation | Create rubric script parsing experiment CSV files | Heuristic scoring rubric (`score_rubric.py`) | Done |
| 12 | Basic Hash Table lookup | Multi-data-structure comparative benchmark | Need empirical evidence comparing Hash Tables against BST and Skip List | Implement BST and Skip List for latency and memory benchmarks | Data structures comparison (`structures_compare.py`) | Done |
| 13 | Theoretical Big-O claims | Empirical log-log scaling analysis | Theoretical complexity unverified empirically | Fit empirical log-log slopes via `np.polyfit` | Scaling benchmark experiment (`run_scaling_experiment`) | Done |
| 14 | Raw code repository | Complete Week 7 & Week 8 documentation | Missing structured analysis reports, pitch, and specifications | Author docs under `docs/week7/` and `docs/week8/` | Comprehensive markdown documentation suite | Done |
