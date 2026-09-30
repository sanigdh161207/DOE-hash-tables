# Baseline System Audit Report

**Date**: 2026-10-01
**Repository**: DOE Hash Table Recommender System (`DOE-hash-tables`)

---

## 1. Baseline System Architecture

The existing project implements a hash-table-backed recommender system built with pure Python data structures and visualised using CustomTkinter and Matplotlib.

### Core Modules

1. **`recommender.py`**:
   - `AbstractHashTable`: Base class for hash table implementations defining `hash_function`, `insert`, `search`, `delete`, `resize`, `load_factor`, `collision_count`, `get_collision_statistics`, and `estimate_memory_bytes`.
   - `HashTableChaining`: Separate chaining collision resolution using nested Python lists `List[List[Tuple[Any, Any]]]`.
   - `HashTableLinearProbing`: Open addressing with linear probing, tombstone deletion (`"<TOMBSTONE>"`), and dynamic auto-resizing when load factor reaches $\ge 0.95$.
   - `HashTable`: Backward compatibility alias for `HashTableChaining`.
   - `RecommenderSystem`: Encapsulates a hash table mapping `user_id -> List[movie_id]`. Baseline recommendation samples unseen movies using `random.seed(user_id)`.

2. **`data_generator.py`**:
   - `data_generator(n_users, n_items)` / `generate_user_data(num_users, num_movies)`: Generates uniform synthetic user interaction data (3–10 movies per user from a pool of `num_movies`). Seeded with `random.seed(42)`.
   - `format_first_n_users`: Text renderer for dataset previews.

3. **`simulator.py`**:
   - `PerformanceSimulator`: Benchmarks average lookup latency across table sizes (10 to 1,000 users) and collision rates across load factors (0.25 to 0.90).
   - `compare_numpy_performance`: Compares Python list item counting against `np.bincount`.

4. **`graphs.py`**:
   - Matplotlib styling and canvas plotting utilities (`plot_lookup_benchmark`, `plot_collision_experiment`, `plot_load_factor_vs_lookup_time`) embedded in Tkinter.

5. **`app.py`**:
   - CustomTkinter GUI application featuring simulator controls, operation step trace panel, real-time statistics, output console log, interactive user lookup search animation, and educational popup guide.

6. **`verify.py`**:
   - Verification script testing dataset generation, chaining insertion/search, linear probing insert/search/delete/re-insert, and simulator benchmarks.

---

## 2. Empirical Baseline Performance

Running `python verify.py` yields:

```
Testing data generation...
  Data generation OK
Testing HashTableChaining...
  HashTableChaining OK
Testing HashTableLinearProbing...
  HashTableLinearProbing OK
Testing Simulator with dynamic strategy parameters...
  Simulator OK
All backend strategy checks passed successfully!
```

---

## 3. Current Dependencies

- `customtkinter >= 5.2.0`
- `matplotlib >= 3.7.0`
- `numpy >= 1.24.0`
- `pytest >= 7.0.0`
- `pandas >= 2.0.0`
- `scipy >= 1.10.0`

---

## 4. Key Limitations Identified for Week 7 / Week 8

1. **Random Preference Structure**: Default `generate_user_data` assigns items uniformly at random without latent genre preferences, limiting recommendation quality evaluation.
2. **Cold-Start Vulnerability**: Unknown users produce empty recommendations rather than fallback recommendations.
3. **Trace Overhead in Benchmarks**: Insertion and search methods log step-by-step trace objects for GUI animation, which adds overhead during performance benchmarking.
4. **Lack of Advanced Recommendation Metrics**: Needs Precision@K, Recall@K, Hit Rate@K, NDCG@K, coverage, and popularity bias.
5. **No Sparse Matrix Representation**: Dense calculations require excessive memory for large user/item domains.
6. **Fixed Static Hashing Growth**: Lacks extendible hashing for dynamic disk/memory bucket splitting without full table rehashing.
7. **No End-to-End ETL Pipeline**: Missing standard Pandas/NumPy export pipeline.
