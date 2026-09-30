# Week 8 — Smartphone & 2 GB RAM Architectural Adaptation Notes

---

## 1. System Design Constraints (2 GB RAM Environment)

In low-resource environments (e.g., budget mobile devices or edge hardware with $\le 2\text{ GB}$ total RAM), operating systems reserve $1.0 - 1.2\text{ GB}$ for system processes, leaving $< 800\text{ MB}$ available for application heap memory.

---

## 2. Adaptation Strategies

### A. Implemented In-Code Strategies
1. **SciPy Sparse CSR Representation**: Compresses user-item matrices to $\sim 16\text{ MB}$ for $1\text{M} \times 100\text{k}$ items ($0.001\%$ density), avoiding $745\text{ GB}$ dense allocations.
2. **Trace Log Disable (`record_trace=False`)**: Prevents allocating step-by-step trace dictionary objects during background evaluation, reducing garbage collector pressure.
3. **Inverted Index Candidate Restricted Filtering**: Evaluates only active neighbors sharing items, skipping 95%+ of irrelevant user computations.

### B. Proposed Mobile & Edge Optimizations
1. **Float32 Precision Casting**: Downcast float64 matrices to float32, halving matrix memory (`data.nbytes` drops by 50%).
2. **Compact Integer Data Types**: Store user/movie IDs using uint16 or int32 instead of Python 64-bit PyObject integers.
3. **Memory-Mapped I/O (`np.memmap`)**: Stream user interaction files directly from disk storage into virtual memory buffers.
4. **Top-K Recommendation Caching**: Cache top-N recommendation results in local LRU hash table cache.
