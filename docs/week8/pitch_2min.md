# Week 8 — 2-Minute Product Pitch Transcript

**Target Spoken Duration**: ~120 seconds (~280 words)

---

## Pitch Script

"Good morning. Traditional recommender systems face a critical dilemma: static top-N lists deliver zero personalization, while brute-force collaborative filtering incurs expensive $O(N)$ lookup latency and massive memory consumption.

To solve this, we built the **DOE Hash Table Recommender Engine**.

At its core, our platform uses custom Hash Table data structures—supporting Separate Chaining, Linear Probing, and Extendible Hashing—to map user profiles into memory with instantaneous $O(1)$ expected lookup speed, completing user preference queries in under **0.35 milliseconds**.

For recommendation intelligence, we developed an **IDF-weighted user-based collaborative filtering engine** enhanced with popularity exponent damping. In empirical tests across 500 structured user profiles, our system achieved a **Hit-Rate@5 of 0.7840**—over **3.3 times higher** than random baselines—while boosting catalog coverage to **94%** and reducing blockbuster popularity bias by **56.5%**.

To handle real-world scale and resource constraints, we implemented **SciPy Sparse CSR matrix representations** and **Extendible Hashing**. For a domain of 1,000,000 users and 100,000 items, sparse CSR representation reduces memory footprint from **745 Gigabytes down to just 16 Megabytes**—a **99.98% memory reduction**. Meanwhile, Extendible Hashing eliminates full-table rehashing latency spikes by reducing key movements during table expansion by **87.5%**.

Finally, our platform includes an end-to-end Pandas and NumPy pipeline, a comprehensive CLI, and an interactive CustomTkinter GUI for real-time operation tracing.

Our system proves that pair-matching tailored data structures with collaborative filtering algorithms yields an ultra-fast, memory-efficient, and highly personalized recommendation engine. Thank you."
