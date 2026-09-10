````markdown
# Hash Table Recommender Simulator & Experimentation Suite

A data-structure-driven **movie recommendation system** built to study the relationship between **hash tables, collision handling, similarity-based recommendations, and performance experimentation**.

This project combines **Separate Chaining**, **Linear Probing**, and **Cosine Similarity** to demonstrate how different data-structure and algorithmic choices affect system performance and recommendation results.

---

## Project Overview

The system simulates a movie recommendation platform where users have lists of movies they like.

The project focuses on two major areas:

1. **Efficient data storage and retrieval using Hash Tables**
2. **Generating recommendations using User-Based Collaborative Filtering and Cosine Similarity**

The project also includes experiments for measuring:

- Lookup performance
- Collision behavior
- Load factor
- Probing behavior
- Memory usage
- Recommendation generation time
- Similarity calculation time
- Recommendation differences between algorithms

---

## Main Objective

The main objective of this project is:

> **To demonstrate how efficient data structures and similarity algorithms can be combined to build and evaluate a recommendation system.**

The project investigates how changing the underlying hash-table collision strategy and recommendation algorithm affects performance and recommendation behavior.

---

# System Architecture

```text
                    User
                     |
                     v
              CustomTkinter GUI
                     |
                     v
            Recommender System
                     |
          +----------+----------+
          |                     |
          v                     v
     Hash Table             Recommendation
      Storage                  Engine
          |                     |
     +----+----+          +-----+------+
     |         |          |            |
     v         v          v            v
 Chaining  Linear      Baseline    Cosine
            Probing                 Similarity
                                      |
                                      v
                             Similar Users
                                      |
                                      v
                            Movie Recommendations
````

---

# Key Features

## 1. Custom Hash Table Implementation

The project implements hash tables from scratch instead of relying only on Python's built-in dictionary.

Each entry stores:

```text
User ID → List of Movie IDs
```

Example:

```text
101 → [2, 5, 8, 12]
102 → [1, 4, 7]
103 → [2, 5, 9]
```

The hash function uses:

```text
index = |key| % table_size
```

This allows User IDs to be mapped to positions in the hash table.

---

# 2. Separate Chaining

The first collision-handling strategy is **Separate Chaining**.

When multiple keys map to the same index, the entries are stored together inside a bucket list.

```text
Index
  0
  1
  2 → User 102 → User 202 → User 302
  3
  4 → User 104
```

### Advantages

* Simple collision handling
* Works well with higher load factors
* Deletion is straightforward
* Does not require probing for another empty position

### Disadvantages

* Additional memory is required for buckets
* Long chains can increase lookup time
* Performance depends on collision distribution

---

# 3. Linear Probing

The second collision-handling strategy is **Linear Probing**, an open-addressing technique.

If the calculated position is already occupied, the system checks the next available position.

```text
h(k, i) = (h(k) + i) mod M
```

Example:

```text
Hash position → Occupied
       ↓
Check next position
       ↓
Check next position
       ↓
Insert into empty position
```

### Advantages

* No separate bucket structure
* Better cache locality
* Efficient memory usage
* Simple implementation

### Disadvantages

* Primary clustering can occur
* Performance decreases as load factor increases
* More probes may be required for searches
* Deletion requires special handling

The implementation uses **tombstones** to preserve the probing sequence after deletion.

---

# 4. Load Factor

Load factor measures how full the hash table is.

```text
α = N / M
```

Where:

* `N` = number of stored entries
* `M` = hash table capacity
* `α` = load factor

For example:

```text
N = 75
M = 100

α = 75 / 100
α = 0.75
```

The project evaluates different load factors to observe their effect on collisions and lookup performance.

---

# 5. Collision Analysis

A collision occurs when two different keys produce the same hash index.

Example:

```text
1057 % 100 = 57
2057 % 100 = 57
```

Both users map to index `57`.

The project measures collision behavior for different dataset sizes and hash-table configurations.

---

# 6. Rehashing and Resizing

When the hash table becomes too full, the table can be resized.

The existing entries are reinserted into the larger table.

```text
Small Table
     ↓
Resize
     ↓
Larger Table
     ↓
Reinsert Existing Entries
```

This helps maintain efficient lookup performance.

---

# 7. Recommendation System

The recommendation engine uses user preferences to generate movie recommendations.

The main recommendation approach is **User-Based Collaborative Filtering**.

The idea is:

> Users with similar movie preferences are likely to enjoy some of the same movies.

Example:

```text
User A → [Movie 1, Movie 2, Movie 3]

User B → [Movie 1, Movie 2, Movie 3, Movie 5]

User C → [Movie 1, Movie 4]

User A and User B are more similar.

Therefore:
Movie 5 can be recommended to User A.
```

---

# 8. Baseline Recommendation

A simple deterministic random recommendation method is maintained as the baseline.

The baseline:

1. Retrieves the target user's preferences.
2. Finds movies the user has not already selected.
3. Selects recommendations using a deterministic random process.

This provides a **control/reference method** for comparison with the more sophisticated cosine-similarity recommender.

The baseline is intentionally simple so that improvements or differences in the cosine-based approach can be studied.

---

# 9. Cosine Similarity

The project uses **Cosine Similarity** to measure how similar two users are.

The formula is:

```text
similarity(A, B) =
        A · B
       --------
       ||A|| ||B||
```

Where:

* `A · B` = dot product
* `||A||` = magnitude of vector A
* `||B||` = magnitude of vector B

For the binary movie-preference vectors used in this project, higher values indicate more similar preference patterns.

---

# 10. User-Item Vectors

All users are represented using a common movie vocabulary.

Example movie vocabulary:

```text
[1, 2, 3, 4, 5]
```

If a user likes:

```text
[1, 3, 5]
```

Their vector becomes:

```text
[1, 0, 1, 0, 1]
```

This allows mathematical similarity calculations between users.

NumPy is used for efficient vector operations such as:

* Dot products
* Vector norms
* Cosine similarity
* Array operations

---

# 11. Recommendation Pipeline

The complete recommendation process is:

```text
Target User
     |
     v
Hash Table Lookup
     |
     v
User Preferences
     |
     v
Convert to Preference Vector
     |
     v
Calculate Cosine Similarity
     |
     v
Compare with Other Users
     |
     v
Select Top-K Similar Users
     |
     v
Collect Movies They Like
     |
     v
Remove Movies Already Seen
     |
     v
Similarity-Weighted Scoring
     |
     v
Rank Candidate Movies
     |
     v
Top-N Recommendations
```

---

# 12. Similarity-Weighted Recommendation

Movies from highly similar users receive stronger scores.

For example:

```text
User A similarity = 0.9
User B similarity = 0.4
```

If both users like Movie X:

```text
Movie X score = 0.9 + 0.4
              = 1.3
```

Movies with higher accumulated similarity scores are ranked higher.

This makes the recommendation process more meaningful than simply counting movie occurrences.

---

# 13. Top-K and Top-N

The system uses two important concepts:

### Top-K

The `K` most similar users selected as neighbors.

Example:

```text
Top-K = 10
```

means the system considers the 10 most similar users.

### Top-N

The number of movies finally recommended.

Example:

```text
Top-N = 5
```

means the system returns 5 recommendations.

---

# 14. Performance Experiments

The project includes experiments using generated user-movie datasets.

Dataset sizes include:

```text
10 users
50 users
100 users
500 users
1000 users
```

The experiments evaluate the effect of different load factors and table configurations.

The system measures:

* Search time
* Insert performance
* Collision counts
* Probing behavior
* Memory estimates
* Recommendation execution time

---

# 15. Week 5 Experimentation

The project compares:

```text
Separate Chaining
        VS
Linear Probing
```

The comparison focuses on:

| Metric          | Purpose                        |
| --------------- | ------------------------------ |
| Lookup Time     | Measures search performance    |
| Collision Count | Measures collision behavior    |
| Probe Count     | Measures Linear Probing effort |
| Load Factor     | Measures table occupancy       |
| Memory Estimate | Compares storage requirements  |

This demonstrates the practical trade-offs between different collision-resolution strategies.

---

# 16. Week 6 Cosine Similarity Experiment

The Week 6 experiment compares:

```text
Baseline Recommender
        VS
Cosine Similarity Recommender
```

The dedicated experiment uses:

```text
Users: 500
Movies: 100
Recommendations: 5
```

The system measures:

* Baseline total time
* Cosine total time
* Hash lookup time
* Similarity calculation time
* Recommendation generation time
* Average top similarity
* Recommendation differences

---

# 17. Reproducible and Fresh Data Modes

The experiment supports two data-generation modes.

### Reproducible Mode

Uses a fixed seed:

```text
Seed = 42
```

This produces the same dataset and results across runs.

Useful for:

* Debugging
* Verification
* Demonstrations
* Reproducible experiments

### Fresh Data Mode

Uses a dynamically generated seed.

This creates new random datasets for each experiment.

Useful for:

* Testing robustness
* Observing variation
* Evaluating behavior across different datasets

---

# 18. NumPy Benchmark

The project also compares standard Python counting with NumPy.

The benchmark compares:

```text
Python list/dictionary-based counting
        VS
NumPy np.bincount()
```

This demonstrates how numerical computing libraries can improve operations on suitable numerical datasets.

---

# 19. Automated Verification

The project contains an automated verification suite.

The verification system checks:

1. Data generation
2. Separate Chaining operations
3. Linear Probing operations
4. Resizing
5. Tombstone deletion
6. Performance simulator
7. Cosine similarity correctness
8. Global movie vocabulary
9. Target-user exclusion
10. Recommendation validity
11. Recommendation determinism
12. Baseline compatibility
13. Linear Probing recommender compatibility

Expected result:

```text
ALL AUTOMATED TESTS PASSED SUCCESSFULLY!
```

---

# Mathematical Foundations

## Hash Function

```text
h(k) = |k| mod M
```

## Load Factor

```text
α = N / M
```

## Linear Probing

```text
h(k,i) = (h(k) + i) mod M
```

## Cosine Similarity

```text
sim(A,B) = (A · B) / (||A|| ||B||)
```

---

# Complexity Analysis

| Operation             |       Average Case | Worst Case |
| --------------------- | -----------------: | ---------: |
| Hash Search           |               O(1) |       O(N) |
| Hash Insert           |               O(1) |       O(N) |
| Hash Delete           |               O(1) |       O(N) |
| Linear Probing Search |               O(1) |       O(M) |
| Cosine Similarity     | O(V) per user pair |       O(V) |
| Similarity Search     |           O(U × V) |   O(U × V) |
| Sorting Similar Users |         O(U log U) | O(U log U) |

Where:

* `N` = number of stored entries
* `M` = table capacity
* `U` = number of users
* `V` = number of movies/features

**Important:** The hash-table lookup is approximately O(1) on average, but the complete cosine recommendation process is not O(1) because it compares the target user against multiple users.

---

# Technologies Used

* **Python**
* **CustomTkinter** — Graphical User Interface
* **NumPy** — Numerical and vector calculations
* **Pandas** — Data analysis
* **Matplotlib** — Graphs and visualization
* **time.perf_counter()** — Performance measurement
* **Git & GitHub** — Version control

---

# Project Structure

```text
DOE-hash-tables/
│
├── app.py
│   └── Main GUI application
│
├── recommender.py
│   └── Hash tables and recommendation algorithms
│
├── simulator.py
│   └── Performance and experimentation logic
│
├── data_generator.py
│   └── Synthetic user/movie data generation
│
├── graphs.py
│   └── Performance and experiment visualizations
│
├── verify.py
│   └── Automated verification suite
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/sanigdh161207/DOE-hash-tables.git
```

Move into the project directory:

```bash
cd DOE-hash-tables
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Application

Run:

```bash
python app.py
```

The graphical interface provides access to the recommendation system, hash-table demonstrations, experiments, and visualizations.

---

# Running Verification Tests

Run:

```bash
python verify.py
```

A successful execution should report:

```text
ALL AUTOMATED TESTS PASSED SUCCESSFULLY!
```

---

# Experimental Design

The project follows an experimental approach rather than only implementing a working recommender.

The main independent variables include:

* Dataset size
* Hash-table capacity
* Load factor
* Collision-resolution strategy
* Recommendation algorithm

Measured outcomes include:

* Lookup time
* Collision count
* Probe count
* Memory usage
* Similarity calculation time
* Recommendation generation time
* Recommendation differences

This helps identify how algorithmic choices affect system behavior.

---

# Key Trade-Offs

## Separate Chaining vs Linear Probing

```text
Separate Chaining
    ↓
More flexible at high load
    ↓
Additional bucket memory
```

```text
Linear Probing
    ↓
Compact memory usage
    ↓
Potential clustering and increased probes
```

## Baseline vs Cosine Similarity

```text
Baseline
    ↓
Simple and fast
    ↓
Limited recommendation intelligence
```

```text
Cosine Similarity
    ↓
Uses relationships between users
    ↓
More computationally expensive
    ↓
Potentially more meaningful recommendations
```

---

# Important Project Insight

A major finding of the project is that:

> **Lookup speed and recommendation quality are different dimensions of a recommendation system.**

A hash table can provide very fast user-data retrieval, but fast retrieval alone does not guarantee good recommendations.

The recommendation algorithm determines how the retrieved data is interpreted.

Therefore, the project studies both:

```text
Data Structure Efficiency
          +
Recommendation Logic
          =
Overall System Behavior
```

---

# Limitations

The current system has several limitations:

* Uses synthetic user/movie data
* Does not use a real-world movie dataset
* Uses binary preference information
* Does not model ratings or watch duration
* Has cold-start limitations for new users
* Does not currently use real-time user feedback
* Cosine similarity currently compares the target user against other users, rather than using an optimized nearest-neighbor index
* Movie identity alone does not capture semantic movie content

---

# Future Improvements

Possible future extensions include:

* Weighted Cosine Similarity
* Sparse user-item matrices
* Real movie datasets
* Rating-based recommendations
* User feedback loops
* Popularity-aware recommendations
* Diversity-aware recommendations
* Context-aware recommendations
* Multi-factor recommendation models
* Optimized nearest-neighbor search
* Extendible Hashing
* A* or graph-based recommendation approaches
* Improved cold-start handling

---

# Educational Concepts Demonstrated

This project demonstrates practical applications of:

* Hash Tables
* Hash Functions
* Collision Resolution
* Separate Chaining
* Linear Probing
* Primary Clustering
* Tombstones
* Load Factor
* Dynamic Resizing
* Rehashing
* Big-O Complexity
* Collaborative Filtering
* Cosine Similarity
* Vector Representation
* Top-K Neighbor Selection
* Similarity-Weighted Scoring
* Experimental Design
* Performance Benchmarking
* Data Visualization
* NumPy Vector Operations

---

# Academic Focus

The project is designed as an experimentation-based implementation rather than simply a recommendation application.

The core academic focus is:

```text
DATA STRUCTURE
      +
ALGORITHM
      +
EXPERIMENTATION
      +
PERFORMANCE ANALYSIS
      +
RECOMMENDATION SYSTEM
```

The project demonstrates how theoretical concepts such as **hashing, collision resolution, complexity analysis, and vector similarity** can be implemented and evaluated in a practical recommendation-system scenario.

---

# Conclusion

The Hash Table Recommender Simulator demonstrates how a recommendation system can be built using fundamental data structures and mathematical similarity techniques.

The project combines:

* Custom hash-table implementations
* Separate Chaining
* Linear Probing
* Collision and load-factor analysis
* Performance benchmarking
* NumPy-based vector operations
* Cosine similarity
* User-based collaborative filtering
* Baseline comparison
* Interactive visualization
* Automated verification

The overall goal is to understand not only **how to build the system**, but also **how different design decisions affect its performance and behavior**.

---

## Repository

**GitHub:**
[https://github.com/sanigdh161207/DOE-hash-tables](https://github.com/sanigdh161207/DOE-hash-tables)

```
```
