# Hash Table Recommender Simulator

A modern educational desktop application that visually demonstrates how Hash Tables work under the hood, compares collision-resolution strategies, and provides intelligent movie recommendations — all powered by CustomTkinter, Matplotlib, NumPy, and SciPy.

## Project Overview

Built for **DOE Week 7 + Week 8**, this simulator provides an interactive, hands-on tool for learning how hash tables function and how they can power a recommender engine. Users generate synthetic movie-preference profiles, inspect how records distribute across memory buckets, observe collision resolution through three strategies, run performance benchmarks, and compare recommendation algorithms.

## Features

- **Dataset Generation**: Reproducible datasets (10–1,000 users) with 70–80% genre-preference skew.
- **Three Collision Strategies**: Separate Chaining, Linear Probing, and Extendible Hashing.
- **Visual Hash Table Grid**: Scrollable bucket array showing chained collisions, empty slots, and occupancy.
- **Search Animation**: Step-by-step trace showing hash computation, bucket traversal, and result.
- **Recommender Lab**: Five algorithms — Baseline, Cosine Similarity, IDF-Weighted Cosine, Hybrid Cold-Start, and Sparse CSR Cosine.
- **Performance Benchmarks**: Lookup latency, collision rate vs. load factor, NumPy bincount comparison.
- **Embedded Charts**: Three Matplotlib plots rendered inside the GUI.
- **Educational Guide**: In-app "How It Works" popup covering every button, panel, and concept.
- **CLI Entry Point**: `main.py` runs all experiments and generates reports from the terminal.
- **Full Test Suite**: 34+ unit and integration tests via `pytest`.

## Folder Structure

```
HashTableSimulator/
├── app.py                  # CustomTkinter GUI (Visualizer, Recommender Lab, Charts, Edu)
├── main.py                 # CLI entry point — runs all experiments & generates reports
├── recommender.py          # Hash Table implementations + Cosine/IDF/Hybrid recommender
├── sparse_recommender.py   # SciPy CSR sparse-matrix recommender
├── extendible_hashing.py   # Extendible Hash Table with directory splitting
├── simulator.py            # Benchmarks: lookup, collision, scaling, NumPy comparison
├── data_generator.py       # Synthetic user-movie dataset generator (genre-skewed)
├── graphs.py               # Matplotlib chart generation functions
├── pipeline.py             # Pandas + NumPy ETL pipeline
├── evaluation.py           # Precision/Recall/F1 evaluation metrics
├── structures_compare.py   # Head-to-head structure comparison experiments
├── hot1_memory_model.py    # Memory footprint analysis model
├── score_rubric.py         # Automated rubric scoring tool
├── verify.py               # Quick project integrity checker
├── requirements.txt        # Python dependencies
├── tests/                  # pytest test suite (34+ tests)
├── docs/
│   ├── week7/              # Week 7 reports and analysis
│   └── week8/              # Week 8 reports and analysis
└── results/                # Generated charts and experiment outputs
```

## Installation

Requires **Python 3.11+**. Clone and install dependencies:

```bash
git clone https://github.com/sanigdh161207/DOE-hash-tables.git
cd DOE-hash-tables
pip install -r requirements.txt
```

## Running the Application

### GUI Simulator
```bash
python app.py
```

### CLI (run all experiments)
```bash
python main.py all
```

### Tests
```bash
pytest tests/ -v
```

---

## Simulator User Guide

### Quick-Start Workflow

1. **Configure** — Pick a Dataset Size, Backend Data Structure, and Load Factor in the left sidebar.
2. **Generate Data** — Click **"1. Generate Structured Data"**.
3. **Populate Table** — Click **"2. Populate Hash Table"**.
4. **Explore** — Use the Hash Table Grid, Recommender Lab, and benchmark buttons.
5. **Visualize** — Click **"Generate Charts"** for embedded performance plots.

### Button-by-Button Reference

| Button | What It Does | Where to Look |
|---|---|---|
| **1. Generate Structured Data** | Creates synthetic user-movie profiles with genre skew | Console: first 10 users printed |
| **2. Populate Hash Table** | Inserts all users into the chosen hash table; initializes recommender engines | Visualizer tab opens; Stats panel updates |
| **3. View Hash Table Grid** | Displays the internal bucket array | Visualizer tab: rows = buckets, pills = records, arrows = chains |
| **Search & Animate** | Traces a lookup step-by-step for a given User ID | Step Trace panel animates; Console logs result |
| **4. Open Recommender Lab** | Switches to the recommendation tab | Enter a User ID, pick Top N and Algorithm, then generate |
| **Run Lookup Benchmark** | Measures avg lookup time at sizes 10–1000 | Console: Users / Avg Lookup / Collisions / Mem |
| **Run Collision Test** | Tests collision rates at load factors 0.25–0.90 | Console: Load Factor / Collision Rate / Avg Lookup |
| **Compare NumPy bincount** | Python list vs NumPy speed test | Console: times and speedup factor |
| **Generate Charts** | Renders 3 Matplotlib plots in the Charts tab | Charts tab: Lookup / Collision Rate / Load Factor charts |
| **How It Works (Edu)** | Opens a detailed educational popup | Popup: full guide for every control and concept |
| **Reset Simulator** | Clears all data and resets the UI | Fresh state — configure and start over |

### How to Read the Output

**Hash Table Visualizer**
- Each row is a bucket index (`Index 00`, `Index 01`, …).
- Cyan-bordered pills show stored records (`User 1005 — 8 items`).
- Arrows (`➔`) between pills indicate chained collisions at the same index.
- Orange-bordered rows highlight buckets with ≥ 2 collisions.
- `[ Empty ]` means the bucket is unused.

**Stats Panel**
- **Users** — records stored.
- **Table Size** — total bucket count.
- **Collisions** — inserts that hashed to an occupied bucket.
- **Load Factor** — Users / Table Size.
- **Avg Bucket Len** — mean chain length (chaining) or N/A (probing).
- **Max Bucket Len** — longest chain (worst-case cost).

**Recommender Lab**
- Header shows the algorithm, user ID, and latency in milliseconds.
- Each card: `Rank #N: Movie ID XX | Source Strategy: xxx`.
- Compare algorithms by switching and re-generating for the same user.

**Performance Charts**
- **Chart 1 (Lookup)**: X = dataset size, Y = avg lookup time. Flat line = O(1).
- **Chart 2 (Collision)**: X = load factor, Y = collision %. Steep rise above 0.75.
- **Chart 3 (LF vs Time)**: X = load factor, Y = lookup latency.

---

## Educational Concepts Covered

1. **Hash Function**: `Index = UserID mod TableSize` — maps keys to bucket positions.
2. **Collisions**: Multiple keys mapping to the same index.
3. **Separate Chaining**: Linked lists at each bucket for colliding keys.
4. **Linear Probing**: Scan forward for the next empty slot.
5. **Extendible Hashing**: Dynamic directory with bitwise depth splitting.
6. **Load Factor**: α = N/M — tables resize when α > 0.75.
7. **Cosine Similarity**: `cos(θ) = (A·B) / (||A|| × ||B||)`.
8. **IDF Weighting**: `IDF(m) = log(TotalUsers / UsersWhoWatched(m))`.
9. **Sparse CSR Matrices**: Compressed row storage for efficient large-scale operations.

## Expected Output

- Stable O(1) lookup times across increasing dataset sizes.
- Exponential collision growth as load factor approaches 1.0.
- 10×–50× speedup with NumPy bincount over pure Python.
- Meaningful, genre-aligned recommendations from the cosine/IDF engines.
