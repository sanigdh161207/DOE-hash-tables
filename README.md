# README.md

````markdown
# DOE Hash Table Recommender System

A Python-based experimental recommender system that combines **Hash Tables, Collision Resolution Techniques, Cosine Similarity, and Performance Analysis**.

## Overview

This project explores how different data structures and recommendation techniques affect the performance of a user-based movie recommendation system.

The system stores user–movie preferences using custom hash tables and compares:

- Separate Chaining
- Linear Probing
- Baseline Recommendation
- Cosine Similarity Recommendation

It also provides experiments for analysing lookup performance, collisions, load factors, probing, and recommendation execution time.

## Features

- Custom Hash Table implementation
- Separate Chaining collision handling
- Linear Probing with tombstone deletion
- Dynamic resizing and rehashing
- Load-factor analysis
- Collision and probe measurement
- Synthetic user/movie dataset generation
- Baseline recommendation system
- User-based collaborative filtering
- Cosine Similarity recommendation
- Similarity-weighted movie scoring
- NumPy-based vector operations
- Performance benchmarking
- Matplotlib visualizations
- Interactive CustomTkinter GUI
- Automated verification tests
- Reproducible and fresh-data experiment modes

## Recommendation Approach

The recommendation pipeline is:

```text
User ID
   ↓
Hash Table Lookup
   ↓
User Preferences
   ↓
Preference Vector
   ↓
Cosine Similarity
   ↓
Similar Users
   ↓
Candidate Movies
   ↓
Similarity-Weighted Ranking
   ↓
Top-N Recommendations
````

Cosine similarity is calculated using:

```text
sim(A,B) = (A · B) / (||A|| × ||B||)
```

The baseline recommender is used as a simple reference method for comparison.

## Hash Table Strategies

### Separate Chaining

Collisions are handled by storing multiple entries in the same bucket.

### Linear Probing

Collisions are handled by searching for the next available position in the table.

The project compares both strategies under different dataset sizes and load factors.

## Experiments

The experimentation suite evaluates different dataset sizes, including:

```text
10
50
100
500
1000 users
```

Key measurements include:

* Lookup time
* Collision count
* Probe count
* Load factor
* Memory estimation
* Similarity calculation time
* Recommendation generation time
* Overall recommendation execution time

## Week 6 Experiment

The cosine-similarity experiment evaluates:

```text
Baseline Recommendation
        vs
Cosine Similarity Recommendation
```

The dedicated experiment uses a **500-user / 100-movie** dataset and evaluates recommendation and timing behaviour.

## Project Structure

```text
DOE-hash-tables/
│
├── app.py
├── recommender.py
├── simulator.py
├── data_generator.py
├── graphs.py
├── verify.py
├── requirements.txt
└── README.md
```

### Main Files

| File                | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `app.py`            | Graphical user interface                   |
| `recommender.py`    | Hash tables and recommendation algorithms  |
| `simulator.py`      | Performance and recommendation experiments |
| `data_generator.py` | Synthetic dataset generation               |
| `graphs.py`         | Experiment visualizations                  |
| `verify.py`         | Automated verification                     |
| `requirements.txt`  | Project dependencies                       |

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* CustomTkinter
* Git & GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/sanigdh161207/DOE-hash-tables.git
cd DOE-hash-tables
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

## Run Verification

```bash
python verify.py
```

The verification suite checks hash-table operations, resizing, deletion, cosine similarity, recommendation correctness, determinism, and compatibility with both hash-table implementations.

## Complexity

| Operation              |  Average | Worst Case |
| ---------------------- | -------: | ---------: |
| Hash Table Search      |     O(1) |       O(N) |
| Hash Table Insert      |     O(1) |       O(N) |
| Hash Table Delete      |     O(1) |       O(N) |
| Cosine Similarity      |     O(V) |       O(V) |
| User Similarity Search | O(U × V) |   O(U × V) |

Where `U` is the number of users and `V` is the number of movies/features.

## Project Objective

The project demonstrates the practical relationship between:

```text
Data Structures
      +
Algorithms
      +
Recommendation Systems
      +
Performance Analysis
      +
Experimental Design
```

The primary focus is to evaluate how **hash-table design and recommendation algorithms influence system performance and recommendation behaviour**.

## Limitations

* Uses synthetic user/movie data
* Binary user preferences
* No real-time user feedback
* Cold-start limitations
* No semantic movie/content information
* Similarity search is performed across users

## Future Scope

Potential extensions include:

* Weighted Cosine Similarity
* Sparse matrices
* Real-world datasets
* Rating-based recommendations
* Feedback-based recommendations
* Diversity-aware recommendations
* Context-aware recommendations
* Multi-factor recommendation
* Extendible Hashing
* Optimized similarity search

## Repository

[https://github.com/sanigdh161207/DOE-hash-tables](https://github.com/sanigdh161207/DOE-hash-tables)

```
```
