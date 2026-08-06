# Hash Table Recommender Simulator

A clean, modern educational desktop application that visually demonstrates how a Hash Table works under the hood while running lookup and collision experiments. Built with CustomTkinter, Matplotlib, and NumPy.

## Project Overview

This simulator provides an interactive, hands-on tool for learning how hash tables function. By generating mock user profiles (which contain User IDs mapped to lists of preferred movie IDs), users can inspect how records are distributed across memory buckets, observe collision resolution through chaining, and analyze performance benchmarks.

## Features

- **Dataset Generation**: Generates reproducible datasets of varying sizes (from 10 up to 1,000 users) containing preferences.
- **Customizable Hash Table**: Populate a hash table dynamically with target load factors or customize the table capacity.
- **Visual Chaining Table**: Displays the underlying bucket array in a scrollable list, highlighting collided buckets in orange.
- **Interactive Animations**: Step-by-step trace animations showing how key elements are hashed, mapped, traversed, and inserted or searched.
- **Performance Benchmarking**:
  - Compare lookup times across multiple scales.
  - Observe how load factor correlates with collision rate.
  - Contrast standard Python list scans, NumPy vectorized searches, and Hash Table operations.
- **Matplotlib Dashboard**: Dynamic performance charts embedded directly inside the desktop window.
- **Educational Guide**: An in-app visual guide explaining hashing math, collisions, load factors, and rehashing.

## Folder Structure

```
HashTableSimulator/
├── app.py              # Main CustomTkinter UI dashboard and animations
├── recommender.py      # Custom Hash Table implementation with chaining
├── simulator.py        # Benchmarks, lookup time analysis, collision rate tracking
├── data_generator.py   # Mock user-movie dataset generator
├── graphs.py           # Matplotlib plot generation functions
├── requirements.txt    # Package dependencies
└── README.md           # Documentation
```

## Installation

Ensure you have Python 3.11+ installed. Clone this repository or download the files, navigate to the folder, and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Launch the application directly using Python:

```bash
python app.py
```

## Educational Concepts Covered

1. **Hash Function**: How keys map to index positions via modulo arithmetic:
   $$\text{Index} = \text{User ID} \pmod{\text{Table Size}}$$
2. **Collisions**: Occurrences where multiple keys map to the identical bucket index.
3. **Collision Chaining**: Storing colliding elements in a linked/chain list at the target index.
4. **Load Factor**: The ratio of occupied space ($N/M$) indicating how packed the table has become.
5. **Rehashing**: Dynamic resizing of the table size to spread out items and reduce chains.

## Expected Output

When running the benchmarks, the output console logs metrics such as:
- Stable lookup times across increasing dataset sizes (showing $O(1)$ lookup complexity).
- Exponential growth of collision rates as the load factor approaches $1.0$.
- Over a **10x to 50x speedup** of the Custom Hash Table compared to raw Python list scans.
