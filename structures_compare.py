import time
import random
import sys
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

from recommender import HashTableChaining, HashTableLinearProbing
from data_generator import generate_user_data

class BSTNode:
    def __init__(self, key: int, value: Any):
        self.key = key
        self.value = value
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None

class BST:
    """
    Minimal Binary Search Tree implementation for benchmark comparisons.
    """
    def __init__(self):
        self.root: Optional[BSTNode] = None
        self.num_keys = 0

    def insert(self, key: int, value: Any) -> None:
        if not self.root:
            self.root = BSTNode(key, value)
            self.num_keys += 1
            return
        curr = self.root
        while True:
            if key == curr.key:
                curr.value = value
                return
            elif key < curr.key:
                if curr.left is None:
                    curr.left = BSTNode(key, value)
                    self.num_keys += 1
                    return
                curr = curr.left
            else:
                if curr.right is None:
                    curr.right = BSTNode(key, value)
                    self.num_keys += 1
                    return
                curr = curr.right

    def search(self, key: int) -> Optional[Any]:
        curr = self.root
        while curr:
            if key == curr.key:
                return curr.value
            elif key < curr.key:
                curr = curr.left
            else:
                curr = curr.right
        return None


class SkipListNode:
    def __init__(self, key: int, value: Any, level: int):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)

class SkipList:
    """
    Minimal Skip List implementation for benchmark comparisons.
    """
    def __init__(self, max_level: int = 16, p: float = 0.5):
        self.max_level = max_level
        self.p = p
        self.header = SkipListNode(-1, None, max_level)
        self.level = 0
        self.num_keys = 0

    def _random_level(self) -> int:
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, key: int, value: Any) -> None:
        update = [None] * (self.max_level + 1)
        curr = self.header
        for i in range(self.level, -1, -1):
            while curr.forward[i] and curr.forward[i].key < key:
                curr = curr.forward[i]
            update[i] = curr

        curr = curr.forward[0]
        if curr and curr.key == key:
            curr.value = value
            return

        rlevel = self._random_level()
        if rlevel > self.level:
            for i in range(self.level + 1, rlevel + 1):
                update[i] = self.header
            self.level = rlevel

        n = SkipListNode(key, value, rlevel)
        for i in range(rlevel + 1):
            n.forward[i] = update[i].forward[i]
            update[i].forward[i] = n
        self.num_keys += 1

    def search(self, key: int) -> Optional[Any]:
        curr = self.header
        for i in range(self.level, -1, -1):
            while curr.forward[i] and curr.forward[i].key < key:
                curr = curr.forward[i]
        curr = curr.forward[0]
        if curr and curr.key == key:
            return curr.value
        return None


def run_structures_comparison(
    sizes: List[int] = [10, 100, 500, 1000, 5000]
) -> List[Dict[str, Any]]:
    results = []

    for size in sizes:
        dataset = generate_user_data(size)
        keys = [u for u, _ in dataset]

        # Table size calculation for hash tables
        t_size = max(13, int(size / 0.7))
        if t_size % 2 == 0:
            t_size += 1

        # 1. HashTableChaining
        ht_chain = HashTableChaining(t_size)
        start = time.perf_counter()
        for u, m in dataset:
            ht_chain.insert(u, m, record_trace=False)
        chain_ins_t = (time.perf_counter() - start) / size

        start = time.perf_counter()
        for k in keys:
            ht_chain.search(k, record_trace=False)
        chain_srch_t = (time.perf_counter() - start) / size

        # 2. HashTableLinearProbing
        ht_probe = HashTableLinearProbing(t_size)
        start = time.perf_counter()
        for u, m in dataset:
            ht_probe.insert(u, m, record_trace=False)
        probe_ins_t = (time.perf_counter() - start) / size

        start = time.perf_counter()
        for k in keys:
            ht_probe.search(k, record_trace=False)
        probe_srch_t = (time.perf_counter() - start) / size

        # 3. Python Dict
        py_dict = {}
        start = time.perf_counter()
        for u, m in dataset:
            py_dict[u] = m
        dict_ins_t = (time.perf_counter() - start) / size

        start = time.perf_counter()
        for k in keys:
            _ = py_dict[k]
        dict_srch_t = (time.perf_counter() - start) / size

        # 4. BST
        bst = BST()
        start = time.perf_counter()
        for u, m in dataset:
            bst.insert(u, m)
        bst_ins_t = (time.perf_counter() - start) / size

        start = time.perf_counter()
        for k in keys:
            bst.search(k)
        bst_srch_t = (time.perf_counter() - start) / size

        # 5. SkipList
        random.seed(42)
        skiplist = SkipList()
        start = time.perf_counter()
        for u, m in dataset:
            skiplist.insert(u, m)
        skip_ins_t = (time.perf_counter() - start) / size

        start = time.perf_counter()
        for k in keys:
            skiplist.search(k)
        skip_srch_t = (time.perf_counter() - start) / size

        row = {
            "size": size,
            "chain_insert_us": chain_ins_t * 1e6,
            "chain_search_us": chain_srch_t * 1e6,
            "probe_insert_us": probe_ins_t * 1e6,
            "probe_search_us": probe_srch_t * 1e6,
            "dict_insert_us": dict_ins_t * 1e6,
            "dict_search_us": dict_srch_t * 1e6,
            "bst_insert_us": bst_ins_t * 1e6,
            "bst_search_us": bst_srch_t * 1e6,
            "skip_insert_us": skip_ins_t * 1e6,
            "skip_search_us": skip_srch_t * 1e6,
        }
        results.append(row)

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    fieldnames = list(results[0].keys())
    with open("results/structures_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Generate plot
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
    sizes_x = [r["size"] for r in results]
    ax.plot(sizes_x, [r["chain_search_us"] for r in results], marker="o", label="Hash Table (Chaining)")
    ax.plot(sizes_x, [r["probe_search_us"] for r in results], marker="s", label="Hash Table (Probing)")
    ax.plot(sizes_x, [r["dict_search_us"] for r in results], marker="^", label="Python dict (C-Hash)")
    ax.plot(sizes_x, [r["bst_search_us"] for r in results], marker="x", label="BST (Unbalanced)")
    ax.plot(sizes_x, [r["skip_search_us"] for r in results], marker="d", label="Skip List")

    ax.set_title("Data Structure Lookup Benchmark", fontweight="bold")
    ax.set_xlabel("Dataset Size (Keys)")
    ax.set_ylabel("Avg Lookup Latency (microseconds)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/structures_comparison.png")
    plt.close(fig)

    return results

if __name__ == "__main__":
    res = run_structures_comparison()
    print("Data structures comparison complete. CSV and plot saved to results/.")
