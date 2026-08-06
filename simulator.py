import time
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from recommender import HashTableChaining, HashTableLinearProbing
from data_generator import generate_user_data

class PerformanceSimulator:
    """
    Executes various benchmark tests comparing Separate Chaining and Linear Probing.
    """
    
    @staticmethod
    def get_strategy_class(name: str):
        if name == "Linear Probing":
            return HashTableLinearProbing
        return HashTableChaining

    @staticmethod
    def run_lookup_benchmark(sizes: List[int] = [10, 50, 100, 500, 1000], strategy: str = "Separate Chaining") -> Dict[str, List[Tuple[int, float, int, int]]]:
        """
        Measures lookup time, collisions, and memory across different dataset sizes.
        Returns a dict mapping strategy name -> list of (size, avg_time, collisions, memory_bytes).
        """
        strategies = ["Separate Chaining", "Linear Probing"] if strategy == "Both" else [strategy]
        all_results = {}

        for strat in strategies:
            strat_class = PerformanceSimulator.get_strategy_class(strat)
            results = []
            for size in sizes:
                dataset = generate_user_data(size)
                table_size = max(13, int(size / 0.7))
                if table_size % 2 == 0:
                    table_size += 1
                    
                ht = strat_class(table_size)
                for user_id, movies in dataset:
                    ht.insert(user_id, movies)
                    
                num_lookups = 1000
                lookup_keys = [random.choice(dataset)[0] for _ in range(num_lookups)]
                
                start_time = time.perf_counter()
                for key in lookup_keys:
                    ht.search(key)
                end_time = time.perf_counter()
                
                avg_time = (end_time - start_time) / num_lookups
                stats = ht.get_collision_statistics()
                mem = ht.estimate_memory_bytes()
                results.append((size, avg_time, stats["collisions"], mem))
            all_results[strat] = results

        return all_results

    @staticmethod
    def run_collision_experiment(load_factors: List[float] = [0.25, 0.50, 0.75, 0.90], dataset_size: int = 1000, strategy: str = "Separate Chaining") -> Dict[str, List[Tuple[float, float, float, int, Dict[str, Any]]]]:
        """
        Measures the collision rate, average lookup time, and strategy-specific metrics.
        Returns a dict mapping strategy name -> list of (lf, collision_rate, avg_lookup_time, memory_bytes, extra_stats).
        """
        strategies = ["Separate Chaining", "Linear Probing"] if strategy == "Both" else [strategy]
        all_results = {}
        dataset = generate_user_data(dataset_size)
        
        for strat in strategies:
            strat_class = PerformanceSimulator.get_strategy_class(strat)
            results = []
            for lf in load_factors:
                table_size = int(dataset_size / lf)
                if table_size <= 0:
                    table_size = 1
                    
                ht = strat_class(table_size)
                
                # We also track probe count during insertions for linear probing
                for user_id, movies in dataset:
                    ht.insert(user_id, movies)
                    
                num_lookups = 1000
                lookup_keys = [random.choice(dataset)[0] for _ in range(num_lookups)]
                
                # Perform lookups and capture total probes if linear probing
                total_search_probes = 0
                start_time = time.perf_counter()
                for key in lookup_keys:
                    _, trace = ht.search(key)
                    if strat == "Linear Probing":
                        # Sum up probes from search trace steps
                        for step in trace:
                            if step.get("step") == "probe_search":
                                total_search_probes += 1
                end_time = time.perf_counter()
                avg_lookup_time = (end_time - start_time) / num_lookups
                
                stats = ht.get_collision_statistics()
                mem = ht.estimate_memory_bytes()
                
                extra_stats = {}
                if strat == "Linear Probing":
                    extra_stats["avg_search_probes"] = total_search_probes / num_lookups
                    extra_stats["insert_collisions"] = ht.collision_count()
                else:
                    # Separate Chaining chain statistics
                    bucket_lengths = [len(b) for b in ht.buckets]
                    extra_stats["max_chain"] = max(bucket_lengths) if bucket_lengths else 0
                    extra_stats["avg_chain"] = sum(bucket_lengths) / len(bucket_lengths) if bucket_lengths else 0.0

                results.append((lf, stats["collision_rate"], avg_lookup_time, mem, extra_stats))
            all_results[strat] = results
            
        return all_results

    @staticmethod
    def compare_numpy_performance(dataset_size: int = 5000) -> Dict[str, Any]:
        """
        Compares speed of storing and manipulating item interaction counts using NumPy vs lists.
        """
        dataset = generate_user_data(dataset_size)
        
        all_items_list = []
        for _, items in dataset:
            all_items_list.extend(items)
            
        iterations = 50
        
        start = time.perf_counter()
        for _ in range(iterations):
            max_item_id = 100
            counts = [0] * (max_item_id + 1)
            for item in all_items_list:
                if item <= max_item_id:
                    counts[item] += 1
        py_list_time = (time.perf_counter() - start) / iterations
        
        np_items = np.array(all_items_list, dtype=np.int32)
        
        start = time.perf_counter()
        for _ in range(iterations):
            counts_np = np.bincount(np_items)
        numpy_time = (time.perf_counter() - start) / iterations
        
        # Reference Chaining Table search time
        ht = HashTableChaining(max(17, int(dataset_size / 0.7)))
        for user_id, movies in dataset:
            ht.insert(user_id, movies)
        target_user_id = dataset[int(dataset_size * 0.8)][0]
        
        start = time.perf_counter()
        for _ in range(iterations * 10):
            ht.search(target_user_id)
        hash_table_time = (time.perf_counter() - start) / (iterations * 10)
        
        return {
            "python_list_time": py_list_time,
            "numpy_time": numpy_time,
            "hash_table_time": hash_table_time,
            "speedup_numpy": py_list_time / numpy_time if numpy_time > 0 else 0,
            "speedup_hash": py_list_time / hash_table_time if hash_table_time > 0 else 0,
            "speedup_hash_vs_numpy": numpy_time / hash_table_time if hash_table_time > 0 else 0
        }
