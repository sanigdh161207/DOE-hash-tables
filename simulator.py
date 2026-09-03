import time
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from recommender import HashTableChaining, HashTableLinearProbing, RecommenderSystem
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

    @staticmethod
    def run_week6_experiment(
        num_users: int = 500,
        num_movies: int = 100,
        top_n: int = 5,
        top_k_users: int = 10,
        test_user_indices: Optional[List[int]] = None,
        seed: Optional[int] = 42,
        mode_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Week 6 experiment comparing Baseline (Random) vs Cosine Similarity Recommendations:
        - Exactly num_users (default 500) and num_movies (default 100).
        - Same generated dataset for both methods.
        - Supports:
            * MODE A: Reproducible Week 6 Experiment (seed=42, fixed users [1912, 2751, 2440])
            * MODE B: Fresh Data Demo (seed=None, freshly generated users)
        - Measures:
            * recommendation time (total, hash lookup, similarity calc, rec gen)
            * similarity scores
            * recommendations
        - Evaluates and prints three test users.
        """
        if test_user_indices is None:
            test_user_indices = [0, num_users // 2, num_users - 1]

        if mode_label is None:
            mode_label = "MODE A — REPRODUCIBLE WEEK 6 EXPERIMENT (Seed: 42)" if seed is not None else "MODE B — FRESH DATA DEMO (Dynamic Seed)"

        # 1. Generate single shared dataset
        dataset = generate_user_data(num_users, num_movies, seed=seed)
        movie_vocab = list(range(1, num_movies + 1))
        
        # 2. Setup Recommender System with appropriate table size
        table_size = max(13, int(num_users / 0.7))
        if table_size % 2 == 0:
            table_size += 1
            
        recommender = RecommenderSystem(table_size=table_size, strategy_class=HashTableChaining)
        recommender.fit(dataset, movie_vocab=movie_vocab)
        
        # 3. Benchmark all users across both methods
        baseline_times = []
        cosine_times = []
        sim_calc_times = []
        lookup_times = []
        gen_times = []
        all_top_sim_scores = []
        
        user_ids = [uid for uid, _ in dataset]
        
        for uid in user_ids:
            # Baseline run
            b_recs, b_timing, _ = recommender.recommend_movies_baseline(uid, all_movies=movie_vocab, top_n=top_n)
            baseline_times.append(b_timing["total_time"])
            
            # Cosine run
            c_recs, c_timing, c_neighbors, _ = recommender.recommend_movies_cosine(uid, top_n=top_n, top_k_users=top_k_users)
            cosine_times.append(c_timing["total_time"])
            sim_calc_times.append(c_timing["similarity_calc_time"])
            lookup_times.append(c_timing["hash_lookup_time"])
            gen_times.append(c_timing["rec_gen_time"])
            
            if c_neighbors:
                all_top_sim_scores.append(c_neighbors[0][1])
            else:
                all_top_sim_scores.append(0.0)

        # 4. Detailed Evaluation for 3 Predefined Test Users
        test_users_results = []
        print("\n" + "=" * 80)
        print(f"WEEK 6 EXPERIMENT: {mode_label}")
        print("EVALUATION OF THREE TEST USERS")
        print("=" * 80)
        
        for idx in test_user_indices:
            target_uid, original_prefs = dataset[idx]
            
            # Run Baseline
            b_recs, b_timing, _ = recommender.recommend_movies_baseline(target_uid, all_movies=movie_vocab, top_n=top_n)
            
            # Run Cosine
            c_recs, c_timing, c_neighbors, _ = recommender.recommend_movies_cosine(target_uid, top_n=top_n, top_k_users=top_k_users)
            
            user_data = {
                "user_id": target_uid,
                "original_preferences": original_prefs,
                "baseline_recommendations": b_recs,
                "cosine_recommendations": c_recs,
                "top_similar_users": [u[0] for u in c_neighbors],
                "similarity_scores": [round(u[1], 4) for u in c_neighbors],
                "baseline_time_sec": b_timing["total_time"],
                "cosine_time_sec": c_timing["total_time"],
                "timing_breakdown": c_timing
            }
            test_users_results.append(user_data)
            
            print(f"\nUser ID: {target_uid}")
            print(f"Original Preferences: {original_prefs}")
            print(f"Baseline Recommendations: {b_recs}")
            print(f"Cosine Recommendations: {c_recs}")
            print(f"Top Similar Users: {user_data['top_similar_users']}")
            print(f"Similarity Scores: {user_data['similarity_scores']}")
            print(f"Recommendation Time:")
            print(f"  - Baseline Total: {b_timing['total_time']*1000:.4f} ms")
            print(f"  - Cosine Total:   {c_timing['total_time']*1000:.4f} ms")
            print(f"      * Hash Lookup:       {c_timing['hash_lookup_time']*1000:.4f} ms")
            print(f"      * Similarity Calc:   {c_timing['similarity_calc_time']*1000:.4f} ms")
            print(f"      * Rec Generation:    {c_timing['rec_gen_time']*1000:.4f} ms")
            print("-" * 80)

        # 5. Aggregate Summary Statistics
        avg_baseline_t = float(np.mean(baseline_times))
        avg_cosine_t = float(np.mean(cosine_times))
        avg_sim_calc_t = float(np.mean(sim_calc_times))
        avg_lookup_t = float(np.mean(lookup_times))
        avg_gen_t = float(np.mean(gen_times))
        avg_max_sim = float(np.mean(all_top_sim_scores))

        summary = {
            "num_users": num_users,
            "num_movies": num_movies,
            "top_n": top_n,
            "top_k_users": top_k_users,
            "avg_baseline_time_sec": avg_baseline_t,
            "avg_cosine_time_sec": avg_cosine_t,
            "avg_sim_calc_time_sec": avg_sim_calc_t,
            "avg_lookup_time_sec": avg_lookup_t,
            "avg_gen_time_sec": avg_gen_t,
            "avg_max_similarity": avg_max_sim,
            "test_users": test_users_results
        }
        
        print("\n" + "=" * 80)
        print(f"AGGREGATE SUMMARY (N={num_users} Users, M={num_movies} Movies)")
        print("=" * 80)
        print(f"Avg Baseline Recommendation Time: {avg_baseline_t*1000:.4f} ms")
        print(f"Avg Cosine Recommendation Time:   {avg_cosine_t*1000:.4f} ms")
        print(f"  - Hash Lookup Time:             {avg_lookup_t*1000:.4f} ms")
        print(f"  - Similarity Calculation Time:  {avg_sim_calc_t*1000:.4f} ms")
        print(f"  - Candidate Scoring & Gen Time: {avg_gen_t*1000:.4f} ms")
        print(f"Avg Top Similar User Score:       {avg_max_sim:.4f}")
        print("=" * 80 + "\n")
        
        return summary

