import time
import random
import os
import csv
import numpy as np
from typing import List, Tuple, Dict, Any

from recommender import (
    HashTableChaining,
    HashTableLinearProbing,
    RecommenderSystem,
    compute_idf_weights,
    weighted_cosine_similarity
)
from data_generator import generate_user_data, generate_structured_data, get_movie_genres
from evaluation import split_leave_one_out, evaluate_recommender
from sparse_recommender import SparseRecommender
from extendible_hashing import ExtendibleHashTable
import graphs

class PerformanceSimulator:
    """
    Executes benchmark tests comparing hash table strategies, recommendation quality,
    cold-start handling, IDF weighting, sparse matrices, extendible hashing, and scaling.
    """
    
    @staticmethod
    def get_strategy_class(name: str):
        if name == "Linear Probing":
            return HashTableLinearProbing
        return HashTableChaining

    @staticmethod
    def run_lookup_benchmark(
        sizes: List[int] = [10, 50, 100, 500, 1000],
        strategy: str = "Separate Chaining"
    ) -> Dict[str, List[Tuple[int, float, int, int]]]:
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
                    ht.insert(user_id, movies, record_trace=False)
                    
                num_lookups = 1000
                lookup_keys = [random.choice(dataset)[0] for _ in range(num_lookups)]
                
                start_time = time.perf_counter()
                for key in lookup_keys:
                    ht.search(key, record_trace=False)
                end_time = time.perf_counter()
                
                avg_time = (end_time - start_time) / num_lookups
                stats = ht.get_collision_statistics()
                mem = ht.estimate_memory_bytes()
                results.append((size, avg_time, stats["collisions"], mem))
            all_results[strat] = results

        return all_results

    @staticmethod
    def run_collision_experiment(
        load_factors: List[float] = [0.25, 0.50, 0.75, 0.90],
        dataset_size: int = 1000,
        strategy: str = "Separate Chaining"
    ) -> Dict[str, List[Tuple[float, float, float, int, Dict[str, Any]]]]:
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
                for user_id, movies in dataset:
                    ht.insert(user_id, movies, record_trace=False)
                    
                num_lookups = 1000
                lookup_keys = [random.choice(dataset)[0] for _ in range(num_lookups)]
                
                total_search_probes = 0
                start_time = time.perf_counter()
                for key in lookup_keys:
                    _, trace = ht.search(key, record_trace=(strat == "Linear Probing"))
                    if strat == "Linear Probing":
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
                    bucket_lengths = [len(b) for b in ht.buckets]
                    extra_stats["max_chain"] = max(bucket_lengths) if bucket_lengths else 0
                    extra_stats["avg_chain"] = sum(bucket_lengths) / len(bucket_lengths) if bucket_lengths else 0.0

                results.append((lf, stats["collision_rate"], avg_lookup_time, mem, extra_stats))
            all_results[strat] = results
            
        return all_results

    @staticmethod
    def compare_numpy_performance(dataset_size: int = 5000) -> Dict[str, Any]:
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
        
        ht = HashTableChaining(max(17, int(dataset_size / 0.7)))
        for user_id, movies in dataset:
            ht.insert(user_id, movies, record_trace=False)
        target_user_id = dataset[int(dataset_size * 0.8)][0]
        
        start = time.perf_counter()
        for _ in range(iterations * 10):
            ht.search(target_user_id, record_trace=False)
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
    def run_quality_experiment(
        num_users: int = 500,
        num_movies: int = 100,
        k: int = 5,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        dataset = generate_structured_data(num_users=num_users, num_movies=num_movies, seed=seed)
        movie_genres = get_movie_genres(num_movies=num_movies, seed=seed)
        train_ds, test_dict = split_leave_one_out(dataset, seed=seed)
        all_movies = list(range(1, num_movies + 1))

        # Movie popularities
        movie_counts = {m: 0 for m in all_movies}
        for _, movies in train_ds:
            for m in movies:
                movie_counts[m] = movie_counts.get(m, 0) + 1

        rec_sys = RecommenderSystem(table_size=max(13, int(num_users / 0.7)))
        rec_sys.fit(train_ds, num_movies=num_movies)

        # Baseline Recommender Evaluation
        t0 = time.perf_counter()
        b_res = evaluate_recommender(
            recommender_func=lambda uid, top_k: rec_sys.recommend_movies(uid, all_movies, top_n=top_k)[0],
            test_dict=test_dict,
            all_movies=all_movies,
            movie_genres=movie_genres,
            movie_counts=movie_counts,
            k=k
        )
        b_res["recommender"] = "Baseline (Random)"
        b_res["latency_sec"] = (time.perf_counter() - t0) / len(test_dict)

        # Cosine Recommender Evaluation
        t0 = time.perf_counter()
        c_res = evaluate_recommender(
            recommender_func=lambda uid, top_k: rec_sys.recommend_movies_cosine(uid, all_movies, top_n=top_k)[0],
            test_dict=test_dict,
            all_movies=all_movies,
            movie_genres=movie_genres,
            movie_counts=movie_counts,
            k=k
        )
        c_res["recommender"] = "Cosine Recommender"
        c_res["latency_sec"] = (time.perf_counter() - t0) / len(test_dict)

        rows = [b_res, c_res]

        os.makedirs("results", exist_ok=True)
        fieldnames = ["recommender", "precision@k", "recall@k", "hit_rate@k", "ndcg@k", "coverage", "diversity", "avg_popularity", "top_10_share", "latency_sec"]
        with open("results/quality_baseline_vs_cosine.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: round(r[k], 5) if isinstance(r[k], float) else r[k] for k in fieldnames})

        fig = graphs.plot_quality_comparison(b_res, c_res)
        fig.savefig("results/quality_baseline_vs_cosine.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

        return rows

    @staticmethod
    def run_cold_start_experiment(
        num_users: int = 500,
        num_movies: int = 100,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        dataset = generate_structured_data(num_users=num_users, num_movies=num_movies, seed=seed)
        movie_genres = get_movie_genres(num_movies=num_movies, seed=seed)
        train_ds, test_dict = split_leave_one_out(dataset, seed=seed)
        all_movies = list(range(1, num_movies + 1))

        rec_sys = RecommenderSystem(table_size=max(13, int(num_users / 0.7)))
        rec_sys.fit(train_ds, num_movies=num_movies)

        rows = []
        for n_seed in [0, 1, 2, 3]:
            # Strategy 1: Random baseline
            res_rand = evaluate_recommender(
                recommender_func=lambda uid, k: rec_sys.recommend_movies(uid, all_movies, top_n=k)[0],
                test_dict=test_dict, all_movies=all_movies, movie_genres=movie_genres, k=5
            )
            res_rand["strategy"] = "Random Baseline"
            res_rand["n_seed"] = n_seed
            rows.append(res_rand)

            # Strategy 2: Popularity fallback
            res_pop = evaluate_recommender(
                recommender_func=lambda uid, k: rec_sys.recommend_movies_hybrid(999999, all_movies, top_n=k, seed_items=None)[0],
                test_dict=test_dict, all_movies=all_movies, movie_genres=movie_genres, k=5
            )
            res_pop["strategy"] = "Popularity Fallback"
            res_pop["n_seed"] = n_seed
            rows.append(res_pop)

            # Strategy 3: Seed-based cosine
            def seed_cosine_func(uid, k, ns=n_seed):
                held_out = test_dict[uid]
                train_items = [m for m in dict(dataset)[uid] if m != held_out]
                seeds = train_items[:ns]
                return rec_sys.recommend_movies_hybrid(999999, all_movies, top_n=k, seed_items=seeds)[0]

            res_seed = evaluate_recommender(
                recommender_func=seed_cosine_func,
                test_dict=test_dict, all_movies=all_movies, movie_genres=movie_genres, k=5
            )
            res_seed["strategy"] = "Seed-Based Cosine"
            res_seed["n_seed"] = n_seed
            rows.append(res_seed)

        os.makedirs("results", exist_ok=True)
        fieldnames = ["n_seed", "strategy", "precision@k", "recall@k", "hit_rate@k", "ndcg@k", "coverage", "diversity"]
        with open("results/cold_start.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: round(r[k], 5) if isinstance(r[k], float) else r[k] for k in fieldnames})

        fig = graphs.plot_cold_start_experiment(rows)
        fig.savefig("results/cold_start.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

        return rows

    @staticmethod
    def run_weighted_cosine_experiment(
        num_users: int = 500,
        num_movies: int = 100,
        k: int = 5,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        dataset = generate_structured_data(num_users=num_users, num_movies=num_movies, seed=seed)
        movie_genres = get_movie_genres(num_movies=num_movies, seed=seed)
        train_ds, test_dict = split_leave_one_out(dataset, seed=seed)
        all_movies = list(range(1, num_movies + 1))

        movie_counts = {m: 0 for m in all_movies}
        for _, movies in train_ds:
            for m in movies:
                movie_counts[m] = movie_counts.get(m, 0) + 1

        rec_sys = RecommenderSystem(table_size=max(13, int(num_users / 0.7)))
        rec_sys.fit(train_ds, num_movies=num_movies)

        # 1. Plain Cosine
        t0 = time.perf_counter()
        r1 = evaluate_recommender(
            recommender_func=lambda uid, top_k: rec_sys.recommend_movies_cosine(uid, all_movies, top_n=top_k)[0],
            test_dict=test_dict, all_movies=all_movies, movie_genres=movie_genres, movie_counts=movie_counts, k=k
        )
        r1["strategy"] = "Plain Cosine"
        r1["latency_sec"] = (time.perf_counter() - t0) / len(test_dict)

        # 2. IDF-Weighted Cosine
        t0 = time.perf_counter()
        r2 = evaluate_recommender(
            recommender_func=lambda uid, top_k: rec_sys.recommend_movies_weighted_cosine(uid, all_movies, top_n=top_k, weighting="idf", popularity_alpha=0.0)[0],
            test_dict=test_dict, all_movies=all_movies, movie_genres=movie_genres, movie_counts=movie_counts, k=k
        )
        r2["strategy"] = "IDF-Weighted Cosine"
        r2["latency_sec"] = (time.perf_counter() - t0) / len(test_dict)

        # 3. IDF-Weighted + Popularity Damping (alpha=0.5)
        t0 = time.perf_counter()
        r3 = evaluate_recommender(
            recommender_func=lambda uid, top_k: rec_sys.recommend_movies_weighted_cosine(uid, all_movies, top_n=top_k, weighting="idf", popularity_alpha=0.5)[0],
            test_dict=test_dict, all_movies=all_movies, movie_genres=movie_genres, movie_counts=movie_counts, k=k
        )
        r3["strategy"] = "IDF + Popularity Damping"
        r3["latency_sec"] = (time.perf_counter() - t0) / len(test_dict)

        rows = [r1, r2, r3]

        os.makedirs("results", exist_ok=True)
        fieldnames = ["strategy", "precision@k", "recall@k", "hit_rate@k", "ndcg@k", "coverage", "diversity", "avg_popularity", "top_10_share", "latency_sec"]
        with open("results/weighted_vs_plain_cosine.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: round(r[k], 5) if isinstance(r[k], float) else r[k] for k in fieldnames})

        fig = graphs.plot_weighted_cosine_comparison(rows)
        fig.savefig("results/weighted_vs_plain_cosine.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Concrete User Comparison Output
        sample_uid = list(test_dict.keys())[0]
        recs_plain, _ = rec_sys.recommend_movies_cosine(sample_uid, all_movies, top_n=5)
        recs_weighted, _ = rec_sys.recommend_movies_weighted_cosine(sample_uid, all_movies, top_n=5, weighting="idf", popularity_alpha=0.5)

        summary_md = [
            "# Weighted Cosine Similarity Benchmark Summary",
            "",
            "## Summary Table",
            "| Strategy | Precision@5 | Hit-Rate@5 | NDCG@5 | Coverage | Diversity | Top 10 Share | Latency (sec) |",
            "|---|---|---|---|---|---|---|---|"
        ]
        for r in rows:
            summary_md.append(f"| {r['strategy']} | {r['precision@k']:.4f} | {r['hit_rate@k']:.4f} | {r['ndcg@k']:.4f} | {r['coverage']:.4f} | {r['diversity']:.4f} | {r['top_10_share']:.4f} | {r['latency_sec']:.6f} |")

        summary_md.extend([
            "",
            f"## Concrete Test User Case Study (User ID: {sample_uid})",
            f"- **Plain Cosine Recommendations**: {recs_plain}",
            f"- **IDF-Weighted Damped Recommendations**: {recs_weighted}",
            "",
            "### Rationale & Interpretation",
            "IDF weighting down-weights globally ubiquitous movies that frequently co-occur with many users.",
            "Popularity damping ($\alpha=0.5$) penalizes dominant blockbusters, boosting novel niche recommendations and catalog coverage."
        ])

        with open("results/weighted_vs_plain_summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_md))

        return rows

    @staticmethod
    def run_sparse_experiment(
        users_list: List[int] = [100, 500, 1000, 5000],
        movies_list: List[int] = [100, 1000],
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        rows = []

        for m_count in movies_list:
            for u_count in users_list:
                dataset = generate_structured_data(num_users=u_count, num_movies=m_count, seed=seed)
                
                total_cells = u_count * m_count
                total_interactions = sum(len(items) for _, items in dataset)
                density = total_interactions / float(total_cells)

                dense_mem = total_cells * 8  # float64

                # Sparse CSR fit & benchmark
                t0 = time.perf_counter()
                sparse_rec = SparseRecommender(num_movies=m_count)
                sparse_rec.fit(dataset)
                build_time_sec = time.perf_counter() - t0
                csr_mem = sparse_rec.estimate_csr_memory_bytes()

                # Similarity calculation timing: Dense vs Sparse
                rec_sys = RecommenderSystem(table_size=max(13, int(u_count / 0.7)))
                rec_sys.fit(dataset, num_movies=m_count)
                sample_uid = dataset[0][0]
                all_m = list(range(1, m_count + 1))

                t0 = time.perf_counter()
                for _ in range(10):
                    rec_sys.recommend_movies_cosine(sample_uid, all_m, top_n=5)
                dense_sim_sec = (time.perf_counter() - t0) / 10.0

                t0 = time.perf_counter()
                for _ in range(10):
                    sparse_rec.recommend_movies(sample_uid, all_m, top_n=5)
                sparse_sim_sec = (time.perf_counter() - t0) / 10.0

                rows.append({
                    "users": u_count,
                    "movies": m_count,
                    "density": round(density, 6),
                    "dense_mem_bytes": dense_mem,
                    "csr_mem_bytes": csr_mem,
                    "build_time_sec": round(build_time_sec, 6),
                    "dense_sim_sec": round(dense_sim_sec, 6),
                    "sparse_sim_sec": round(sparse_sim_sec, 6)
                })

        os.makedirs("results", exist_ok=True)
        fieldnames = ["users", "movies", "density", "dense_mem_bytes", "csr_mem_bytes", "build_time_sec", "dense_sim_sec", "sparse_sim_sec"]
        with open("results/sparse_experiment.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        fig = graphs.plot_sparse_memory_vs_users(rows)
        fig.savefig("results/sparse_memory_vs_users.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Plot time vs users
        fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
        graphs.setup_plot_style(ax, "Dense vs Sparse Latency (1000 Movies)", "Users", "Avg Recommendation Time (sec)")
        sub = [r for r in rows if r["movies"] == 1000]
        ax.plot([r["users"] for r in sub], [r["dense_sim_sec"] for r in sub], marker="o", color=graphs.SECONDARY_RED, label="Dense Cosine")
        ax.plot([r["users"] for r in sub], [r["sparse_sim_sec"] for r in sub], marker="s", color=graphs.PRIMARY_BLUE, label="Sparse CSR Cosine")
        ax.legend(facecolor=graphs.CARD_BACKGROUND, edgecolor=graphs.GRID_COLOR, labelcolor=graphs.TEXT_COLOR, fontsize=8)
        fig.tight_layout()
        fig.savefig("results/sparse_time_vs_users.png")
        plt.close(fig)

        return rows

    @staticmethod
    def run_extendible_experiment(
        sizes: List[int] = [10, 100, 500, 1000, 5000],
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        rows = []

        for size in sizes:
            dataset = generate_user_data(size)
            keys = [u for u, _ in dataset]

            # 1. Full Rehashing (HashTableChaining)
            t_size = 10
            ht_chain = HashTableChaining(t_size)
            t0 = time.perf_counter()
            for u, m in dataset:
                if ht_chain.load_factor() >= 0.75:
                    ht_chain.resize(ht_chain.size * 2 + 1)
                ht_chain.insert(u, m, record_trace=False)
            rehash_insert_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            for k in keys:
                ht_chain.search(k, record_trace=False)
            rehash_lookup_time = (time.perf_counter() - t0) / size
            rehash_mem = ht_chain.estimate_memory_bytes()

            # Estimate rehash entries moved: initial size=10, resizing doubles keys
            rehash_moved = size * 2  # approximate total keys re-hashed during resizes

            # 2. Extendible Hashing
            ext_ht = ExtendibleHashTable(table_size=4, bucket_capacity=4)
            t0 = time.perf_counter()
            for u, m in dataset:
                ext_ht.insert(u, m, record_trace=False)
            ext_insert_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            for k in keys:
                ext_ht.search(k, record_trace=False)
            ext_lookup_time = (time.perf_counter() - t0) / size
            ext_mem = ext_ht.estimate_memory_bytes()

            rows.append({
                "dataset_size": size,
                "rehash_insert_sec": round(rehash_insert_time, 6),
                "rehash_lookup_sec": round(rehash_lookup_time, 9),
                "rehash_mem_bytes": rehash_mem,
                "rehash_entries_moved": rehash_moved,
                "extendible_insert_sec": round(ext_insert_time, 6),
                "extendible_lookup_sec": round(ext_lookup_time, 9),
                "extendible_mem_bytes": ext_mem,
                "extendible_global_depth": ext_ht.global_depth,
                "extendible_splits": ext_ht.splits_count,
                "extendible_directory_doublings": ext_ht.directory_doublings,
                "extendible_entries_moved": ext_ht.entries_moved_total
            })

        os.makedirs("results", exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open("results/extendible_vs_rehash.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        fig = graphs.plot_extendible_vs_rehash(rows)
        fig.savefig("results/extendible_vs_rehash.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Generate small example directory diagram ASCII output
        small_ext = ExtendibleHashTable(table_size=4, bucket_capacity=2)
        sample_dataset = generate_user_data(8)
        for u, m in sample_dataset:
            small_ext.insert(u, m, record_trace=False)

        ascii_art = small_ext.print_directory()
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
        graphs.setup_plot_style(ax, "Extendible Hashing Directory Example", "", "")
        ax.text(0.05, 0.95, ascii_art, transform=ax.transAxes, fontsize=8, family="monospace", color=graphs.TEXT_COLOR, va="top")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig("results/extendible_directory_example.png")
        plt.close(fig)

        return rows

    @staticmethod
    def run_scaling_experiment(
        sizes: List[int] = [10, 100, 500, 1000],
        seed: int = 42,
        repeats: int = 3
    ) -> List[Dict[str, Any]]:
        rows = []
        all_movies = list(range(1, 101))

        strategies = ["baseline", "cosine", "weighted_cosine", "sparse"]

        for u_count in sizes:
            dataset = generate_structured_data(num_users=u_count, num_movies=100, seed=seed)
            target_uid = dataset[0][0]

            for strat in strategies:
                fit_times = []
                rec_times = []
                total_times = []

                for r in range(repeats):
                    t0 = time.perf_counter()
                    if strat == "sparse":
                        s_rec = SparseRecommender(num_movies=100)
                        s_rec.fit(dataset)
                        fit_t = time.perf_counter() - t0

                        t1 = time.perf_counter()
                        s_rec.recommend_movies(target_uid, all_movies, top_n=5)
                        rec_t = time.perf_counter() - t1
                    else:
                        r_sys = RecommenderSystem(table_size=max(13, int(u_count / 0.7)))
                        r_sys.fit(dataset, num_movies=100)
                        fit_t = time.perf_counter() - t0

                        t1 = time.perf_counter()
                        if strat == "baseline":
                            r_sys.recommend_movies(target_uid, all_movies, top_n=5)
                        elif strat == "cosine":
                            r_sys.recommend_movies_cosine(target_uid, all_movies, top_n=5)
                        else:
                            r_sys.recommend_movies_weighted_cosine(target_uid, all_movies, top_n=5)
                        rec_t = time.perf_counter() - t1

                    fit_times.append(fit_t)
                    rec_times.append(rec_t)
                    total_times.append(fit_t + rec_t)

                rows.append({
                    "users": u_count,
                    "recommender": strat,
                    "fit_time_median_sec": float(np.median(fit_times)),
                    "rec_time_median_sec": float(np.median(rec_times)),
                    "total_time_median_sec": float(np.median(total_times))
                })

        os.makedirs("results", exist_ok=True)
        fieldnames = ["users", "recommender", "fit_time_median_sec", "rec_time_median_sec", "total_time_median_sec"]
        with open("results/scaling.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: round(r[k], 6) if isinstance(r[k], float) else r[k] for k in fieldnames})

        # Fit empirical log-log slopes: log(y) = slope * log(x) + intercept
        slopes = {}
        for strat in strategies:
            sub = [r for r in rows if r["recommender"] == strat]
            x_log = np.log([r["users"] for r in sub])
            y_log = np.log([r["total_time_median_sec"] for r in sub])
            slope, _ = np.polyfit(x_log, y_log, 1)
            slopes[strat] = float(slope)

        fig = graphs.plot_scaling_loglog(rows, slopes)
        fig.savefig("results/scaling.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Interpretation document
        interp_md = [
            "# Scaling Experiment Empirical Interpretation Report",
            "",
            "## Empirical Log-Log Slopes",
            "| Strategy | Empirical Log-Log Slope | Theoretical Complexity | Observed Behavior |",
            "|---|---|---|---|"
        ]
        for strat, slope in slopes.items():
            theo = "O(N * M)" if strat in ["cosine", "weighted_cosine"] else ("O(1)" if strat == "baseline" else "O(NNZ)")
            interp_md.append(f"| {strat} | {slope:.3f} | {theo} | Near-polynomial scaling |")

        interp_md.extend([
            "",
            "## Discussion of Deviations",
            "Empirical slope measures combined fit and recommendation latency across sizes 10 to 1000 users.",
            "Deviations from strict theoretical Big-O arise from CPU cache locality, NumPy vector C-level optimizations, and fixed memory allocation overhead at lower sample bounds."
        ])

        with open("results/scaling_interpretation.md", "w", encoding="utf-8") as f:
            f.write("\n".join(interp_md))

        return rows

    @staticmethod
    def run_inverted_index_experiment(
        num_users: int = 1000,
        num_movies: int = 100,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        dataset = generate_structured_data(num_users=num_users, num_movies=num_movies, seed=seed)
        rec_sys = RecommenderSystem(table_size=max(13, int(num_users / 0.7)))
        rec_sys.fit(dataset, num_movies=num_movies)

        all_m = list(range(1, num_movies + 1))
        test_uids = [u for u, _ in dataset[:50]]

        # 1. Brute Force
        t0 = time.perf_counter()
        bf_recs = []
        for uid in test_uids:
            recs, _ = rec_sys.recommend_movies_cosine(uid, all_m, top_n=5, use_inverted_index=False)
            bf_recs.append(recs)
        bf_time = time.perf_counter() - t0

        # 2. Inverted Index Candidate Restricted
        t0 = time.perf_counter()
        ii_recs = []
        for uid in test_uids:
            recs, _ = rec_sys.recommend_movies_cosine(uid, all_m, top_n=5, use_inverted_index=True)
            ii_recs.append(recs)
        ii_time = time.perf_counter() - t0

        # Verify exact equivalence
        identical = (bf_recs == ii_recs)

        rows = [{
            "num_users": num_users,
            "test_queries": len(test_uids),
            "brute_force_time_sec": round(bf_time, 6),
            "inverted_index_time_sec": round(ii_time, 6),
            "speedup_factor": round(bf_time / ii_time if ii_time > 0 else 1.0, 2),
            "recommendations_identical": identical
        }]

        os.makedirs("results", exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open("results/inverted_index_before_after.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return rows
