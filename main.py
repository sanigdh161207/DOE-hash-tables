import sys
import argparse
import subprocess

from recommender import RecommenderSystem, HashTableChaining
from data_generator import generate_structured_data
from simulator import PerformanceSimulator
from structures_compare import run_structures_comparison
from hot1_memory_model import calculate_hot1_memory
from score_rubric import calculate_heuristic_score
from pipeline import run_pipeline
import verify

def run_demo():
    print("=== DOE Hash Table Recommender Demo ===")
    dataset = generate_structured_data(num_users=10, num_movies=20, seed=42)
    rec_sys = RecommenderSystem(table_size=13, strategy_class=HashTableChaining)
    rec_sys.fit(dataset, num_movies=20)
    target_user = dataset[0][0]
    all_movies = list(range(1, 21))

    print(f"Target User ID: {target_user}")
    print(f"User Preferences: {dataset[0][1]}")

    recs_base, _ = rec_sys.recommend_movies(target_user, all_movies, top_n=3)
    print(f"Random Baseline Recs: {recs_base}")

    recs_cos, _ = rec_sys.recommend_movies_cosine(target_user, all_movies, top_n=3)
    print(f"Cosine Similarity Recs: {recs_cos}")

    recs_w, _ = rec_sys.recommend_movies_weighted_cosine(target_user, all_movies, top_n=3, weighting="idf")
    print(f"IDF-Weighted Cosine Recs: {recs_w}")


def main():
    parser = argparse.ArgumentParser(description="DOE Hash Table Recommender System Entry Point")
    parser.add_argument("command", choices=[
        "demo", "benchmark", "week6", "quality", "cold-start",
        "weighted", "sparse", "extendible", "pipeline", "scaling",
        "verify", "gui", "all"
    ], help="Command mode to execute")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "verify":
        print("Running verification checks...")
        verify.run_tests()
    elif args.command == "gui":
        print("Launching CustomTkinter GUI Application...")
        import app
        app_instance = app.App()
        app_instance.mainloop()
    elif args.command == "benchmark":
        print("Running Lookup and Collision Benchmarks...")
        PerformanceSimulator.run_lookup_benchmark([10, 50, 100, 500, 1000], strategy="Both")
        PerformanceSimulator.run_collision_experiment([0.25, 0.50, 0.75, 0.90], strategy="Both")
        print("Benchmarks complete.")
    elif args.command == "week6":
        print("Running Week 6 Baseline Benchmarks...")
        PerformanceSimulator.run_lookup_benchmark([10, 50, 100], strategy="Both")
        print("Week 6 complete.")
    elif args.command == "quality":
        print("Executing Quality Experiment (Baseline vs Cosine)...")
        PerformanceSimulator.run_quality_experiment()
        print("Quality experiment complete.")
    elif args.command == "cold-start":
        print("Executing Cold Start Experiment...")
        PerformanceSimulator.run_cold_start_experiment()
        print("Cold start experiment complete.")
    elif args.command == "weighted":
        print("Executing IDF-Weighted Cosine Experiment...")
        PerformanceSimulator.run_weighted_cosine_experiment()
        print("Weighted cosine experiment complete.")
    elif args.command == "sparse":
        print("Executing Sparse Matrix Memory and Timing Experiment...")
        PerformanceSimulator.run_sparse_experiment()
        calculate_hot1_memory()
        print("Sparse experiment complete.")
    elif args.command == "extendible":
        print("Executing Extendible Hashing Experiment...")
        PerformanceSimulator.run_extendible_experiment()
        print("Extendible hashing experiment complete.")
    elif args.command == "pipeline":
        print("Executing Pandas + NumPy ETL Pipeline...")
        run_pipeline(num_users=500, num_movies=100, top_n=5, out_csv="results/top_n_recommendations.csv")
        print("Pipeline complete.")
    elif args.command == "scaling":
        print("Executing Empirical Scaling Benchmark...")
        PerformanceSimulator.run_scaling_experiment()
        print("Scaling experiment complete.")
    elif args.command == "all":
        print("Running ALL Verification Checks and Experiments...")
        verify.run_tests()
        run_demo()
        PerformanceSimulator.run_quality_experiment()
        PerformanceSimulator.run_cold_start_experiment()
        PerformanceSimulator.run_weighted_cosine_experiment()
        PerformanceSimulator.run_sparse_experiment()
        calculate_hot1_memory()
        PerformanceSimulator.run_extendible_experiment()
        run_structures_comparison()
        run_pipeline()
        PerformanceSimulator.run_scaling_experiment()
        PerformanceSimulator.run_inverted_index_experiment()
        score = calculate_heuristic_score()
        print(f"All experiments complete. Final Quality Score: {score['final_score']} / 10.0")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main_args = ["demo"]
        sys.argv.append("demo")
    main()
