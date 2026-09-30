import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from data_generator import generate_structured_ratings

def load_data(
    input_csv: Optional[str] = None,
    num_users: int = 500,
    num_movies: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Loads interaction data from an input CSV file or generates structured synthetic data.
    Expected DataFrame schema: [user_id, movie_id, rating].
    """
    if input_csv and os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        required_cols = {"user_id", "movie_id", "rating"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input CSV missing required columns: {required_cols - set(df.columns)}")
        return df

    # Generate structured synthetic ratings
    ratings_dict = generate_structured_ratings(num_users=num_users, num_movies=num_movies, seed=seed)
    rows = []
    for user_id, movies_map in ratings_dict.items():
        for movie_id, rating in movies_map.items():
            rows.append({"user_id": user_id, "movie_id": movie_id, "rating": rating})

    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates DataFrame: removes duplicates, drops NaNs, enforces positive integer IDs and ratings 1-5,
    and sorts deterministically by user_id and movie_id.
    """
    df_clean = df.dropna().drop_duplicates(subset=["user_id", "movie_id"])
    df_clean = df_clean[(df_clean["user_id"] > 0) & (df_clean["movie_id"] > 0)]
    df_clean = df_clean[(df_clean["rating"] >= 1) & (df_clean["rating"] <= 5)]
    df_clean = df_clean.sort_values(by=["user_id", "movie_id"]).reset_index(drop=True)
    return df_clean


def normalise(df: pd.DataFrame, method: str = "l2") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs user-item matrix and applies normalization.
    Supported methods: 'l2', 'mean_center', 'minmax'.
    """
    pivot = df.pivot(index="user_id", columns="movie_id", values="rating").fillna(0.0)
    matrix = pivot.values.astype(np.float64)

    if method == "mean_center":
        # Mean center non-zero entries
        row_means = np.true_divide(matrix.sum(1), (matrix != 0).sum(1) + 1e-9)
        norm_matrix = np.where(matrix != 0, matrix - row_means[:, np.newaxis], 0.0)
    elif method == "minmax":
        # MinMax scale [1, 5] -> [0, 1]
        norm_matrix = np.where(matrix != 0, (matrix - 1.0) / 4.0, 0.0)
    else:
        # Default: L2 row normalization
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        norm_matrix = matrix / row_norms

    norm_df = pd.DataFrame(norm_matrix, index=pivot.index, columns=pivot.columns)
    return pivot, norm_df


def compute_similarity(norm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes vectorized NumPy user cosine similarity matrix and excludes diagonal self-similarity.
    """
    norm_matrix = norm_df.values
    dot_prod = np.dot(norm_matrix, norm_matrix.T)
    row_norms = np.linalg.norm(norm_matrix, axis=1, keepdims=True)
    norm_outer = np.dot(row_norms, row_norms.T)
    norm_outer[norm_outer == 0.0] = 1.0
    
    sim_matrix = dot_prod / norm_outer
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
    
    # Exclude diagonal self-similarity
    np.fill_diagonal(sim_matrix, 0.0)

    sim_df = pd.DataFrame(sim_matrix, index=norm_df.index, columns=norm_df.index)
    return sim_df


def generate_top_n(
    pivot_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    top_n: int = 5,
    top_k_users: int = 10
) -> pd.DataFrame:
    """
    Generates Top-N recommendations with calculated confidence scores.
    Confidence = (candidate support from contributing neighbors) / (sum of top-k neighbor similarities).
    """
    users = pivot_df.index.tolist()
    movies = pivot_df.columns.tolist()
    user_item_arr = pivot_df.values
    sim_arr = sim_df.values

    results = []

    for u_idx, user_id in enumerate(users):
        user_sims = sim_arr[u_idx]
        # Get indices of top-k most similar users
        top_k_indices = np.argsort(-user_sims)[:top_k_users]
        top_k_sims = user_sims[top_k_indices]
        sum_sims = np.sum(top_k_sims)
        if sum_sims <= 0.0:
            sum_sims = 1.0

        seen_movie_indices = np.where(user_item_arr[u_idx] > 0)[0]
        seen_movie_ids = set(movies[m_idx] for m_idx in seen_movie_indices)

        candidate_scores = {}
        candidate_support = {}

        for k_idx, neighbor_idx in enumerate(top_k_indices):
            sim = top_k_sims[k_idx]
            if sim <= 0.0:
                continue
            neighbor_ratings = user_item_arr[neighbor_idx]
            interacting_m_indices = np.where(neighbor_ratings > 0)[0]
            
            for m_idx in interacting_m_indices:
                m_id = movies[m_idx]
                if m_id not in seen_movie_ids:
                    candidate_scores[m_id] = candidate_scores.get(m_id, 0.0) + sim * neighbor_ratings[m_idx]
                    candidate_support[m_id] = candidate_support.get(m_id, 0.0) + sim

        # Rank candidates deterministically by score descending, movie_id ascending
        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]

        for rank, (m_id, score) in enumerate(ranked, start=1):
            conf = float(np.clip(candidate_support[m_id] / sum_sims, 0.0, 1.0))
            results.append({
                "user_id": user_id,
                "rank": rank,
                "movie_id": m_id,
                "score": round(float(score), 4),
                "confidence": round(conf, 4)
            })

    return pd.DataFrame(results)


def export_csv(df: pd.DataFrame, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def run_pipeline(
    input_csv: Optional[str] = None,
    num_users: int = 500,
    num_movies: int = 100,
    top_n: int = 5,
    out_csv: str = "results/top_n_recommendations.csv"
) -> Dict[str, float]:
    timings = {}

    t0 = time.perf_counter()
    raw_df = load_data(input_csv=input_csv, num_users=num_users, num_movies=num_movies)
    timings["load_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clean_df = validate(raw_df)
    timings["validate_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pivot_df, norm_df = normalise(clean_df, method="l2")
    timings["normalise_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim_df = compute_similarity(norm_df)
    timings["similarity_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    top_n_df = generate_top_n(pivot_df, sim_df, top_n=top_n)
    timings["recommendation_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    export_csv(top_n_df, out_csv)
    timings["export_sec"] = time.perf_counter() - t0

    timings["total_sec"] = sum(timings.values())

    # Export sample input ratings if generating synthetic data
    if not input_csv:
        export_csv(clean_df, "results/sample_input_ratings.csv")

    # Export timings
    timings_df = pd.DataFrame([timings])
    export_csv(timings_df, "results/pipeline_timings.csv")

    return timings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandas + NumPy Recommendation Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to input ratings CSV")
    parser.add_argument("--users", type=int, default=500, help="Number of users if synthetic")
    parser.add_argument("--movies", type=int, default=100, help="Number of movies")
    parser.add_argument("--top-n", type=int, default=5, help="Top N recommendations per user")
    parser.add_argument("--out", type=str, default="results/top_n_recommendations.csv", help="Output CSV path")

    args = parser.parse_args()

    print(f"Executing Pandas + NumPy Pipeline (Users={args.users}, Movies={args.movies}, Top-N={args.top_n})...")
    timings = run_pipeline(
        input_csv=args.input,
        num_users=args.users,
        num_movies=args.movies,
        top_n=args.top_n,
        out_csv=args.out
    )
    print("Pipeline Execution Completed Successfully!")
    print(f"  - Total Elapsed Time: {timings['total_sec']:.4f} sec")
    print(f"  - Results exported to: {args.out}")
