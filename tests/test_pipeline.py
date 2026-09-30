import os
import pytest
import pandas as pd
from pipeline import load_data, validate, normalise, compute_similarity, generate_top_n, export_csv, run_pipeline

def test_pipeline_end_to_end(tmp_path):
    out_csv = os.path.join(tmp_path, "top_n.csv")
    timings = run_pipeline(num_users=20, num_movies=10, top_n=3, out_csv=out_csv)
    
    assert os.path.exists(out_csv)
    assert timings["total_sec"] > 0.0
    
    df = pd.read_csv(out_csv)
    assert set(df.columns) == {"user_id", "rank", "movie_id", "score", "confidence"}
    assert df["user_id"].nunique() == 20
    assert not df.isnull().values.any()

def test_pipeline_validation():
    raw_df = pd.DataFrame([
        {"user_id": 1, "movie_id": 10, "rating": 5},
        {"user_id": 1, "movie_id": 10, "rating": 5},  # Duplicate
        {"user_id": -1, "movie_id": 10, "rating": 3}, # Invalid User ID
    ])
    clean_df = validate(raw_df)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["user_id"] == 1

def test_normalisation_and_similarity():
    df = pd.DataFrame([
        {"user_id": 1, "movie_id": 1, "rating": 5},
        {"user_id": 2, "movie_id": 1, "rating": 5},
    ])
    clean_df = validate(df)
    pivot, norm_df = normalise(clean_df, method="l2")
    sim_df = compute_similarity(norm_df)
    # Exclude diagonal
    assert sim_df.loc[1, 1] == 0.0
    assert sim_df.loc[1, 2] == pytest.approx(1.0)
