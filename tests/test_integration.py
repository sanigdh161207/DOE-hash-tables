import pytest
import random
from recommender import RecommenderSystem
from simulator import PerformanceSimulator
from score_rubric import calculate_heuristic_score

def test_global_rng_side_effect_regression():
    """
    Ensures baseline recommendation does not corrupt global random.seed.
    """
    random.seed(12345)
    val_before = random.random()
    
    dataset = [(100, [1, 2, 3])]
    rec_sys = RecommenderSystem(10)
    rec_sys.fit(dataset, num_movies=10)
    
    random.seed(12345)
    _ = random.random()
    rec_sys.recommend_movies(100, list(range(1, 11)), top_n=3)
    val_after = random.random()
    
    random.seed(12345)
    _ = random.random()
    val_expected = random.random()
    
    assert val_after == pytest.approx(val_expected)

def test_inverted_index_equivalence():
    rows = PerformanceSimulator.run_inverted_index_experiment(num_users=100, num_movies=20, seed=42)
    assert rows[0]["recommendations_identical"] is True

def test_score_rubric():
    res = calculate_heuristic_score()
    assert "final_score" in res
    assert 1.0 <= res["final_score"] <= 10.0
