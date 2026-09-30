import pytest
import random
from recommender import RecommenderSystem, HashTableChaining, HashTableLinearProbing, compute_cosine_similarity
from data_generator import generate_structured_data

def test_cosine_similarity_math():
    import numpy as np
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0])
    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0)

def test_recommender_fit_and_search():
    dataset = [(1001, [1, 2, 3]), (1002, [2, 3, 4])]
    rec_sys = RecommenderSystem(10, strategy_class=HashTableChaining)
    rec_sys.fit(dataset, num_movies=10)
    
    prefs, _ = rec_sys.get_user_preferences(1001)
    assert prefs == [1, 2, 3]

def test_recommendation_cosine_correctness():
    dataset = [
        (1, [1, 2, 3]),
        (2, [1, 2, 4]),
        (3, [5, 6, 7])
    ]
    rec_sys = RecommenderSystem(10)
    rec_sys.fit(dataset, num_movies=10)
    
    recs, _ = rec_sys.recommend_movies_cosine(1, [1, 2, 3, 4, 5, 6, 7], top_n=1)
    assert recs == [4]

def test_deterministic_recommendations():
    dataset = generate_structured_data(num_users=20, num_movies=30, seed=42)
    rec_sys1 = RecommenderSystem(13)
    rec_sys1.fit(dataset, num_movies=30)
    
    rec_sys2 = RecommenderSystem(13)
    rec_sys2.fit(dataset, num_movies=30)
    
    uid = dataset[0][0]
    r1, _ = rec_sys1.recommend_movies_cosine(uid, list(range(1, 31)), top_n=3)
    r2, _ = rec_sys2.recommend_movies_cosine(uid, list(range(1, 31)), top_n=3)
    assert r1 == r2

def test_backend_equivalence():
    dataset = generate_structured_data(num_users=20, num_movies=30, seed=42)
    rec_sc = RecommenderSystem(13, strategy_class=HashTableChaining)
    rec_sc.fit(dataset, num_movies=30)
    
    rec_lp = RecommenderSystem(13, strategy_class=HashTableLinearProbing)
    rec_lp.fit(dataset, num_movies=30)
    
    uid = dataset[0][0]
    r_sc, _ = rec_sc.recommend_movies_cosine(uid, list(range(1, 31)), top_n=3)
    r_lp, _ = rec_lp.recommend_movies_cosine(uid, list(range(1, 31)), top_n=3)
    assert r_sc == r_lp

def test_duplicate_user_ids():
    dataset = [(1001, [1, 2]), (1001, [3, 4])]
    rec_sys = RecommenderSystem(10)
    rec_sys.fit(dataset, num_movies=10)
    prefs, _ = rec_sys.get_user_preferences(1001)
    assert prefs == [3, 4]

def test_unknown_user():
    dataset = [(1, [1, 2])]
    rec_sys = RecommenderSystem(10)
    rec_sys.fit(dataset, num_movies=10)
    recs, _ = rec_sys.recommend_movies_cosine(999, list(range(1, 11)), top_n=3)
    assert recs == []
