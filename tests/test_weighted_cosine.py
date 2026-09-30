import pytest
import numpy as np
from recommender import compute_idf_weights, weighted_cosine_similarity, RecommenderSystem

def test_idf_weights_calculation():
    dataset = [(1, [1, 2]), (2, [1])]
    idf = compute_idf_weights(dataset, num_movies=2)
    # Movie 1 appears in 2 users, Movie 2 appears in 1 user -> IDF(2) > IDF(1)
    assert idf[2] > idf[1]

def test_weighted_cosine_zero_vector_safety():
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 2.0])
    weights = np.array([1.0, 1.0])
    assert weighted_cosine_similarity(v1, v2, weights) == 0.0

def test_recommend_movies_weighted_cosine():
    dataset = [
        (1, [1, 2, 3]),
        (2, [1, 2, 4]),
        (3, [1, 5, 6])
    ]
    rec_sys = RecommenderSystem(10)
    rec_sys.fit(dataset, num_movies=10)
    
    recs, _ = rec_sys.recommend_movies_weighted_cosine(1, list(range(1, 11)), top_n=1, weighting="idf")
    assert len(recs) == 1
