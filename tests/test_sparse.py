import pytest
from sparse_recommender import SparseRecommender, build_user_item_matrix
from recommender import RecommenderSystem

def test_sparse_matrix_construction():
    dataset = [(10, [1, 2]), (20, [2, 3])]
    matrix, table, idx_map = build_user_item_matrix(dataset, num_movies=5)
    assert matrix.shape == (2, 6)
    row_idx, _ = table.search(10, record_trace=False)
    assert row_idx == 0

def test_sparse_recommender_equivalence():
    dataset = [
        (1, [1, 2, 3]),
        (2, [1, 2, 4]),
        (3, [5, 6, 7])
    ]
    sparse_rec = SparseRecommender(num_movies=10)
    sparse_rec.fit(dataset)
    
    dense_rec = RecommenderSystem(10)
    dense_rec.fit(dataset, num_movies=10)
    
    all_movies = list(range(1, 11))
    recs_sparse, _ = sparse_rec.recommend_movies(1, all_movies, top_n=1)
    recs_dense, _ = dense_rec.recommend_movies_cosine(1, all_movies, top_n=1)
    assert recs_sparse == recs_dense

def test_sparse_memory_estimation():
    dataset = [(1, [1, 2]), (2, [2, 3])]
    sparse_rec = SparseRecommender(num_movies=5)
    sparse_rec.fit(dataset)
    assert sparse_rec.estimate_csr_memory_bytes() > 0
