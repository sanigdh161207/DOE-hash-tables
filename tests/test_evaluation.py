import pytest
from evaluation import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    catalog_coverage,
    intra_list_diversity,
    split_leave_one_out
)

def test_precision_and_recall_at_k():
    recs = [1, 2, 3, 4, 5]
    relevant = {2, 5, 8}
    assert precision_at_k(recs, relevant, 5) == pytest.approx(2.0 / 5.0)
    assert recall_at_k(recs, relevant, 5) == pytest.approx(2.0 / 3.0)

def test_hit_rate_at_k():
    recs = [1, 2, 3]
    assert hit_rate_at_k(recs, {2}, 3) == 1.0
    assert hit_rate_at_k(recs, {99}, 3) == 0.0

def test_ndcg_at_k():
    recs = [1, 2, 3]
    relevant = {1}
    # Rank 1 hit: DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> NDCG = 1.0
    assert ndcg_at_k(recs, relevant, 3) == pytest.approx(1.0)
    
    recs_rank2 = [2, 1, 3]
    # Rank 2 hit: DCG = 1/log2(3) = 0.6309, IDCG = 1.0 -> NDCG = 0.6309
    assert ndcg_at_k(recs_rank2, relevant, 3) < 1.0

def test_catalog_coverage():
    all_recs = [[1, 2], [2, 3], [3, 4]]
    assert catalog_coverage(all_recs, 10) == pytest.approx(4.0 / 10.0)

def test_intra_list_diversity():
    movie_genres = {1: 0, 2: 0, 3: 1, 4: 2}
    recs_same = [1, 2]
    assert intra_list_diversity(recs_same, movie_genres) == 0.0
    recs_diff = [1, 3]
    assert intra_list_diversity(recs_diff, movie_genres) == 1.0

def test_split_leave_one_out():
    dataset = [(1, [10, 20, 30]), (2, [40])]
    train_ds, test_dict = split_leave_one_out(dataset, seed=42)
    assert len(train_ds) == 2
    assert 1 in test_dict
    assert 2 not in test_dict
    assert len(dict(train_ds)[1]) == 2
