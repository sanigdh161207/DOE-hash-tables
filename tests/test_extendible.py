import pytest
from extendible_hashing import ExtendibleHashTable
from recommender import RecommenderSystem

def test_extendible_hashing_invariants():
    ext_ht = ExtendibleHashTable(table_size=4, bucket_capacity=2)
    for i in range(100):
        ext_ht.insert(i, [i])
        
    assert len(ext_ht.directory) == (1 << ext_ht.global_depth)
    
    # Check each bucket's local depth invariant
    visited = set()
    for b in ext_ht.directory:
        if id(b) not in visited:
            visited.add(id(b))
            assert b.local_depth <= ext_ht.global_depth

def test_extendible_hashing_search_and_delete():
    ext_ht = ExtendibleHashTable(table_size=4, bucket_capacity=2)
    for i in range(50):
        ext_ht.insert(i, f"val_{i}")
        
    for i in range(50):
        val, _ = ext_ht.search(i)
        assert val == f"val_{i}"
        
    assert ext_ht.delete(10) is True
    val_after, _ = ext_ht.search(10)
    assert val_after is None

def test_recommender_with_extendible_backend():
    dataset = [(1, [1, 2, 3]), (2, [1, 2, 4])]
    rec_sys = RecommenderSystem(table_size=4, strategy_class=ExtendibleHashTable)
    rec_sys.fit(dataset, num_movies=5)
    
    prefs, _ = rec_sys.get_user_preferences(1)
    assert prefs == [1, 2, 3]
