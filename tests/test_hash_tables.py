import pytest
from recommender import HashTableChaining, HashTableLinearProbing, HashTable

def test_chaining_insert_and_search():
    ht = HashTableChaining(10)
    ht.insert(101, [1, 2, 3])
    val, trace = ht.search(101)
    assert val == [1, 2, 3]
    assert len(trace) > 0

def test_chaining_delete():
    ht = HashTableChaining(10)
    ht.insert(101, [1, 2, 3])
    assert ht.delete(101) is True
    val, _ = ht.search(101)
    assert val is None
    assert ht.delete(101) is False

def test_linear_probing_insert_search_delete():
    ht = HashTableLinearProbing(10)
    ht.insert(101, [4, 5])
    val, trace = ht.search(101)
    assert val == [4, 5]
    
    assert ht.delete(101) is True
    val_after, _ = ht.search(101)
    assert val_after is None
    
    # Re-insert into tombstone slot
    ht.insert(101, [6, 7])
    val_re, _ = ht.search(101)
    assert val_re == [6, 7]

def test_hash_table_resize():
    ht = HashTableChaining(4)
    for i in range(10):
        ht.insert(i, [i])
    ht.resize(10)
    assert ht.size == 10
    for i in range(10):
        val, _ = ht.search(i)
        assert val == [i]

def test_collision_statistics():
    ht = HashTableChaining(5)
    ht.insert(1, [10])
    ht.insert(6, [20])  # Collides with 1 (6 % 5 == 1)
    stats = ht.get_collision_statistics()
    assert stats["users"] == 2
    assert stats["collisions"] == 1
    assert stats["max_bucket_length"] == 2

def test_legacy_hash_table_alias():
    ht = HashTable(10)
    ht.insert(10, [1])
    assert ht.get_load_factor() == 0.1
