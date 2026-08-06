from data_generator import generate_user_data, format_first_n_users
from recommender import HashTable, RecommenderSystem, HashTableChaining, HashTableLinearProbing
from simulator import PerformanceSimulator

def run_tests():
    print("Testing data generation...")
    dataset = generate_user_data(10)
    assert len(dataset) == 10
    print("  Data generation OK")

    print("Testing HashTableChaining...")
    ht_sc = HashTableChaining(10)
    trace = ht_sc.insert(105, [1, 2, 3])
    val, search_trace = ht_sc.search(105)
    assert val == [1, 2, 3]
    stats = ht_sc.get_collision_statistics()
    assert stats["users"] == 1
    print("  HashTableChaining OK")

    print("Testing HashTableLinearProbing...")
    ht_lp = HashTableLinearProbing(10)
    trace = ht_lp.insert(105, [1, 2, 3])
    val, search_trace = ht_lp.search(105)
    assert val == [1, 2, 3]
    # Test delete
    deleted = ht_lp.delete(105)
    assert deleted is True
    val_after, _ = ht_lp.search(105)
    assert val_after is None
    # Re-insert
    ht_lp.insert(105, [4, 5])
    val_re, _ = ht_lp.search(105)
    assert val_re == [4, 5]
    print("  HashTableLinearProbing OK")

    print("Testing Simulator with dynamic strategy parameters...")
    lookup_results = PerformanceSimulator.run_lookup_benchmark([10, 50], strategy="Both")
    assert "Separate Chaining" in lookup_results
    assert "Linear Probing" in lookup_results
    assert len(lookup_results["Separate Chaining"]) == 2
    
    collision_results = PerformanceSimulator.run_collision_experiment([0.25, 0.50], strategy="Both")
    assert "Separate Chaining" in collision_results
    assert "Linear Probing" in collision_results
    assert len(collision_results["Separate Chaining"]) == 2
    
    numpy_results = PerformanceSimulator.compare_numpy_performance(100)
    assert "numpy_time" in numpy_results
    print("  Simulator OK")

    print("All backend strategy checks passed successfully!")

if __name__ == "__main__":
    run_tests()
