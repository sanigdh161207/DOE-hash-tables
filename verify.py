from data_generator import generate_user_data, format_first_n_users
from recommender import (
    HashTable,
    RecommenderSystem,
    HashTableChaining,
    HashTableLinearProbing,
    cosine_similarity
)
from simulator import PerformanceSimulator
import numpy as np

def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED REPOSITORY VERIFICATION TESTS")
    print("=" * 60)

    # 1. Data Generation Test
    print("1. Testing data generation...")
    dataset = generate_user_data(10)
    assert len(dataset) == 10
    print("   Data generation OK")

    # 2. HashTableChaining Test
    print("2. Testing HashTableChaining...")
    ht_sc = HashTableChaining(10)
    trace = ht_sc.insert(105, [1, 2, 3])
    val, search_trace = ht_sc.search(105)
    assert val == [1, 2, 3], "HashTableChaining search failed"
    stats = ht_sc.get_collision_statistics()
    assert stats["users"] == 1
    # Test delete
    assert ht_sc.delete(105) is True
    val_after, _ = ht_sc.search(105)
    assert val_after is None
    # Test resize
    ht_sc.insert(101, [1])
    ht_sc.insert(102, [2])
    ht_sc.resize(25)
    assert ht_sc.search(101)[0] == [1]
    assert ht_sc.search(102)[0] == [2]
    print("   HashTableChaining (insert, search, delete, resize) OK")

    # 3. HashTableLinearProbing Test
    print("3. Testing HashTableLinearProbing...")
    ht_lp = HashTableLinearProbing(10)
    trace = ht_lp.insert(105, [1, 2, 3])
    val, search_trace = ht_lp.search(105)
    assert val == [1, 2, 3], "HashTableLinearProbing search failed"
    # Test delete with tombstone
    deleted = ht_lp.delete(105)
    assert deleted is True, "HashTableLinearProbing deletion failed"
    val_after, _ = ht_lp.search(105)
    assert val_after is None
    # Re-insert into tombstone/slot
    ht_lp.insert(105, [4, 5])
    val_re, _ = ht_lp.search(105)
    assert val_re == [4, 5]
    # Test resize
    ht_lp.insert(205, [6, 7])
    ht_lp.resize(25)
    assert ht_lp.search(105)[0] == [4, 5]
    assert ht_lp.search(205)[0] == [6, 7]
    print("   HashTableLinearProbing (insert, search, delete/tombstone, resize) OK")

    # 4. PerformanceSimulator Benchmarks Test
    print("4. Testing PerformanceSimulator benchmarks...")
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
    print("   Simulator benchmarks OK")

    # 5. Cosine Similarity Correctness Tests
    print("5. Testing cosine_similarity...")
    # Test identical vectors produce similarity ~ 1.0
    v1 = np.array([1, 0, 1, 1, 0], dtype=np.float64)
    sim_ident = cosine_similarity(v1, v1)
    assert abs(sim_ident - 1.0) < 1e-6, f"Expected 1.0, got {sim_ident}"
    print("   - Identical vectors produce similarity ~ 1.0 OK")

    # Test orthogonal vectors produce similarity ~ 0.0
    v_orth1 = np.array([1, 0, 1, 0], dtype=np.float64)
    v_orth2 = np.array([0, 1, 0, 1], dtype=np.float64)
    sim_orth = cosine_similarity(v_orth1, v_orth2)
    assert abs(sim_orth - 0.0) < 1e-6, f"Expected 0.0, got {sim_orth}"
    print("   - Orthogonal vectors produce similarity ~ 0.0 OK")

    # Test zero vectors do not crash and return 0.0
    v_zero = np.zeros(5, dtype=np.float64)
    assert cosine_similarity(v_zero, v1) == 0.0, "Zero vector comparison should return 0.0"
    assert cosine_similarity(v1, v_zero) == 0.0, "Zero vector comparison should return 0.0"
    assert cosine_similarity(v_zero, v_zero) == 0.0, "Zero vector pair should return 0.0"
    print("   - Zero vectors handled safely without crash OK")

    # 6. Global Movie Vocabulary & Vector Dimensions Test
    print("6. Testing Global Movie Vocabulary & Preference Vectors...")
    vocab = [1, 2, 3, 4, 5]
    test_ds = [(101, [1, 2, 5]), (102, [2, 3, 4])]
    rec_vocab = RecommenderSystem(table_size=10)
    rec_vocab.fit(test_ds, movie_vocab=vocab)
    assert rec_vocab.movie_vocab == vocab
    vec_101 = rec_vocab.user_to_vector([1, 2, 5])
    assert len(vec_101) == len(vocab)
    assert np.array_equal(vec_101, np.array([1.0, 1.0, 0.0, 0.0, 1.0]))
    print("   - Global movie vocabulary & vector dimensions OK")

    # 7. Target User Excluded From Similarity Results Test
    print("7. Testing User-User Similarity (Target user exclusion)...")
    rec_sim = RecommenderSystem(table_size=20)
    small_ds = generate_user_data(15, 20)
    rec_sim.fit(small_ds, movie_vocab=list(range(1, 21)))
    target_uid = small_ds[0][0]
    sims = rec_sim.compute_user_similarities(target_uid)
    assert len(sims) == len(small_ds) - 1, f"Expected {len(small_ds) - 1} similarities, got {len(sims)}"
    assert all(uid != target_uid for uid, _ in sims), "Target user must be excluded from similarities!"
    # Ensure descending sort
    for i in range(len(sims) - 1):
        assert sims[i][1] >= sims[i+1][1], "Similarities must be sorted in descending order!"
    print("   - Target user excluded and similarity sorted descending OK")

    # 8. Recommendations Do Not Contain Already-Liked Movies Test
    print("8. Testing Recommendation Uniqueness vs User Preferences...")
    exp_ds = generate_user_data(50, 30)
    recommender = RecommenderSystem(table_size=75)
    recommender.fit(exp_ds, movie_vocab=list(range(1, 31)))
    test_uid = exp_ds[0][0]
    liked_movies, _ = recommender.get_user_preferences(test_uid)
    liked_set = set(liked_movies)

    # Cosine recommendations check
    cos_recs, cos_timing, _, _ = recommender.recommend_movies_cosine(test_uid, top_n=5)
    assert len(cos_recs) <= 5
    assert all(m not in liked_set for m in cos_recs), f"Cosine recs contain already-liked movie: {cos_recs} vs {liked_set}"
    assert "hash_lookup_time" in cos_timing
    assert "similarity_calc_time" in cos_timing
    assert "rec_gen_time" in cos_timing
    assert "total_time" in cos_timing
    print("   - Cosine recommendations exclude already-liked movies OK")

    # Baseline recommendations check
    base_recs, base_timing, _ = recommender.recommend_movies_baseline(test_uid, top_n=5)
    assert len(base_recs) <= 5
    assert all(m not in liked_set for m in base_recs), f"Baseline recs contain already-liked movie: {base_recs} vs {liked_set}"
    assert "total_time" in base_timing
    print("   - Baseline recommendations exclude already-liked movies OK")

    # 9. Cosine Recommendations Determinism Test
    print("9. Testing Determinism of Cosine Recommendations...")
    runs = [recommender.recommend_movies_cosine(test_uid, top_n=5)[0] for _ in range(5)]
    for run_rec in runs[1:]:
        assert run_rec == runs[0], f"Cosine recommendations not deterministic! {runs[0]} vs {run_rec}"
    print("   - Cosine recommendations are strictly deterministic OK")

    # 10. Baseline Preservation & Compatibility Test
    print("10. Testing Baseline Preservation & Legacy recommend_movies wrapper...")
    legacy_recs, _ = recommender.recommend_movies(test_uid, all_movies=recommender.movie_vocab, top_n=5)
    assert legacy_recs == base_recs, "recommend_movies legacy wrapper must match recommend_movies_baseline!"
    print("   - Baseline preservation & legacy compatibility OK")

    # 11. Cross-Verification with Linear Probing
    print("11. Testing Recommender with Linear Probing Hash Table...")
    rec_lp = RecommenderSystem(table_size=100, strategy_class=HashTableLinearProbing)
    rec_lp.fit(exp_ds, movie_vocab=list(range(1, 31)))
    lp_cos_recs, _, _, _ = rec_lp.recommend_movies_cosine(test_uid, top_n=5)
    assert lp_cos_recs == cos_recs, "Cosine recommendations should produce identical results across both hash table backends"
    print("   - Recommender with Linear Probing backend OK")

    print("\n" + "=" * 60)
    print("ALL AUTOMATED TESTS PASSED SUCCESSFULLY! (11/11 Checks OK)")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

