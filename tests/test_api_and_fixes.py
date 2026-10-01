import pytest
from fastapi.testclient import TestClient
from api.index import app, STATE
from extendible_hashing import ExtendibleHashTable

client = TestClient(app)

def test_extendible_hashing_api_construction_and_populate():
    """FIX #1: Verify Extendible Hashing constructs without TypeError and populates correctly."""
    # Reset state first
    reset_resp = client.post("/api/reset")
    assert reset_resp.status_code == 200

    # Generate data
    gen_resp = client.post("/api/generate-data", json={"num_users": 20, "seed": 42})
    assert gen_resp.status_code == 200
    assert gen_resp.json()["total_users"] == 20

    # Populate Extendible Hashing
    pop_resp = client.post("/api/populate", json={
        "strategy": "Extendible Hashing",
        "num_users": 20,
        "seed": 42
    })
    assert pop_resp.status_code == 200
    data = pop_resp.json()
    assert data["success"] is True
    assert data["strategy"] == "Extendible Hashing"
    assert data["is_extendible"] is True
    assert "buckets" in data
    assert "directory" in data["buckets"]
    assert len(data["buckets"]["directory"]) > 0

    # Search a known key
    user_id = data["user_ids"][0]
    search_resp = client.post("/api/search", json={"key": user_id})
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["found"] is True
    assert search_data["value"] is not None

def test_reset_endpoint_clears_all_state():
    """FIX #3: Verify /api/reset clears dataset, tables, and prevents stale searches."""
    # Populate something first
    client.post("/api/generate-data", json={"num_users": 10, "seed": 42})
    pop_res = client.post("/api/populate", json={"strategy": "Separate Chaining", "num_users": 10, "seed": 42})
    test_uid = pop_res.json()["user_ids"][0]

    # Verify searchable
    assert client.post("/api/search", json={"key": test_uid}).json()["found"] is True

    # Call reset
    reset_resp = client.post("/api/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["success"] is True
    assert reset_resp.json()["state"]["has_table"] is False

    # Check /api/state
    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    assert state_resp.json()["has_table"] is False
    assert state_resp.json()["has_dataset"] is False

    # After reset, search must return 400
    search_after = client.post("/api/search", json={"key": test_uid})
    assert search_after.status_code == 400

    # After reset, recommend must return 400
    rec_after = client.post("/api/recommend", json={"user_id": test_uid, "top_n": 5})
    assert rec_after.status_code == 400

def test_state_invalidation_after_generate_data():
    """FIX #4: Verify generating a new dataset invalidates previously populated table."""
    # Populate initial table
    client.post("/api/generate-data", json={"num_users": 10, "seed": 42})
    client.post("/api/populate", json={"strategy": "Separate Chaining", "num_users": 10, "seed": 42})
    assert STATE["hash_table"] is not None

    # Generate new dataset
    gen_resp = client.post("/api/generate-data", json={"num_users": 25, "seed": 99})
    assert gen_resp.status_code == 200
    assert gen_resp.json()["state_invalidated"] is True

    # Verify backend hash_table is cleared
    assert STATE["hash_table"] is None
    assert STATE["recommender"] is None

    # Search should fail until populated again
    search_resp = client.post("/api/search", json={"key": 1000})
    assert search_resp.status_code == 400

def test_search_trace_schema_and_fields():
    """FIX #2: Verify search trace schema contains both index and bucket across all strategies."""
    strategies = ["Separate Chaining", "Linear Probing", "Extendible Hashing"]
    for strat in strategies:
        client.post("/api/generate-data", json={"num_users": 15, "seed": 42})
        pop_res = client.post("/api/populate", json={"strategy": strat, "num_users": 15, "seed": 42})
        user_id = pop_res.json()["user_ids"][0]

        # Successful search
        res = client.post("/api/search", json={"key": user_id})
        assert res.status_code == 200
        data = res.json()
        assert data["found"] is True
        trace = data["trace"]
        assert len(trace) > 0

        # Check that every step has index, bucket, step_num, action, details
        for step in trace:
            assert "index" in step
            assert "bucket" in step
            assert "action" in step
            assert "details" in step
            assert "step_num" in step

        # Unsuccessful search (nonexistent key)
        res_fail = client.post("/api/search", json={"key": 999999})
        assert res_fail.status_code == 200
        data_fail = res_fail.json()
        assert data_fail["found"] is False
        last_step = data_fail["trace"][-1]
        assert last_step.get("found") is False

def test_extendible_hashing_statistics():
    """FIX #5: Verify Extendible Hashing stats report valid bucket utilization and physical depth."""
    client.post("/api/generate-data", json={"num_users": 40, "seed": 42})
    pop_res = client.post("/api/populate", json={"strategy": "Extendible Hashing", "num_users": 40, "seed": 42})
    assert pop_res.status_code == 200
    stats = pop_res.json()["stats"]

    assert stats["is_extendible"] is True
    assert stats["global_depth"] >= 1
    assert stats["directory_size"] == (1 << stats["global_depth"])
    assert stats["num_unique_buckets"] > 0
    assert stats["bucket_capacity"] == 4
    assert stats["total_physical_capacity"] == stats["num_unique_buckets"] * stats["bucket_capacity"]

    # Bucket utilization must be in [0.0, 1.0]
    utilization = stats["bucket_utilization"]
    assert 0.0 <= utilization <= 1.0
    assert stats["load_factor"] == pytest.approx(utilization)

def test_safe_production_error_response():
    """FIX #6: Verify server does not leak tracebacks or file paths on error."""
    # Send an invalid payload that triggers validation or handling
    res = client.post("/api/search", json={"key": "invalid_not_an_int"})
    assert res.status_code == 422  # Pydantic validation error

    # Nonexistent route
    res_404 = client.get("/api/nonexistent-route-testing")
    assert res_404.status_code == 404
    data_404 = res_404.json()
    assert "detail" in data_404
    # Ensure no traceback or local drive paths in response
    assert "Traceback" not in str(data_404)
    assert "C:\\" not in str(data_404)

def test_root_prefixed_route_compatibility():
    """Verify both /api/health and /health work identically."""
    res_api = client.get("/api/health")
    res_root = client.get("/health")
    assert res_api.status_code == 200
    assert res_root.status_code == 200
    assert res_api.json() == res_root.json()

def test_recommendation_algorithms():
    """Verify all 5 recommendation algorithms return top-N without error."""
    client.post("/api/generate-data", json={"num_users": 20, "seed": 42})
    pop_res = client.post("/api/populate", json={"strategy": "Separate Chaining", "num_users": 20, "seed": 42})
    uid = pop_res.json()["user_ids"][0]

    algos = [
        "Cosine Similarity",
        "IDF-Weighted Cosine",
        "Hybrid Cold-Start",
        "Sparse CSR Cosine",
        "Baseline (Random)"
    ]

    for algo in algos:
        rec_res = client.post("/api/recommend", json={
            "user_id": uid,
            "top_n": 3,
            "algorithm": algo
        })
        assert rec_res.status_code == 200, f"Algorithm {algo} failed"
        recs = rec_res.json()["recommendations"]
        assert len(recs) == 3
        # Ensure user's watched movies are not recommended
        watched = rec_res.json()["user_history"]["movie_ids"]
        for r in recs:
            assert r["movie_id"] not in watched
