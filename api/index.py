import os
import sys
import time
import math
import traceback
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from data_generator import generate_structured_data, get_movie_genres, format_first_n_users
from recommender import (
    HashTableChaining,
    HashTableLinearProbing,
    RecommenderSystem,
)
from extendible_hashing import ExtendibleHashTable
from sparse_recommender import SparseRecommender
from simulator import PerformanceSimulator

app = FastAPI(
    title="Hash Table Recommender Simulator API",
    description="Interactive backend for Hash Table Simulator & Recommender Lab",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel Path Normalization Middleware
@app.middleware("http")
async def vercel_path_fix(request: Request, call_next):
    path = request.scope.get("path", "")
    if path.startswith("/api/index.py"):
        new_path = path[len("/api/index.py"):]
        request.scope["path"] = new_path if new_path else "/"
    return await call_next(request)

# Global Exception Handler for debugging serverless errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    err_msg = str(exc)
    err_trace = traceback.format_exc()
    print(f"Error handling {request.method} {request.url.path}: {err_msg}")
    print(err_trace)
    return JSONResponse(
        status_code=500,
        content={"detail": err_msg, "trace": err_trace, "path": request.url.path}
    )

GENRE_NAMES = {
    0: "Action",
    1: "Comedy",
    2: "Drama",
    3: "Sci-Fi",
    4: "Thriller"
}

# In-memory session state for fast serverless interaction
STATE: Dict[str, Any] = {
    "dataset": None,
    "user_ids": [],
    "hash_table": None,
    "recommender": None,
    "sparse_recommender": None,
    "selected_strategy": "Separate Chaining",
    "dataset_size": 100,
    "target_lf": 0.75,
    "calculated_size": 133,
    "seed": 42
}

def ensure_dataset(n_users: int = 100, seed: int = 42, force: bool = False):
    if force or STATE["dataset"] is None or len(STATE["dataset"]) != n_users or STATE["seed"] != seed:
        dataset = generate_structured_data(num_users=n_users, popularity_skew=1.0, seed=seed)
        STATE["dataset"] = dataset
        STATE["user_ids"] = [u[0] for u in dataset]
        STATE["dataset_size"] = n_users
        STATE["seed"] = seed
    return STATE["dataset"]

def build_hash_table(strategy: str, size_mode: str, custom_size: int, target_lf: float, n_users: int, seed: int = 42):
    dataset = ensure_dataset(n_users=n_users, seed=seed)
    
    if size_mode == "Custom" and custom_size > 0:
        calculated_size = custom_size
    else:
        calculated_size = max(10, int(math.ceil(n_users / max(0.01, target_lf))))

    if strategy == "Extendible Hashing":
        table = ExtendibleHashTable(capacity=4)
        for uid, items in dataset:
            table.insert(uid, items, record_trace=False)
        rec_sys = RecommenderSystem(table_size=calculated_size, strategy_class=HashTableChaining)
        rec_sys.fit(dataset, num_movies=100)
        rec_sys.hash_table = table
    else:
        strat_cls = HashTableLinearProbing if strategy == "Linear Probing" else HashTableChaining
        rec_sys = RecommenderSystem(table_size=calculated_size, strategy_class=strat_cls)
        rec_sys.fit(dataset, num_movies=100)
        table = rec_sys.hash_table

    STATE["hash_table"] = table
    STATE["recommender"] = rec_sys
    try:
        sparse_rec = SparseRecommender(num_movies=100)
        sparse_rec.fit(dataset)
        STATE["sparse_recommender"] = sparse_rec
    except Exception as e:
        print(f"Warning: SparseRecommender initialization skipped: {e}")
        STATE["sparse_recommender"] = None

    STATE["selected_strategy"] = strategy
    STATE["target_lf"] = target_lf
    STATE["calculated_size"] = calculated_size
    
    return table

class GenerateRequest(BaseModel):
    num_users: int = 100
    seed: int = 42

class PopulateRequest(BaseModel):
    strategy: str = "Separate Chaining"
    size_mode: str = "Auto"
    custom_size: int = 150
    target_lf: float = 0.75
    num_users: int = 100
    seed: int = 42

class SearchRequest(BaseModel):
    key: int

class RecommendRequest(BaseModel):
    user_id: int
    top_n: int = 5
    algorithm: str = "Cosine Similarity"

# APIRouter allows mounting both at /api and root /
router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "message": "Hash Table Simulator API is running"}

@router.get("/state")
def get_state():
    has_dataset = STATE["dataset"] is not None
    has_table = STATE["hash_table"] is not None
    return {
        "has_dataset": has_dataset,
        "has_table": has_table,
        "dataset_size": len(STATE["dataset"]) if has_dataset else 0,
        "user_ids": STATE["user_ids"][:50] if has_dataset else [],
        "selected_strategy": STATE["selected_strategy"],
        "calculated_size": STATE["calculated_size"],
        "target_lf": STATE["target_lf"]
    }

@router.post("/generate-data")
def generate_data(req: GenerateRequest):
    dataset = ensure_dataset(n_users=req.num_users, seed=req.seed, force=True)
    genres = get_movie_genres(100, 5, seed=42)
    
    preview = []
    for uid, items in dataset[:10]:
        preview.append({
            "user_id": uid,
            "items_count": len(items),
            "items": items,
            "genre_breakdown": {GENRE_NAMES[g]: sum(1 for m in items if genres.get(m) == g) for g in range(5)}
        })
        
    return {
        "success": True,
        "total_users": len(dataset),
        "seed": req.seed,
        "preview": preview,
        "all_user_ids": [u[0] for u in dataset]
    }

@router.post("/populate")
def populate_table(req: PopulateRequest):
    table = build_hash_table(
        strategy=req.strategy,
        size_mode=req.size_mode,
        custom_size=req.custom_size,
        target_lf=req.target_lf,
        n_users=req.num_users,
        seed=req.seed
    )
    
    total_users = len(STATE["dataset"])
    table_size = table.size if hasattr(table, "size") else len(getattr(table, "directory", []))
    load_factor = round(table.load_factor(), 4) if hasattr(table, "load_factor") else round(total_users / max(1, table_size), 4)
    
    try:
        collisions = table.collision_count()
    except Exception:
        collisions = 0
    
    collision_rate = round((collisions / total_users * 100), 2) if total_users > 0 else 0.0

    avg_bucket_len = 0.0
    max_bucket_len = 0
    memory_bytes = 0
    
    if hasattr(table, "buckets"):
        lens = [len(b) if isinstance(b, list) else (1 if b else 0) for b in table.buckets]
        avg_bucket_len = round(sum(lens) / max(1, len(lens)), 2)
        max_bucket_len = max(lens) if lens else 0
        memory_bytes = table.estimate_memory_bytes()
    elif isinstance(table, ExtendibleHashTable):
        memory_bytes = table.estimate_memory_bytes()

    is_extendible = isinstance(table, ExtendibleHashTable)
    
    if not is_extendible:
        limit = min(table.size, 100)
        buckets_data = []
        for idx in range(limit):
            b = table.buckets[idx]
            items = []
            if b:
                b_list = b if isinstance(b, list) else [b]
                for item in b_list:
                    if item is not None and item != "<TOMBSTONE>":
                        k, v = item
                        items.append({"key": k, "count": len(v), "preview": v[:4]})
            
            buckets_data.append({
                "index": idx,
                "items": items,
                "is_empty": len(items) == 0,
                "is_collision": len(items) > 1
            })
    else:
        buckets_data = {
            "global_depth": table.global_depth,
            "directory_size": len(table.directory),
            "directory": [
                {
                    "dir_index": i,
                    "bin_index": f"{i:0{table.global_depth}b}",
                    "bucket_id": id(b),
                    "local_depth": b.local_depth,
                    "items": [{"key": k, "count": len(v), "preview": v[:4]} for k, v in b.items]
                }
                for i, b in enumerate(table.directory)
            ]
        }

    return {
        "success": True,
        "strategy": req.strategy,
        "stats": {
            "total_users": total_users,
            "table_size": table_size,
            "load_factor": load_factor,
            "collisions": collisions,
            "collision_rate_pct": collision_rate,
            "avg_bucket_len": avg_bucket_len,
            "max_bucket_len": max_bucket_len,
            "memory_bytes": memory_bytes
        },
        "is_extendible": is_extendible,
        "buckets": buckets_data,
        "user_ids": STATE["user_ids"]
    }

@router.post("/search")
def search_key(req: SearchRequest):
    if STATE["hash_table"] is None:
        build_hash_table("Separate Chaining", "Auto", 150, 0.75, 100, 42)

    table = STATE["hash_table"]
    t0 = time.perf_counter()
    val, trace = table.search(req.key, record_trace=True)
    latency_us = round((time.perf_counter() - t0) * 1e6, 2)
    
    found = val is not None
    hash_val = table.hash_function(req.key) if hasattr(table, "hash_function") else None

    return {
        "key": req.key,
        "found": found,
        "value": val if found else None,
        "items_count": len(val) if found and isinstance(val, list) else 0,
        "hash_val": hash_val,
        "latency_us": latency_us,
        "trace": trace
    }

@router.post("/recommend")
def get_recommendations(req: RecommendRequest):
    if STATE["recommender"] is None or STATE["hash_table"] is None:
        build_hash_table("Separate Chaining", "Auto", 150, 0.75, 100, 42)

    recommender = STATE["recommender"]
    sparse_rec = STATE["sparse_recommender"]
    all_movies = list(range(1, 101))
    genres = get_movie_genres(100, 5, seed=42)

    t0 = time.perf_counter()
    algo = req.algorithm
    sources = []
    recs = []

    if algo == "Baseline (Random)":
        recs, _ = recommender.recommend_movies(req.user_id, all_movies, top_n=req.top_n)
        sources = ["random"] * len(recs)
    elif algo == "Cosine Similarity":
        recs, _ = recommender.recommend_movies_cosine(req.user_id, all_movies, top_n=req.top_n)
        sources = ["cosine"] * len(recs)
    elif algo == "IDF-Weighted Cosine":
        recs, _ = recommender.recommend_movies_weighted_cosine(req.user_id, all_movies, top_n=req.top_n, weighting="idf", popularity_alpha=0.5)
        sources = ["idf-cosine"] * len(recs)
    elif algo == "Hybrid Cold-Start":
        recs, sources, _ = recommender.recommend_movies_hybrid(req.user_id, all_movies, top_n=req.top_n)
    elif algo == "Sparse CSR Cosine":
        if sparse_rec:
            recs, _ = sparse_rec.recommend_movies(req.user_id, all_movies, top_n=req.top_n)
            sources = ["sparse-csr"] * len(recs)
        else:
            recs, _ = recommender.recommend_movies_cosine(req.user_id, all_movies, top_n=req.top_n)
            sources = ["fallback-cosine"] * len(recs)
    else:
        recs, _ = recommender.recommend_movies_cosine(req.user_id, all_movies, top_n=req.top_n)
        sources = ["cosine"] * len(recs)

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    items = []
    for rank, (m_id, src) in enumerate(zip(recs, sources), start=1):
        g_id = genres.get(m_id, 0)
        items.append({
            "rank": rank,
            "movie_id": m_id,
            "source": src,
            "genre_id": g_id,
            "genre_name": GENRE_NAMES.get(g_id, "Unknown")
        })

    user_items, _ = STATE["hash_table"].search(req.user_id, record_trace=False)
    user_genres = {}
    if user_items:
        for m in user_items:
            g = GENRE_NAMES.get(genres.get(m, 0), "Other")
            user_genres[g] = user_genres.get(g, 0) + 1

    return {
        "user_id": req.user_id,
        "algorithm": algo,
        "top_n": req.top_n,
        "latency_ms": latency_ms,
        "recommendations": items,
        "user_history": {
            "movie_ids": user_items or [],
            "genre_distribution": user_genres
        }
    }

@router.get("/benchmarks/lookup")
def benchmark_lookup():
    sizes = [10, 50, 100, 500, 1000]
    res = PerformanceSimulator.run_lookup_benchmark(sizes, strategy="Both")
    
    formatted = {}
    for strat, data in res.items():
        formatted[strat] = [
            {
                "size": size,
                "avg_lookup_us": round(avg_t * 1e6, 3),
                "collisions": coll,
                "memory_bytes": mem
            }
            for size, avg_t, coll, mem in data
        ]
    return {"sizes": sizes, "results": formatted}

@router.get("/benchmarks/collision")
def benchmark_collision():
    load_factors = [0.25, 0.50, 0.75, 0.90]
    res = PerformanceSimulator.run_collision_experiment(load_factors, strategy="Both")
    
    formatted = {}
    for strat, data in res.items():
        formatted[strat] = [
            {
                "load_factor": lf,
                "collision_rate_pct": round(rate * 100, 2),
                "avg_lookup_us": round(avg_t * 1e6, 3),
                "memory_bytes": mem
            }
            for lf, rate, avg_t, mem, _ in data
        ]
    return {"load_factors": load_factors, "results": formatted}

@router.get("/benchmarks/numpy")
def benchmark_numpy():
    res = PerformanceSimulator.compare_numpy_performance(5000)
    return {
        "python_list_time_sec": round(res["python_list_time"], 6),
        "numpy_time_sec": round(res["numpy_time"], 6),
        "speedup": round(res["speedup_numpy"], 2)
    }

# Mount both with /api prefix and at root so Vercel routing works regardless of path stripping
app.include_router(router, prefix="/api")
app.include_router(router, prefix="")

# Mount static files for local server.py development
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
if os.path.exists(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
