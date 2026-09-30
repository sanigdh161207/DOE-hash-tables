import math
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional, Set

def split_leave_one_out(
    dataset: List[Tuple[int, List[int]]],
    seed: int = 42
) -> Tuple[List[Tuple[int, List[int]]], Dict[int, int]]:
    """
    Splits interaction dataset using leave-one-out strategy.
    For users with >= 2 items, holds out 1 item for testing and retains the rest for training.
    Returns (train_dataset, test_dict) where test_dict maps user_id -> held_out_movie_id.
    """
    import random
    rng = random.Random(seed)
    
    train_dataset = []
    test_dict = {}
    
    for user_id, movies in dataset:
        if len(movies) >= 2:
            test_item = rng.choice(movies)
            train_items = [m for m in movies if m != test_item]
            train_dataset.append((user_id, train_items))
            test_dict[user_id] = test_item
        else:
            train_dataset.append((user_id, list(movies)))
            
    return train_dataset, test_dict


def precision_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    """
    Precision@K = (number of relevant recommended items in top-K) / K
    """
    if k <= 0:
        return 0.0
    top_k = recs[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / float(k)


def recall_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    """
    Recall@K = (number of relevant recommended items in top-K) / total relevant items
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = recs[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / float(len(relevant))


def hit_rate_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    """
    Hit Rate@K = 1.0 if at least one relevant item is present in top-K recommendations, else 0.0.
    """
    top_k = recs[:k]
    return 1.0 if any(m in relevant for m in top_k) else 0.0


def ndcg_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    IDCG@K = sum_{i=1}^min(|relevant|, k) 1 / log2(i + 1)
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = recs[:k]
    
    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 2)
            
    idcg = 0.0
    ideal_hits = min(len(relevant), k)
    for idx in range(ideal_hits):
        idcg += 1.0 / math.log2(idx + 2)
        
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recommendations: List[List[int]], total_catalog_size: int) -> float:
    """
    Catalog Coverage = unique recommended items across all users / total catalog items.
    """
    if total_catalog_size <= 0:
        return 0.0
    unique_recs = set()
    for rec_list in all_recommendations:
        unique_recs.update(rec_list)
    return len(unique_recs) / float(total_catalog_size)


def intra_list_diversity(recs: List[int], movie_genres: Dict[int, int]) -> float:
    """
    Intra-List Diversity (ILD) based on genre mismatch distance:
    Distance(i, j) = 1.0 if genre(i) != genre(j) else 0.0.
    ILD = Average pairwise distance among top recommendations.
    """
    n = len(recs)
    if n <= 1:
        return 0.0
    pairs = 0
    dissimilarity_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            g1 = movie_genres.get(recs[i], -1)
            g2 = movie_genres.get(recs[j], -2)
            if g1 != g2:
                dissimilarity_sum += 1.0
    return dissimilarity_sum / float(pairs) if pairs > 0 else 0.0


def popularity_bias(
    all_recommendations: List[List[int]],
    movie_counts: Dict[int, int],
    top_n_popular_count: int = 10
) -> Dict[str, float]:
    """
    Computes popularity bias metrics:
    - avg_popularity: Average interaction count of recommended items.
    - top_pct_share: Percentage of recommendations belonging to top N most popular movies.
    """
    if not all_recommendations:
        return {"avg_popularity": 0.0, "top_10_share": 0.0}
        
    all_recs_flat = [m for recs in all_recommendations for m in recs]
    if not all_recs_flat:
        return {"avg_popularity": 0.0, "top_10_share": 0.0}

    sorted_by_pop = sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)
    top_popular_items = set(m for m, _ in sorted_by_pop[:top_n_popular_count])

    avg_pop = sum(movie_counts.get(m, 0) for m in all_recs_flat) / float(len(all_recs_flat))
    top_share = sum(1 for m in all_recs_flat if m in top_popular_items) / float(len(all_recs_flat))

    return {
        "avg_popularity": avg_pop,
        "top_10_share": top_share
    }


def evaluate_recommender(
    recommender_func: Callable[[int, int], List[int]],
    test_dict: Dict[int, int],
    all_movies: List[int],
    movie_genres: Optional[Dict[int, int]] = None,
    movie_counts: Optional[Dict[int, int]] = None,
    k: int = 5
) -> Dict[str, float]:
    """
    Evaluates a recommendation function using leave-one-out evaluation.
    recommender_func signature: func(user_id, top_k) -> List[movie_id]
    """
    precisions = []
    recalls = []
    hit_rates = []
    ndcgs = []
    all_recs = []
    diversities = []

    for user_id, held_out in test_dict.items():
        recs = recommender_func(user_id, k)
        all_recs.append(recs)
        relevant = {held_out}

        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        hit_rates.append(hit_rate_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

        if movie_genres:
            diversities.append(intra_list_diversity(recs, movie_genres))

    cov = catalog_coverage(all_recs, len(all_movies))
    pop_metrics = popularity_bias(all_recs, movie_counts or {}) if movie_counts else {"avg_popularity": 0.0, "top_10_share": 0.0}

    return {
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "hit_rate@k": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": cov,
        "diversity": float(np.mean(diversities)) if diversities else 0.0,
        "avg_popularity": pop_metrics["avg_popularity"],
        "top_10_share": pop_metrics["top_10_share"]
    }
