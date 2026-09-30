import random
import sys
import math
import time
import numpy as np
from typing import Any, List, Tuple, Optional, Dict, Set, Union

class AbstractHashTable:
    """
    Abstract Base Class defining the interface for Hash Table implementations.
    """
    def __init__(self, size: int = 10):
        self.size = size
        self.num_keys = 0

    def hash_function(self, key: Any) -> int:
        try:
            val = int(key)
        except (ValueError, TypeError):
            val = hash(key)
        return abs(val) % self.size

    def insert(self, key: Any, value: Any, record_trace: bool = True) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search(self, key: Any, record_trace: bool = True) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        raise NotImplementedError

    def delete(self, key: Any) -> bool:
        raise NotImplementedError

    def resize(self, new_size: int) -> None:
        raise NotImplementedError

    def load_factor(self) -> float:
        return self.num_keys / self.size if self.size > 0 else 0.0

    def collision_count(self) -> int:
        raise NotImplementedError

    def get_collision_statistics(self) -> Dict[str, Any]:
        raise NotImplementedError

    def estimate_memory_bytes(self) -> int:
        """
        Estimates the memory overhead of the hash table structures.
        """
        return sys.getsizeof(self)


class HashTableChaining(AbstractHashTable):
    """
    Hash Table implementing collision resolution via Separate Chaining.
    """
    def __init__(self, size: int = 10):
        super().__init__(size)
        self.buckets: List[List[Tuple[Any, Any]]] = [[] for _ in range(size)]

    def insert(self, key: Any, value: Any, record_trace: bool = True) -> List[Dict[str, Any]]:
        idx = self.hash_function(key)
        bucket = self.buckets[idx]
        bucket_before = list(bucket) if record_trace else []
        
        trace = []
        if record_trace:
            trace = [
                {"step": "input", "val": key},
                {"step": "hash", "formula": f"{key} % {self.size}", "index": idx}
            ]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                if record_trace:
                    trace.append({
                        "step": "traverse",
                        "index": idx,
                        "bucket": bucket_before,
                        "target_index": i,
                        "action": "update"
                    })
                    trace.append({"step": "done", "index": idx, "success": True, "action": "update"})
                return trace
                
        bucket.append((key, value))
        self.num_keys += 1
        if record_trace:
            trace.append({
                "step": "traverse",
                "index": idx,
                "bucket": bucket_before,
                "target_index": len(bucket_before),
                "action": "insert"
            })
            trace.append({"step": "done", "index": idx, "success": True, "action": "insert"})
        return trace

    def search(self, key: Any, record_trace: bool = True) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        idx = self.hash_function(key)
        bucket = self.buckets[idx]
        
        trace = []
        if record_trace:
            trace = [
                {"step": "input", "val": key},
                {"step": "hash", "formula": f"{key} % {self.size}", "index": idx}
            ]
        
        for i, (k, v) in enumerate(bucket):
            if record_trace:
                trace.append({
                    "step": "compare",
                    "index": idx,
                    "bucket_idx": i,
                    "current_key": k,
                    "target_key": key,
                    "matched": k == key
                })
            if k == key:
                if record_trace:
                    trace.append({"step": "done", "index": idx, "found": True, "value": v})
                return v, trace
                
        if record_trace:
            trace.append({"step": "done", "index": idx, "found": False, "value": None})
        return None, trace

    def delete(self, key: Any) -> bool:
        idx = self.hash_function(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.num_keys -= 1
                return True
        return False

    def resize(self, new_size: int) -> None:
        old_buckets = self.buckets
        self.size = new_size
        self.buckets = [[] for _ in range(new_size)]
        self.num_keys = 0
        for bucket in old_buckets:
            for k, v in bucket:
                self.insert(k, v, record_trace=False)

    def collision_count(self) -> int:
        occupied = sum(1 for b in self.buckets if len(b) > 0)
        return max(0, self.num_keys - occupied)

    def get_collision_statistics(self) -> Dict[str, Any]:
        collisions = self.collision_count()
        bucket_lengths = [len(b) for b in self.buckets]
        max_len = max(bucket_lengths) if bucket_lengths else 0
        avg_len = sum(bucket_lengths) / len(bucket_lengths) if bucket_lengths else 0.0
        
        return {
            "users": self.num_keys,
            "table_size": self.size,
            "collisions": collisions,
            "load_factor": self.load_factor(),
            "avg_bucket_length": avg_len,
            "max_bucket_length": max_len,
            "collision_rate": collisions / self.num_keys if self.num_keys > 0 else 0.0
        }

    def estimate_memory_bytes(self) -> int:
        total = sys.getsizeof(self.buckets)
        for b in self.buckets:
            total += sys.getsizeof(b)
            for item in b:
                total += sys.getsizeof(item)
        return total


class HashTableLinearProbing(AbstractHashTable):
    """
    Hash Table implementing collision resolution via Linear Probing (Open Addressing).
    """
    TOMBSTONE = "<TOMBSTONE>"

    def __init__(self, size: int = 10):
        super().__init__(size)
        self.buckets: List[Optional[Tuple[Any, Any]]] = [None] * size
        self._collisions = 0

    def insert(self, key: Any, value: Any, record_trace: bool = True) -> List[Dict[str, Any]]:
        if self.load_factor() >= 0.95:
            self.resize(self.size * 2 + 1)

        idx = self.hash_function(key)
        trace = []
        if record_trace:
            trace = [
                {"step": "input", "val": key},
                {"step": "hash", "formula": f"{key} % {self.size}", "index": idx}
            ]

        first_tombstone_idx = None
        probe_idx = idx
        probe_count = 0
        collision_detected = False

        while probe_count < self.size:
            slot = self.buckets[probe_idx]
            
            if slot is None:
                target_idx = first_tombstone_idx if first_tombstone_idx is not None else probe_idx
                self.buckets[target_idx] = (key, value)
                self.num_keys += 1
                if collision_detected:
                    self._collisions += 1
                if record_trace:
                    trace.append({
                        "step": "probe",
                        "index": probe_idx,
                        "state": "empty",
                        "probes": probe_count,
                        "action": "inserted"
                    })
                    trace.append({"step": "done", "index": target_idx, "success": True, "action": "insert"})
                return trace
                
            elif slot == self.TOMBSTONE:
                if first_tombstone_idx is None:
                    first_tombstone_idx = probe_idx
                if record_trace:
                    trace.append({
                        "step": "probe",
                        "index": probe_idx,
                        "state": "tombstone",
                        "probes": probe_count,
                        "action": "continue"
                    })
                
            else:
                k, v = slot
                if k == key:
                    self.buckets[probe_idx] = (key, value)
                    if record_trace:
                        trace.append({
                            "step": "probe",
                            "index": probe_idx,
                            "state": "occupied (match)",
                            "probes": probe_count,
                            "action": "updated"
                        })
                        trace.append({"step": "done", "index": probe_idx, "success": True, "action": "update"})
                    return trace
                
                collision_detected = True
                if record_trace:
                    trace.append({
                        "step": "probe",
                        "index": probe_idx,
                        "state": "occupied (collision)",
                        "probes": probe_count,
                        "action": "continue"
                    })

            probe_idx = (probe_idx + 1) % self.size
            probe_count += 1

        self.resize(self.size * 2 + 1)
        return self.insert(key, value, record_trace=record_trace)

    def search(self, key: Any, record_trace: bool = True) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        idx = self.hash_function(key)
        trace = []
        if record_trace:
            trace = [
                {"step": "input", "val": key},
                {"step": "hash", "formula": f"{key} % {self.size}", "index": idx}
            ]

        probe_idx = idx
        probe_count = 0

        while probe_count < self.size:
            slot = self.buckets[probe_idx]
            
            if slot is None:
                if record_trace:
                    trace.append({
                        "step": "probe_search",
                        "index": probe_idx,
                        "state": "empty",
                        "probes": probe_count,
                        "matched": False
                    })
                    trace.append({"step": "done", "index": probe_idx, "found": False, "value": None})
                return None, trace
                
            elif slot == self.TOMBSTONE:
                if record_trace:
                    trace.append({
                        "step": "probe_search",
                        "index": probe_idx,
                        "state": "tombstone",
                        "probes": probe_count,
                        "matched": False
                    })
                
            else:
                k, v = slot
                matched = (k == key)
                if record_trace:
                    trace.append({
                        "step": "probe_search",
                        "index": probe_idx,
                        "state": f"occupied ({k})",
                        "probes": probe_count,
                        "matched": matched,
                        "current_key": k,
                        "target_key": key
                    })
                if matched:
                    if record_trace:
                        trace.append({"step": "done", "index": probe_idx, "found": True, "value": v})
                    return v, trace

            probe_idx = (probe_idx + 1) % self.size
            probe_count += 1

        if record_trace:
            trace.append({"step": "done", "index": idx, "found": False, "value": None})
        return None, trace

    def delete(self, key: Any) -> bool:
        idx = self.hash_function(key)
        probe_idx = idx
        probe_count = 0

        while probe_count < self.size:
            slot = self.buckets[probe_idx]
            if slot is None:
                return False
            elif slot != self.TOMBSTONE:
                k, v = slot
                if k == key:
                    self.buckets[probe_idx] = self.TOMBSTONE
                    self.num_keys -= 1
                    return True
            probe_idx = (probe_idx + 1) % self.size
            probe_count += 1
        return False

    def resize(self, new_size: int) -> None:
        old_buckets = self.buckets
        self.size = new_size
        self.buckets = [None] * new_size
        self.num_keys = 0
        self._collisions = 0
        for slot in old_buckets:
            if slot is not None and slot != self.TOMBSTONE:
                k, v = slot
                self.insert(k, v, record_trace=False)

    def collision_count(self) -> int:
        return self._collisions

    def get_collision_statistics(self) -> Dict[str, Any]:
        collisions = self.collision_count()
        return {
            "users": self.num_keys,
            "table_size": self.size,
            "collisions": collisions,
            "load_factor": self.load_factor(),
            "avg_bucket_length": 0.0,
            "max_bucket_length": 0,
            "collision_rate": collisions / self.num_keys if self.num_keys > 0 else 0.0
        }

    def estimate_memory_bytes(self) -> int:
        total = sys.getsizeof(self.buckets)
        for slot in self.buckets:
            if slot is not None:
                total += sys.getsizeof(slot)
        return total


# Backward compatibility alias
class HashTable(HashTableChaining):
    def get_load_factor(self) -> float:
        return self.load_factor()


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D NumPy arrays with zero-vector safety.
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return float(np.clip(sim, -1.0, 1.0))


def weighted_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, weights: np.ndarray) -> float:
    """
    Computes weighted cosine similarity:
    sim = sum(w * a * b) / ( sqrt(sum(w * a^2)) * sqrt(sum(w * b^2)) )
    Safe for zero vectors and clipped to [-1, 1].
    """
    w_a = vec_a * weights
    w_b = vec_b * weights
    dot = np.sum(vec_a * w_b)
    norm_a = math.sqrt(float(np.sum(weights * (vec_a ** 2))))
    norm_b = math.sqrt(float(np.sum(weights * (vec_b ** 2))))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return float(np.clip(sim, -1.0, 1.0))


def compute_idf_weights(dataset: List[Tuple[int, List[int]]], num_movies: int = 100) -> np.ndarray:
    """
    Computes smoothed IDF weights per movie:
    IDF(i) = log((1 + U) / (1 + df_i)) + 1
    """
    u_total = len(dataset)
    df = np.zeros(num_movies + 1, dtype=np.float64)
    for _, movies in dataset:
        for m in movies:
            if 1 <= m <= num_movies:
                df[m] += 1.0
                
    idf = np.log((1.0 + u_total) / (1.0 + df)) + 1.0
    return idf


class RecommenderSystem:
    """
    Recommender System backed by a Hash Table user lookup layer with cosine, weighted cosine,
    and cold-start hybrid support.
    """
    def __init__(self, table_size: int = 10, strategy_class = HashTableChaining):
        self.strategy_class = strategy_class
        self.hash_table: AbstractHashTable = strategy_class(table_size)
        self.all_users: List[int] = []
        self.dataset: List[Tuple[int, List[int]]] = []
        self.movie_popularities: Dict[int, int] = {}
        self.idf_weights: Optional[np.ndarray] = None
        self.inverted_index: Dict[int, Set[int]] = {}

    def fit(self, dataset: List[Tuple[int, List[int]]], num_movies: int = 100) -> None:
        self.dataset = dataset
        self.all_users = []
        self.movie_popularities = {m: 0 for m in range(1, num_movies + 1)}
        self.inverted_index = {m: set() for m in range(1, num_movies + 1)}

        for user_id, movies in dataset:
            self.hash_table.insert(user_id, movies, record_trace=False)
            self.all_users.append(user_id)
            for m in movies:
                self.movie_popularities[m] = self.movie_popularities.get(m, 0) + 1
                if m in self.inverted_index:
                    self.inverted_index[m].add(user_id)

        self.idf_weights = compute_idf_weights(dataset, num_movies)

    def get_user_preferences(self, user_id: int) -> Tuple[Optional[List[int]], List[Dict[str, Any]]]:
        return self.hash_table.search(user_id)

    def recommend_movies(self, user_id: int, all_movies: List[int], top_n: int = 3) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Random baseline recommendation (preserves backward compatibility, uses local RNG).
        """
        user_movies, trace = self.get_user_preferences(user_id)
        if not user_movies:
            return [], trace
            
        recommendations = [m for m in all_movies if m not in user_movies]
        local_rng = random.Random(user_id)
        if len(recommendations) <= top_n:
            return sorted(recommendations), trace
        return local_rng.sample(recommendations, top_n), trace

    def _user_movies_to_vector(self, movies: List[int], num_movies: int) -> np.ndarray:
        vec = np.zeros(num_movies + 1, dtype=np.float64)
        for m in movies:
            if 1 <= m <= num_movies:
                vec[m] = 1.0
        return vec

    def recommend_movies_cosine(
        self,
        user_id: int,
        all_movies: List[int],
        top_n: int = 3,
        top_k_users: int = 10,
        use_inverted_index: bool = False
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        User-based collaborative filtering using cosine similarity.
        """
        target_movies, trace = self.get_user_preferences(user_id)
        if not target_movies:
            return [], trace

        num_movies = max(all_movies) if all_movies else 100
        target_vec = self._user_movies_to_vector(target_movies, num_movies)

        # Candidate user selection
        candidate_user_ids = set()
        if use_inverted_index:
            for m in target_movies:
                candidate_user_ids.update(self.inverted_index.get(m, set()))
            candidate_user_ids.discard(user_id)
        else:
            candidate_user_ids = set(self.all_users) - {user_id}

        similarities: List[Tuple[float, int, List[int]]] = []
        for other_id in candidate_user_ids:
            other_movies, _ = self.hash_table.search(other_id, record_trace=False)
            if not other_movies:
                continue
            other_vec = self._user_movies_to_vector(other_movies, num_movies)
            sim = compute_cosine_similarity(target_vec, other_vec)
            if sim > 0.0:
                similarities.append((sim, other_id, other_movies))

        # Deterministic sorting: sort by similarity descending, then other_id ascending
        similarities.sort(key=lambda x: (-x[0], x[1]))
        top_neighbors = similarities[:top_k_users]

        # Aggregate candidate movie scores
        candidate_scores: Dict[int, float] = {}
        for sim, _, other_movies in top_neighbors:
            for m in other_movies:
                if m not in target_movies and m in all_movies:
                    candidate_scores[m] = candidate_scores.get(m, 0.0) + sim

        # Deterministic sorting of candidates: score descending, movie_id ascending
        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        recs = [m for m, _ in ranked[:top_n]]
        return recs, trace

    def recommend_movies_weighted_cosine(
        self,
        user_id: int,
        all_movies: List[int],
        top_n: int = 3,
        top_k_users: int = 10,
        weighting: str = "idf",
        popularity_alpha: float = 0.0
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        User-based collaborative filtering using IDF-weighted cosine similarity and popularity damping.
        """
        target_movies, trace = self.get_user_preferences(user_id)
        if not target_movies:
            return [], trace

        num_movies = max(all_movies) if all_movies else 100
        target_vec = self._user_movies_to_vector(target_movies, num_movies)

        if weighting == "idf" and self.idf_weights is not None:
            weights = self.idf_weights
        else:
            weights = np.ones(num_movies + 1, dtype=np.float64)

        similarities: List[Tuple[float, int, List[int]]] = []
        for other_id in self.all_users:
            if other_id == user_id:
                continue
            other_movies, _ = self.hash_table.search(other_id, record_trace=False)
            if not other_movies:
                continue
            other_vec = self._user_movies_to_vector(other_movies, num_movies)
            sim = weighted_cosine_similarity(target_vec, other_vec, weights)
            if sim > 0.0:
                similarities.append((sim, other_id, other_movies))

        similarities.sort(key=lambda x: (-x[0], x[1]))
        top_neighbors = similarities[:top_k_users]

        candidate_scores: Dict[int, float] = {}
        for sim, _, other_movies in top_neighbors:
            for m in other_movies:
                if m not in target_movies and m in all_movies:
                    score = candidate_scores.get(m, 0.0) + sim
                    candidate_scores[m] = score

        # Apply popularity damping: adjusted = score / (pop^alpha)
        if popularity_alpha > 0.0:
            damped_scores: Dict[int, float] = {}
            for m, score in candidate_scores.items():
                pop = max(1, self.movie_popularities.get(m, 1))
                damped_scores[m] = score / (pop ** popularity_alpha)
            candidate_scores = damped_scores

        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        recs = [m for m, _ in ranked[:top_n]]
        return recs, trace

    def recommend_movies_hybrid(
        self,
        user_id: int,
        all_movies: List[int],
        top_n: int = 3,
        seed_items: Optional[List[int]] = None
    ) -> Tuple[List[int], List[str], Dict[str, float]]:
        """
        Hybrid recommendation handling cold-start scenarios deterministically.
        Returns (recommendations, source_labels, timing_info).
        """
        start_time = time.perf_counter()
        target_movies, _ = self.get_user_preferences(user_id)
        
        recs: List[int] = []
        labels: List[str] = []
        num_movies = max(all_movies) if all_movies else 100

        # Scenario A: Unknown user with no seed items -> Popularity fallback
        if not target_movies and not seed_items:
            sorted_pop = sorted(self.movie_popularities.items(), key=lambda x: (-x[1], x[0]))
            recs = [m for m, _ in sorted_pop if m in all_movies][:top_n]
            labels = ["popular-fallback"] * len(recs)

        # Scenario B: Unknown user with seed items -> Seed-based cosine
        elif not target_movies and seed_items:
            seed_vec = self._user_movies_to_vector(seed_items, num_movies)
            weights = self.idf_weights if self.idf_weights is not None else np.ones(num_movies + 1)
            
            sims = []
            for other_id in self.all_users:
                other_movies, _ = self.hash_table.search(other_id, record_trace=False)
                if not other_movies:
                    continue
                other_vec = self._user_movies_to_vector(other_movies, num_movies)
                sim = weighted_cosine_similarity(seed_vec, other_vec, weights)
                if sim > 0.0:
                    sims.append((sim, other_id, other_movies))
                    
            sims.sort(key=lambda x: (-x[0], x[1]))
            top_neighbors = sims[:10]

            candidate_scores: Dict[int, float] = {}
            for sim, _, other_movies in top_neighbors:
                for m in other_movies:
                    if m not in seed_items and m in all_movies:
                        candidate_scores[m] = candidate_scores.get(m, 0.0) + sim

            ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
            recs = [m for m, _ in ranked[:top_n]]
            labels = ["seed-cosine"] * len(recs)

            # Backfill with popular if fewer than top_n
            if len(recs) < top_n:
                sorted_pop = sorted(self.movie_popularities.items(), key=lambda x: (-x[1], x[0]))
                for m, _ in sorted_pop:
                    if m in all_movies and m not in seed_items and m not in recs:
                        recs.append(m)
                        labels.append("popular-fallback")
                        if len(recs) == top_n:
                            break

        # Scenario C: Known user -> Personalised Cosine + Popular fallback backfill
        else:
            cosine_recs, _ = self.recommend_movies_cosine(user_id, all_movies, top_n=top_n)
            recs = cosine_recs
            labels = ["cosine"] * len(recs)

            if len(recs) < top_n:
                seen = set(target_movies or []) | set(recs)
                sorted_pop = sorted(self.movie_popularities.items(), key=lambda x: (-x[1], x[0]))
                for m, _ in sorted_pop:
                    if m in all_movies and m not in seen:
                        recs.append(m)
                        labels.append("popular-fallback")
                        if len(recs) == top_n:
                            break

        elapsed = time.perf_counter() - start_time
        return recs, labels, {"latency_sec": elapsed}
