import random
import sys
import time
from typing import Any, List, Tuple, Optional, Dict, Set
import numpy as np

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

    def insert(self, key: Any, value: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def search(self, key: Any) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
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

    def insert(self, key: Any, value: Any) -> List[Dict[str, Any]]:
        idx = self.hash_function(key)
        bucket = self.buckets[idx]
        bucket_before = list(bucket)
        
        trace = [
            {"step": "input", "val": key},
            {"step": "hash", "formula": f"{key} % {self.size}", "index": idx}
        ]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
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
        trace.append({
            "step": "traverse",
            "index": idx,
            "bucket": bucket_before,
            "target_index": len(bucket_before),
            "action": "insert"
        })
        trace.append({"step": "done", "index": idx, "success": True, "action": "insert"})
        return trace

    def search(self, key: Any) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        idx = self.hash_function(key)
        bucket = self.buckets[idx]
        
        trace = [
            {"step": "input", "val": key},
            {"step": "hash", "formula": f"{key} % {self.size}", "index": idx}
        ]
        
        for i, (k, v) in enumerate(bucket):
            trace.append({
                "step": "compare",
                "index": idx,
                "bucket_idx": i,
                "current_key": k,
                "target_key": key,
                "matched": k == key
            })
            if k == key:
                trace.append({"step": "done", "index": idx, "found": True, "value": v})
                return v, trace
                
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
                self.insert(k, v)

    def collision_count(self) -> int:
        occupied = sum(1 for b in self.buckets if len(b) > 0)
        return max(0, self.num_keys - occupied)

    def get_collision_statistics(self) -> Dict[str, Any]:
        occupied_buckets = sum(1 for b in self.buckets if len(b) > 0)
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
        # Estimate size of buckets list + contents
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

    def insert(self, key: Any, value: Any) -> List[Dict[str, Any]]:
        # Auto resize if load factor is extremely high (e.g. >= 0.95)
        if self.load_factor() >= 0.95:
            self.resize(self.size * 2 + 1)

        idx = self.hash_function(key)
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
                # Found empty spot
                target_idx = first_tombstone_idx if first_tombstone_idx is not None else probe_idx
                self.buckets[target_idx] = (key, value)
                self.num_keys += 1
                if collision_detected:
                    self._collisions += 1
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
                trace.append({
                    "step": "probe",
                    "index": probe_idx,
                    "state": "tombstone",
                    "probes": probe_count,
                    "action": "continue"
                })
                
            else:
                # Slot has active elements
                k, v = slot
                if k == key:
                    # Update scenario
                    self.buckets[probe_idx] = (key, value)
                    trace.append({
                        "step": "probe",
                        "index": probe_idx,
                        "state": "occupied (match)",
                        "probes": probe_count,
                        "action": "updated"
                    })
                    trace.append({"step": "done", "index": probe_idx, "success": True, "action": "update"})
                    return trace
                
                # Collision detected
                collision_detected = True
                trace.append({
                    "step": "probe",
                    "index": probe_idx,
                    "state": "occupied (collision)",
                    "probes": probe_count,
                    "action": "continue"
                })

            probe_idx = (probe_idx + 1) % self.size
            probe_count += 1

        # Table full scenario
        self.resize(self.size * 2 + 1)
        return self.insert(key, value)

    def search(self, key: Any) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        idx = self.hash_function(key)
        trace = [
            {"step": "input", "val": key},
            {"step": "hash", "formula": f"{key} % {self.size}", "index": idx}
        ]

        probe_idx = idx
        probe_count = 0

        while probe_count < self.size:
            slot = self.buckets[probe_idx]
            
            if slot is None:
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
                    trace.append({"step": "done", "index": probe_idx, "found": True, "value": v})
                    return v, trace

            probe_idx = (probe_idx + 1) % self.size
            probe_count += 1

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
                self.insert(k, v)

    def collision_count(self) -> int:
        return self._collisions

    def get_collision_statistics(self) -> Dict[str, Any]:
        collisions = self.collision_count()
        return {
            "users": self.num_keys,
            "table_size": self.size,
            "collisions": collisions,
            "load_factor": self.load_factor(),
            "avg_bucket_length": 0.0,  # N/A for Probing
            "max_bucket_length": 0,    # N/A for Probing
            "collision_rate": collisions / self.num_keys if self.num_keys > 0 else 0.0
        }

    def estimate_memory_bytes(self) -> int:
        total = sys.getsizeof(self.buckets)
        for slot in self.buckets:
            if slot is not None:
                total += sys.getsizeof(slot)
        return total


# Backward compatibility wrapper
class HashTable(HashTableChaining):
    """
    HashTable alias for HashTableChaining to preserve legacy imports.
    """
    def get_load_factor(self) -> float:
        return self.load_factor()


def cosine_similarity(vector_a: Any, vector_b: Any) -> float:
    """
    Computes cosine similarity between two numeric vectors using NumPy.
    
    Formula:
        cos(A, B) = dot(A, B) / (norm(A) * norm(B))
        
    Handles zero vectors safely by returning 0.0 when either vector norm is 0.
    """
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cos_val = np.dot(a, b) / (norm_a * norm_b)
    # Clip to [-1.0, 1.0] to guard against floating-point inaccuracies
    return float(np.clip(cos_val, -1.0, 1.0))


class RecommenderSystem:
    """
    Recommender System supporting both random baseline and Week 6 Cosine Similarity
    collaborative filtering using custom Hash Table backend storage.
    """
    def __init__(self, table_size: int = 10, strategy_class = HashTableChaining):
        self.strategy_class = strategy_class
        self.hash_table: AbstractHashTable = strategy_class(table_size)
        self.movie_vocab: List[int] = []
        self.movie_to_idx: Dict[int, int] = {}
        self.all_user_ids: List[int] = []

    def set_movie_vocabulary(self, movies: List[int]) -> None:
        """
        Initializes the global movie vocabulary and mapping dictionary.
        Ensures all user preference vectors share identical dimensionality.
        """
        self.movie_vocab = sorted(list(set(movies)))
        self.movie_to_idx = {m: idx for idx, m in enumerate(self.movie_vocab)}

    def fit(self, dataset: List[Tuple[int, List[int]]], movie_vocab: Optional[List[int]] = None) -> None:
        """
        Populates the underlying hash table with user preferences and establishes
        the global movie vocabulary.
        """
        self.all_user_ids = []
        all_movies_set: Set[int] = set()
        for user_id, movies in dataset:
            self.hash_table.insert(user_id, movies)
            self.all_user_ids.append(user_id)
            all_movies_set.update(movies)
            
        if movie_vocab is not None:
            self.set_movie_vocabulary(movie_vocab)
        elif not self.movie_vocab:
            self.set_movie_vocabulary(sorted(list(all_movies_set)))

    def user_to_vector(self, movies: List[int]) -> np.ndarray:
        """
        Converts a list of liked movie IDs into a fixed-dimension binary preference vector
        matching the global movie vocabulary.
        Example:
            Vocab: [1, 2, 3, 4, 5]
            Movies: [1, 2, 5] -> [1.0, 1.0, 0.0, 0.0, 1.0]
        """
        vec = np.zeros(len(self.movie_vocab), dtype=np.float64)
        for m in movies:
            if m in self.movie_to_idx:
                vec[self.movie_to_idx[m]] = 1.0
        return vec

    def get_user_preferences(self, user_id: int) -> Tuple[Optional[List[int]], List[Dict[str, Any]]]:
        """
        Retrieves user movie preferences and trace from the hash table.
        """
        return self.hash_table.search(user_id)

    def get_user_vector(self, user_id: int) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Fetches user preferences from the hash table and transforms them into a preference vector.
        """
        user_movies, trace = self.get_user_preferences(user_id)
        if user_movies is None:
            return None, trace
        return self.user_to_vector(user_movies), trace

    def compute_user_similarities(self, target_user_id: int) -> List[Tuple[int, float]]:
        """
        Calculates cosine similarity between a target user and every other user in the dataset:
        1. Obtains target user preference vector
        2. Compares against each other user in the system
        3. Excludes the target user itself
        4. Sorts users by descending similarity score (breaking ties with user_id ascending)
        
        Returns:
            List of (user_id, similarity_score) sorted descending
        """
        target_movies, _ = self.get_user_preferences(target_user_id)
        if target_movies is None:
            return []
        target_vec = self.user_to_vector(target_movies)
        
        similarities: List[Tuple[int, float]] = []
        for other_id in self.all_user_ids:
            if other_id == target_user_id:
                continue
            other_movies, _ = self.get_user_preferences(other_id)
            if other_movies is None:
                continue
            other_vec = self.user_to_vector(other_movies)
            sim = cosine_similarity(target_vec, other_vec)
            similarities.append((other_id, sim))
            
        # Sort descending by similarity, tie-break ascending by user_id for determinism
        similarities.sort(key=lambda item: (-item[1], item[0]))
        return similarities

    def recommend_movies_baseline(self, user_id: int, all_movies: Optional[List[int]] = None, top_n: int = 3) -> Tuple[List[int], Dict[str, float], List[Dict[str, Any]]]:
        """
        Preserved baseline recommendation algorithm.
        Randomly samples unliked movies using random.seed(user_id).
        Measures recommendation execution time.
        """
        t0 = time.perf_counter()
        
        t_hl_start = time.perf_counter()
        user_movies, trace = self.get_user_preferences(user_id)
        hash_lookup_time = time.perf_counter() - t_hl_start
        
        if not user_movies:
            total_time = time.perf_counter() - t0
            timing = {
                "hash_lookup_time": hash_lookup_time,
                "similarity_calc_time": 0.0,
                "rec_gen_time": 0.0,
                "total_time": total_time
            }
            return [], timing, trace
            
        pool = all_movies if all_movies is not None else self.movie_vocab
        
        t_gen_start = time.perf_counter()
        candidates = [m for m in pool if m not in user_movies]
        random.seed(user_id)
        recs = random.sample(candidates, min(top_n, len(candidates))) if candidates else []
        rec_gen_time = time.perf_counter() - t_gen_start
        
        total_time = time.perf_counter() - t0
        timing = {
            "hash_lookup_time": hash_lookup_time,
            "similarity_calc_time": 0.0,
            "rec_gen_time": rec_gen_time,
            "total_time": total_time
        }
        return recs, timing, trace

    def recommend_movies_cosine(self, user_id: int, top_n: int = 3, top_k_users: int = 10) -> Tuple[List[int], Dict[str, float], List[Tuple[int, float]], List[Dict[str, Any]]]:
        """
        Similarity-based collaborative filtering recommendations:
        - Finds top similar users using cosine similarity
        - Collects candidate movies liked by those users
        - Excludes movies already liked by target user
        - Scores candidate movies by similarity-weighted support: sum(similarity_score)
        - Deterministically ranks candidates (score descending, movie ID ascending)
        - Returns top-N recommendations, timing metrics, top similar users, and trace
        
        No random sampling is used in this method.
        """
        t0 = time.perf_counter()
        
        t_hl_start = time.perf_counter()
        user_movies, trace = self.get_user_preferences(user_id)
        hash_lookup_time = time.perf_counter() - t_hl_start
        
        if not user_movies:
            total_time = time.perf_counter() - t0
            timing = {
                "hash_lookup_time": hash_lookup_time,
                "similarity_calc_time": 0.0,
                "rec_gen_time": 0.0,
                "total_time": total_time
            }
            return [], timing, [], trace
            
        # Step 1: Compute user-user similarities
        t_sim_start = time.perf_counter()
        similar_users = self.compute_user_similarities(user_id)
        similarity_calc_time = time.perf_counter() - t_sim_start
        
        # Step 2: Formulate recommendations from top similar users
        t_gen_start = time.perf_counter()
        user_movies_set = set(user_movies)
        
        # Select top k users with positive similarity
        candidate_neighbors = [u for u in similar_users if u[1] > 0][:top_k_users]
        if not candidate_neighbors and similar_users:
            candidate_neighbors = similar_users[:top_k_users]
            
        # Calculate similarity-weighted support for each movie candidate
        candidate_scores: Dict[int, float] = {}
        for sim_user_id, sim_score in candidate_neighbors:
            if sim_score <= 0:
                continue
            neighbor_movies, _ = self.get_user_preferences(sim_user_id)
            if not neighbor_movies:
                continue
            for m in neighbor_movies:
                if m not in user_movies_set:
                    candidate_scores[m] = candidate_scores.get(m, 0.0) + sim_score
                    
        # Deterministic ranking: score descending, movie ID ascending
        ranked_candidates = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        recs = [m for m, _ in ranked_candidates[:top_n]]
        rec_gen_time = time.perf_counter() - t_gen_start
        
        total_time = time.perf_counter() - t0
        timing = {
            "hash_lookup_time": hash_lookup_time,
            "similarity_calc_time": similarity_calc_time,
            "rec_gen_time": rec_gen_time,
            "total_time": total_time
        }
        return recs, timing, candidate_neighbors, trace

    def recommend_movies(self, user_id: int, all_movies: List[int], top_n: int = 3) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Backwards-compatible wrapper routing to recommend_movies_baseline.
        """
        recs, _, trace = self.recommend_movies_baseline(user_id, all_movies, top_n)
        return recs, trace

