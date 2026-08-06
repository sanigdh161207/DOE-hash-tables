import random
import sys
from typing import Any, List, Tuple, Optional, Dict

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


class RecommenderSystem:
    """
    Recommender System that uses a customizable HashTable implementation.
    """
    def __init__(self, table_size: int = 10, strategy_class = HashTableChaining):
        self.strategy_class = strategy_class
        self.hash_table: AbstractHashTable = strategy_class(table_size)

    def fit(self, dataset: List[Tuple[int, List[int]]]) -> None:
        for user_id, movies in dataset:
            self.hash_table.insert(user_id, movies)

    def get_user_preferences(self, user_id: int) -> Tuple[Optional[List[int]], List[Dict[str, Any]]]:
        return self.hash_table.search(user_id)

    def recommend_movies(self, user_id: int, all_movies: List[int], top_n: int = 3) -> Tuple[List[int], List[Dict[str, Any]]]:
        user_movies, trace = self.get_user_preferences(user_id)
        if not user_movies:
            return [], trace
            
        recommendations = [m for m in all_movies if m not in user_movies]
        random.seed(user_id)
        return random.sample(recommendations, min(top_n, len(recommendations))), trace
