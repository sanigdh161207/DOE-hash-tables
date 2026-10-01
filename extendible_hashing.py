import sys
from typing import Any, List, Tuple, Optional, Dict
from recommender import AbstractHashTable

class Bucket:
    """
    Represents a single bucket in an Extendible Hash Table.
    """
    def __init__(self, capacity: int = 4, local_depth: int = 1):
        self.capacity = capacity
        self.local_depth = local_depth
        self.items: List[Tuple[Any, Any]] = []

    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    def insert(self, key: Any, value: Any) -> bool:
        for i, (k, v) in enumerate(self.items):
            if k == key:
                self.items[i] = (key, value)
                return True
        if self.is_full():
            return False
        self.items.append((key, value))
        return True

    def search(self, key: Any) -> Optional[Any]:
        for k, v in self.items:
            if k == key:
                return v
        return None

    def delete(self, key: Any) -> bool:
        for i, (k, v) in enumerate(self.items):
            if k == key:
                self.items.pop(i)
                return True
        return False


class ExtendibleHashTable(AbstractHashTable):
    """
    Extendible Hash Table implementation satisfying AbstractHashTable interface.
    Uses bitwise hash masking, dynamic bucket splitting, and directory doubling.
    """
    def __init__(self, table_size: int = 4, bucket_capacity: int = 4, **kwargs):
        super().__init__(table_size)
        if "capacity" in kwargs and kwargs["capacity"] is not None:
            bucket_capacity = kwargs["capacity"]
        self.bucket_capacity = max(1, bucket_capacity)
        self.global_depth = 1
        
        # Initial directory of 2^1 = 2 pointers pointing to 2 distinct buckets
        initial_bucket_0 = Bucket(capacity=self.bucket_capacity, local_depth=1)
        initial_bucket_1 = Bucket(capacity=self.bucket_capacity, local_depth=1)
        self.directory: List[Bucket] = [initial_bucket_0, initial_bucket_1]
        
        self.splits_count = 0
        self.directory_doublings = 0
        self.entries_moved_total = 0

    def hash_function(self, key: Any) -> int:
        """
        Integer hash mixing function returning non-negative integer.
        """
        try:
            val = int(key)
        except (ValueError, TypeError):
            val = hash(key)
        val = ((val >> 16) ^ val) * 0x45d9f3b
        val = ((val >> 16) ^ val) * 0x45d9f3b
        val = (val >> 16) ^ val
        return abs(val)

    def _get_dir_index(self, hash_val: int) -> int:
        """
        Calculates directory index using lower global_depth bits of hash_val.
        """
        mask = (1 << self.global_depth) - 1
        return hash_val & mask

    def insert(self, key: Any, value: Any, record_trace: bool = True) -> List[Dict[str, Any]]:
        hash_val = self.hash_function(key)
        dir_idx = self._get_dir_index(hash_val)
        
        trace = []
        if record_trace:
            trace = [
                {"step": "input", "val": key},
                {"step": "hash", "formula": f"hash({key}) & ((1<<{self.global_depth})-1)", "index": dir_idx}
            ]

        bucket = self.directory[dir_idx]
        
        # Check if key already exists in target bucket (update case)
        for i, (k, v) in enumerate(bucket.items):
            if k == key:
                bucket.items[i] = (key, value)
                if record_trace:
                    trace.append({"step": "done", "index": dir_idx, "success": True, "action": "update"})
                return trace

        # If bucket is full, perform bucket split and possible directory doubling
        max_splits_safeguard = 30
        splits_done = 0

        while bucket.is_full() and splits_done < max_splits_safeguard:
            splits_done += 1
            self.splits_count += 1

            if bucket.local_depth == self.global_depth:
                self._double_directory()
                if record_trace:
                    trace.append({
                        "step": "directory_double",
                        "new_global_depth": self.global_depth,
                        "dir_size": len(self.directory)
                    })

            # Create new bucket for splitting
            old_local_depth = bucket.local_depth
            new_local_depth = old_local_depth + 1
            bucket.local_depth = new_local_depth

            new_bucket = Bucket(capacity=self.bucket_capacity, local_depth=new_local_depth)
            
            # Re-distribute existing items between bucket and new_bucket
            all_items = list(bucket.items)
            bucket.items = []

            for k, v in all_items:
                h = self.hash_function(k)
                if (h >> old_local_depth) & 1:
                    new_bucket.items.append((k, v))
                else:
                    bucket.items.append((k, v))
                    
            self.entries_moved_total += len(all_items)

            # Update directory pointers
            stride = 1 << old_local_depth
            start_bit = (h_target := self._get_dir_index(hash_val)) & ((1 << old_local_depth) - 1)
            for idx in range(start_bit + stride, len(self.directory), 1 << new_local_depth):
                self.directory[idx] = new_bucket

            if record_trace:
                trace.append({
                    "step": "split",
                    "bucket_local_depth": new_local_depth,
                    "dir_index": dir_idx
                })

            dir_idx = self._get_dir_index(hash_val)
            bucket = self.directory[dir_idx]

        # Insert key into appropriate bucket
        bucket.items.append((key, value))
        self.num_keys += 1
        
        if record_trace:
            trace.append({"step": "done", "index": dir_idx, "success": True, "action": "insert"})
        return trace

    def _double_directory(self) -> None:
        """
        Doubles directory size and duplicates existing bucket references.
        """
        self.directory = self.directory + list(self.directory)
        self.global_depth += 1
        self.directory_doublings += 1
        self.size = len(self.directory)

    def search(self, key: Any, record_trace: bool = True) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        hash_val = self.hash_function(key)
        dir_idx = self._get_dir_index(hash_val)
        
        trace = []
        if record_trace:
            trace = [
                {"step": "input", "val": key},
                {"step": "hash", "formula": f"hash({key}) & ((1<<{self.global_depth})-1)", "index": dir_idx}
            ]

        bucket = self.directory[dir_idx]
        val = bucket.search(key)
        
        if record_trace:
            found = val is not None
            trace.append({"step": "done", "index": dir_idx, "found": found, "value": val})

        return val, trace

    def delete(self, key: Any) -> bool:
        hash_val = self.hash_function(key)
        dir_idx = self._get_dir_index(hash_val)
        bucket = self.directory[dir_idx]
        deleted = bucket.delete(key)
        if deleted:
            self.num_keys -= 1
        return deleted

    def resize(self, new_size: int) -> None:
        """
        No-op for Extendible Hashing as directory doubles automatically during insertion.
        """
        pass

    def load_factor(self) -> float:
        """
        Calculates bucket utilization: stored records / (num_unique_buckets * bucket_capacity).
        In Extendible Hashing, this represents true physical storage utilization.
        """
        visited = set(id(b) for b in self.directory)
        total_capacity = len(visited) * self.bucket_capacity
        return self.num_keys / float(total_capacity) if total_capacity > 0 else 0.0

    def directory_load_factor(self) -> float:
        """
        Directory pointer density: stored records / directory size.
        """
        return self.num_keys / float(len(self.directory)) if len(self.directory) > 0 else 0.0

    def collision_count(self) -> int:
        """
        Number of entries sharing buckets with other keys.
        """
        collisions = 0
        visited_buckets = set()
        for b in self.directory:
            b_id = id(b)
            if b_id not in visited_buckets:
                visited_buckets.add(b_id)
                if len(b.items) > 1:
                    collisions += (len(b.items) - 1)
        return collisions

    def get_collision_statistics(self) -> Dict[str, Any]:
        visited_buckets = set()
        bucket_lengths = []
        for b in self.directory:
            b_id = id(b)
            if b_id not in visited_buckets:
                visited_buckets.add(b_id)
                bucket_lengths.append(len(b.items))

        collisions = self.collision_count()
        num_unique = len(visited_buckets)
        total_capacity = num_unique * self.bucket_capacity
        bucket_utilization = self.num_keys / float(total_capacity) if total_capacity > 0 else 0.0

        return {
            "users": self.num_keys,
            "table_size": len(self.directory),
            "directory_size": len(self.directory),
            "global_depth": self.global_depth,
            "num_unique_buckets": num_unique,
            "bucket_capacity": self.bucket_capacity,
            "total_physical_capacity": total_capacity,
            "bucket_utilization": bucket_utilization,
            "collisions": collisions,
            "load_factor": bucket_utilization,
            "directory_load_factor": self.directory_load_factor(),
            "avg_bucket_length": sum(bucket_lengths) / len(bucket_lengths) if bucket_lengths else 0.0,
            "max_bucket_length": max(bucket_lengths) if bucket_lengths else 0,
            "splits_count": self.splits_count,
            "directory_doublings": self.directory_doublings,
            "collision_rate": collisions / self.num_keys if self.num_keys > 0 else 0.0
        }

    def estimate_memory_bytes(self) -> int:
        mem = sys.getsizeof(self.directory)
        visited = set()
        for b in self.directory:
            if id(b) not in visited:
                visited.add(id(b))
                mem += sys.getsizeof(b)
                mem += sys.getsizeof(b.items)
        return mem

    def print_directory(self) -> str:
        """
        Returns ASCII representation of global directory pointers and local buckets.
        """
        lines = []
        lines.append(f"=== Extendible Hash Table (Global Depth = {self.global_depth}) ===")
        visited = {}
        for idx, bucket in enumerate(self.directory):
            b_id = id(bucket)
            if b_id not in visited:
                visited[b_id] = len(visited)
            bucket_num = visited[b_id]
            binary_str = format(idx, f"0{self.global_depth}b")
            items_str = ", ".join(f"{k}" for k, _ in bucket.items)
            lines.append(f"Dir [{binary_str}] (idx {idx:02d}) ──➔ Bucket #{bucket_num} [Local Depth {bucket.local_depth}]: [{items_str}]")
        return "\n".join(lines)
