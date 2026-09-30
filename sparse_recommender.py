import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from scipy.sparse import csr_matrix
from recommender import HashTableChaining, AbstractHashTable

def build_user_item_matrix(
    dataset: List[Tuple[int, List[int]]],
    num_movies: int = 100
) -> Tuple[csr_matrix, HashTableChaining, Dict[int, int]]:
    """
    Constructs a binary CSR matrix representation of the user-item interaction dataset.
    Maps user_id -> row_index using a HashTableChaining instance.
    Returns (csr_matrix, user_hash_table, idx_to_user_map).
    """
    num_users = len(dataset)
    user_hash_table = HashTableChaining(max(13, int(num_users / 0.7)))
    idx_to_user: Dict[int, int] = {}

    rows = []
    cols = []
    data = []

    for row_idx, (user_id, movies) in enumerate(dataset):
        user_hash_table.insert(user_id, row_idx, record_trace=False)
        idx_to_user[row_idx] = user_id
        for m in movies:
            if 1 <= m <= num_movies:
                rows.append(row_idx)
                cols.append(m)
                data.append(1.0)

    # Note: 1-indexed movie column alignment (dimension num_movies + 1)
    matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_movies + 1), dtype=np.float64)
    return matrix, user_hash_table, idx_to_user


class SparseRecommender:
    """
    Sparse Matrix Recommender System powered by scipy.sparse.csr_matrix and HashTableChaining lookup layer.
    """
    def __init__(self, num_movies: int = 100):
        self.num_movies = num_movies
        self.matrix: Optional[csr_matrix] = None
        self.norm_matrix: Optional[csr_matrix] = None
        self.user_table: Optional[HashTableChaining] = None
        self.idx_to_user: Dict[int, int] = {}
        self.user_movies_dict: Dict[int, List[int]] = {}

    def fit(self, dataset: List[Tuple[int, List[int]]]) -> None:
        self.matrix, self.user_table, self.idx_to_user = build_user_item_matrix(dataset, self.num_movies)
        self.user_movies_dict = {uid: movies for uid, movies in dataset}

        # Row-normalize the CSR matrix for efficient cosine similarity via dot product: norm(r) = r / ||r||_2
        row_norms = np.sqrt(self.matrix.multiply(self.matrix).sum(axis=1)).A1
        row_norms[row_norms == 0.0] = 1.0  # Safe zero division
        
        inv_norms = 1.0 / row_norms
        from scipy.sparse import diags
        diag_inv = diags(inv_norms)
        self.norm_matrix = diag_inv.dot(self.matrix).tocsr()

    def recommend_movies(
        self,
        user_id: int,
        all_movies: List[int],
        top_n: int = 3,
        top_k_users: int = 10
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Generates recommendations using sparse matrix vectorization.
        """
        row_idx, trace = self.user_table.search(user_id, record_trace=False)
        if row_idx is None or self.norm_matrix is None:
            return [], []

        target_movies = self.user_movies_dict.get(user_id, [])

        # Compute sparse cosine similarity vector against all users: sim_vector = NormMatrix * target_row^T
        target_row = self.norm_matrix.getrow(row_idx)
        sim_vector = self.norm_matrix.dot(target_row.T).toarray().ravel()

        # Exclude self-similarity
        sim_vector[row_idx] = 0.0

        # Select top-k users
        top_user_indices = np.argsort(-sim_vector)[:top_k_users]

        candidate_scores: Dict[int, float] = {}
        for other_idx in top_user_indices:
            sim = sim_vector[other_idx]
            if sim <= 0.0:
                continue
            other_user_id = self.idx_to_user[other_idx]
            other_movies = self.user_movies_dict.get(other_user_id, [])
            for m in other_movies:
                if m not in target_movies and m in all_movies:
                    candidate_scores[m] = candidate_scores.get(m, 0.0) + sim

        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        recs = [m for m, _ in ranked[:top_n]]
        return recs, trace

    def estimate_csr_memory_bytes(self) -> int:
        """
        Calculates exact CSR matrix memory footprint: data.nbytes + indices.nbytes + indptr.nbytes.
        """
        if self.matrix is None:
            return 0
        return self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes
