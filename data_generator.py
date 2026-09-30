import random
from typing import Dict, List, Tuple, Optional

def data_generator(n_users: int, n_items: int, seed: Optional[int] = 42) -> List[Tuple[int, List[int]]]:
    """
    Generates n_users unique user IDs and assigns each a random list of 3-10 items
    from a pool of n_items.
    
    Parameters:
        n_users: Number of users to generate.
        n_items: Number of available items/movies.
        seed: Random seed for reproducibility (default 42). If None, generates fresh random data.
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()  # Use system entropy for fresh generation
        
    dataset = []
    start_id = 1000
    user_ids = random.sample(range(start_id, start_id + n_users * 10), n_users)
    
    for user_id in user_ids:
        num_user_items = random.randint(3, 10)
        item_ids = sorted(random.sample(range(1, n_items + 1), num_user_items))
        dataset.append((user_id, item_ids))
        
    return dataset

def generate_user_data(num_users: int, num_movies: int = 100, seed: Optional[int] = 42) -> List[Tuple[int, List[int]]]:
    """
    Backwards compatibility wrapper for data_generator supporting optional seed.
    """
    return data_generator(num_users, num_movies, seed=seed)


def get_movie_genres(num_movies: int = 100, num_genres: int = 5, seed: int = 42) -> Dict[int, int]:
    """
    Deterministically assigns each movie ID (1 to num_movies) to one of num_genres genres (0 to num_genres - 1).
    """
    rng = random.Random(seed)
    genres = [i % num_genres for i in range(num_movies)]
    rng.shuffle(genres)
    return {m: genres[m - 1] for m in range(1, num_movies + 1)}


def _zipf_sample_without_replacement(
    rng: random.Random,
    items: List[int],
    weights: List[float],
    k: int
) -> List[int]:
    """
    Samples k items from items without replacement according to weights.
    """
    if k <= 0 or not items:
        return []
    k = min(k, len(items))
    available_items = list(items)
    available_weights = list(weights)
    selected = []
    
    for _ in range(k):
        total_w = sum(available_weights)
        if total_w <= 0:
            pick_idx = rng.randrange(len(available_items))
        else:
            probs = [w / total_w for w in available_weights]
            cum = 0.0
            r = rng.random()
            pick_idx = len(available_items) - 1
            for idx, p in enumerate(probs):
                cum += p
                if r <= cum:
                    pick_idx = idx
                    break
        selected.append(available_items.pop(pick_idx))
        available_weights.pop(pick_idx)
        
    return selected


def generate_structured_data(
    num_users: int,
    num_movies: int = 100,
    num_genres: int = 5,
    seed: int = 42,
    popularity_skew: float = 1.0,
    min_items: int = 3,
    max_items: int = 10
) -> List[Tuple[int, List[int]]]:
    """
    Generates structured user interaction data where:
    - Movies are assigned to genres.
    - Each user has 1-2 favourite genres (70-80% of their picks come from them).
    - Picks inside a genre are popularity-skewed (Zipf-like with exponent popularity_skew).
    Returns the same shape as generate_user_data: List[Tuple[user_id, List[movie_id]]].
    """
    rng = random.Random(seed)
    movie_genres = get_movie_genres(num_movies=num_movies, num_genres=num_genres, seed=seed)
    
    # Group movies by genre and compute Zipf-like weights per genre
    genre_movies: Dict[int, List[int]] = {g: [] for g in range(num_genres)}
    for m in range(1, num_movies + 1):
        genre_movies[movie_genres[m]].append(m)
    for g in genre_movies:
        genre_movies[g].sort()

    genre_weights: Dict[int, List[float]] = {}
    for g, m_list in genre_movies.items():
        genre_weights[g] = [1.0 / ((rank + 1) ** popularity_skew) for rank in range(len(m_list))]

    start_id = 1000
    user_ids = rng.sample(range(start_id, start_id + num_users * 10), num_users)
    dataset = []

    for user_id in user_ids:
        num_user_items = rng.randint(min_items, max_items)
        num_fav_genres = rng.choice([1, 2])
        fav_genres = rng.sample(range(num_genres), min(num_fav_genres, num_genres))
        other_genres = [g for g in range(num_genres) if g not in fav_genres]

        # Gather favorite and other pools
        fav_pool = []
        fav_pool_weights = []
        for g in fav_genres:
            fav_pool.extend(genre_movies[g])
            fav_pool_weights.extend(genre_weights[g])

        other_pool = []
        other_pool_weights = []
        for g in other_genres:
            other_pool.extend(genre_movies[g])
            other_pool_weights.extend(genre_weights[g])

        fav_ratio = rng.uniform(0.70, 0.80)
        target_fav = int(round(num_user_items * fav_ratio))
        target_fav = max(1, min(len(fav_pool), target_fav))
        target_other = num_user_items - target_fav

        if target_other > len(other_pool):
            target_other = len(other_pool)
            target_fav = min(len(fav_pool), num_user_items - target_other)

        fav_picks = _zipf_sample_without_replacement(rng, fav_pool, fav_pool_weights, target_fav)
        other_picks = _zipf_sample_without_replacement(rng, other_pool, other_pool_weights, target_other)

        user_picks = sorted(fav_picks + other_picks)
        # Fallback if pool limitations resulted in fewer items
        if len(user_picks) < min_items:
            remaining = [m for m in range(1, num_movies + 1) if m not in user_picks]
            needed = min(min_items - len(user_picks), len(remaining))
            user_picks = sorted(user_picks + rng.sample(remaining, needed))

        dataset.append((user_id, user_picks))

    return dataset


def generate_structured_ratings(
    num_users: int,
    num_movies: int = 100,
    num_genres: int = 5,
    seed: int = 42,
    popularity_skew: float = 1.0,
    min_items: int = 3,
    max_items: int = 10
) -> Dict[int, Dict[int, int]]:
    """
    Generates structured per-user ratings {movie_id: rating (1-5)} consistent with the same seed,
    giving higher ratings (4-5) to favourite genres and lower ratings (1-3) to other genres.
    """
    rng = random.Random(seed)
    movie_genres = get_movie_genres(num_movies=num_movies, num_genres=num_genres, seed=seed)

    genre_movies: Dict[int, List[int]] = {g: [] for g in range(num_genres)}
    for m in range(1, num_movies + 1):
        genre_movies[movie_genres[m]].append(m)
    for g in genre_movies:
        genre_movies[g].sort()

    genre_weights: Dict[int, List[float]] = {}
    for g, m_list in genre_movies.items():
        genre_weights[g] = [1.0 / ((rank + 1) ** popularity_skew) for rank in range(len(m_list))]

    start_id = 1000
    user_ids = rng.sample(range(start_id, start_id + num_users * 10), num_users)
    ratings: Dict[int, Dict[int, int]] = {}

    for user_id in user_ids:
        num_user_items = rng.randint(min_items, max_items)
        num_fav_genres = rng.choice([1, 2])
        fav_genres = rng.sample(range(num_genres), min(num_fav_genres, num_genres))
        fav_genres_set = set(fav_genres)
        other_genres = [g for g in range(num_genres) if g not in fav_genres]

        fav_pool = []
        fav_pool_weights = []
        for g in fav_genres:
            fav_pool.extend(genre_movies[g])
            fav_pool_weights.extend(genre_weights[g])

        other_pool = []
        other_pool_weights = []
        for g in other_genres:
            other_pool.extend(genre_movies[g])
            other_pool_weights.extend(genre_weights[g])

        fav_ratio = rng.uniform(0.70, 0.80)
        target_fav = int(round(num_user_items * fav_ratio))
        target_fav = max(1, min(len(fav_pool), target_fav))
        target_other = num_user_items - target_fav

        if target_other > len(other_pool):
            target_other = len(other_pool)
            target_fav = min(len(fav_pool), num_user_items - target_other)

        fav_picks = _zipf_sample_without_replacement(rng, fav_pool, fav_pool_weights, target_fav)
        other_picks = _zipf_sample_without_replacement(rng, other_pool, other_pool_weights, target_other)

        user_picks = sorted(fav_picks + other_picks)
        if len(user_picks) < min_items:
            remaining = [m for m in range(1, num_movies + 1) if m not in user_picks]
            needed = min(min_items - len(user_picks), len(remaining))
            user_picks = sorted(user_picks + rng.sample(remaining, needed))

        user_ratings: Dict[int, int] = {}
        for m in user_picks:
            m_genre = movie_genres[m]
            if m_genre in fav_genres_set:
                rating = rng.choices([4, 5], weights=[0.35, 0.65])[0]
            else:
                rating = rng.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            user_ratings[m] = rating

        ratings[user_id] = user_ratings

    return ratings


def format_first_n_users(dataset: List[Tuple[int, List[int]]], n: int = 10) -> str:
    """
    Formats the first n users in the dataset for a text output console.
    """
    lines = []
    lines.append(f"Generated {len(dataset)} Users\n")
    
    for i, (user_id, movies) in enumerate(dataset[:n]):
        lines.append(f"User ID: {user_id}")
        lines.append("Movies:")
        for movie in movies:
            lines.append(f"  - {movie}")
        lines.append("-" * 30)
        
    if len(dataset) > n:
        lines.append(f"... and {len(dataset) - n} more users.")
        
    return "\n".join(lines)

