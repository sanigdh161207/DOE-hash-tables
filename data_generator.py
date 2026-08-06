import random
from typing import Dict, List, Tuple

def data_generator(n_users: int, n_items: int) -> List[Tuple[int, List[int]]]:
    """
    Generates n_users unique user IDs and assigns each a random list of 3-10 items
    from a pool of n_items.
    """
    random.seed(42)  # For reproducibility
    dataset = []
    start_id = 1000
    user_ids = random.sample(range(start_id, start_id + n_users * 10), n_users)
    
    for user_id in user_ids:
        num_user_items = random.randint(3, 10)
        item_ids = sorted(random.sample(range(1, n_items + 1), num_user_items))
        dataset.append((user_id, item_ids))
        
    return dataset

def generate_user_data(num_users: int, num_movies: int = 100) -> List[Tuple[int, List[int]]]:
    """
    Backwards compatibility wrapper for data_generator.
    """
    return data_generator(num_users, num_movies)


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
