"""
Utility functions for the MovieLens-32M Generative Recommendation System
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import logging

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_movielens_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens ratings and movies data"""
    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path = os.path.join(data_dir, "movies.csv")
    
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    return ratings, movies

def filter_data(ratings: pd.DataFrame, min_rating: float = 4.0, 
                min_interactions: int = 5) -> pd.DataFrame:
    """Filter ratings data"""
    # Filter by rating threshold
    positive_ratings = ratings[ratings['rating'] >= min_rating].copy()
    
    # Filter users with minimum interactions
    user_counts = positive_ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    filtered_ratings = positive_ratings[positive_ratings['userId'].isin(valid_users)]
    
    # Filter items with minimum interactions
    item_counts = filtered_ratings['movieId'].value_counts()
    valid_items = item_counts[item_counts >= min_interactions].index
    filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(valid_items)]
    
    return filtered_ratings

def create_user_sequences(ratings: pd.DataFrame, max_seq_length: int = 50) -> Dict[int, List[int]]:
    """Create user interaction sequences"""
    # Sort by timestamp
    ratings_sorted = ratings.sort_values(['userId', 'timestamp'])
    
    # Group by user and create sequences
    user_sequences = {}
    for user_id, group in ratings_sorted.groupby('userId'):
        sequence = group['movieId'].tolist()
        # Keep only the most recent interactions
        if len(sequence) > max_seq_length:
            sequence = sequence[-max_seq_length:]
        user_sequences[user_id] = sequence
    
    return user_sequences

def split_sequences(user_sequences: Dict[int, List[int]], 
                   test_ratio: float = 0.2, 
                   val_ratio: float = 0.1) -> Tuple[Dict, Dict, Dict]:
    """Split sequences into train/val/test"""
    train_seqs, val_seqs, test_seqs = {}, {}, {}
    
    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:  # Need at least 3 items for train/val/test
            continue
            
        seq_len = len(sequence)
        test_size = max(1, int(seq_len * test_ratio))
        val_size = max(1, int(seq_len * val_ratio))
        train_size = seq_len - test_size - val_size
        
        if train_size < 1:
            continue
            
        train_seqs[user_id] = sequence[:train_size]
        val_seqs[user_id] = sequence[train_size:train_size + val_size]
        test_seqs[user_id] = sequence[train_size + val_size:]
    
    return train_seqs, val_seqs, test_seqs

def calculate_metrics(predictions: List[List[int]], 
                     ground_truth: List[List[int]], 
                     k_values: List[int]) -> Dict[str, float]:
    """Calculate Recall@K and NDCG@K metrics"""
    metrics = {}
    
    for k in k_values:
        recall_scores = []
        ndcg_scores = []
        
        for pred, true in zip(predictions, ground_truth):
            pred_k = pred[:k]
            true_set = set(true)
            
            # Recall@K
            hits = len(set(pred_k) & true_set)
            recall = hits / len(true_set) if len(true_set) > 0 else 0
            recall_scores.append(recall)
            
            # NDCG@K
            dcg = 0
            for i, item in enumerate(pred_k):
                if item in true_set:
                    dcg += 1 / np.log2(i + 2)
            
            idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(true_set))))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        metrics[f'Recall@{k}'] = np.mean(recall_scores)
        metrics[f'NDCG@{k}'] = np.mean(ndcg_scores)
    
    return metrics
