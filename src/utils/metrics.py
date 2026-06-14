"""Evaluation metrics for recommendation systems"""

import numpy as np
from typing import List, Tuple, Set


class RecommendationMetrics:
    """Calculate evaluation metrics for recommendation systems"""
    
    @staticmethod
    def ndcg_at_k(relevant_items: Set, recommended_items: List, k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K
        
        Args:
            relevant_items: Set of relevant items for the user
            recommended_items: List of recommended items in order
            k: Top-K value
            
        Returns:
            NDCG@K score
        """
        # Discounted cumulative gain
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because indexing starts at 0
        
        # Ideal DCG
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        # Normalized DCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    @staticmethod
    def map_at_k(relevant_items: Set, recommended_items: List, k: int = 10) -> float:
        """
        Calculate Mean Average Precision at K
        
        Args:
            relevant_items: Set of relevant items for the user
            recommended_items: List of recommended items in order
            k: Top-K value
            
        Returns:
            MAP@K score
        """
        num_correct = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                num_correct += 1
                precision_at_i = num_correct / (i + 1)
                sum_precisions += precision_at_i
        
        map_score = sum_precisions / min(k, len(relevant_items)) if len(relevant_items) > 0 else 0.0
        return map_score
    
    @staticmethod
    def precision_at_k(relevant_items: Set, recommended_items: List, k: int = 10) -> float:
        """
        Calculate Precision at K
        
        Args:
            relevant_items: Set of relevant items for the user
            recommended_items: List of recommended items in order
            k: Top-K value
            
        Returns:
            Precision@K score
        """
        num_relevant = sum(1 for item in recommended_items[:k] if item in relevant_items)
        precision = num_relevant / k if k > 0 else 0.0
        return precision
    
    @staticmethod
    def recall_at_k(relevant_items: Set, recommended_items: List, k: int = 10) -> float:
        """
        Calculate Recall at K
        
        Args:
            relevant_items: Set of relevant items for the user
            recommended_items: List of recommended items in order
            k: Top-K value
            
        Returns:
            Recall@K score
        """
        num_relevant = sum(1 for item in recommended_items[:k] if item in relevant_items)
        recall = num_relevant / len(relevant_items) if len(relevant_items) > 0 else 0.0
        return recall
    
    @staticmethod
    def coverage(recommended_items_per_user: List[List], total_items: int) -> float:
        """
        Calculate catalog coverage
        
        Args:
            recommended_items_per_user: List of recommendation lists per user
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score (0-1)
        """
        unique_items = set()
        for user_recs in recommended_items_per_user:
            unique_items.update(user_recs)
        
        coverage = len(unique_items) / total_items if total_items > 0 else 0.0
        return coverage
    
    @staticmethod
    def diversity_score(recommended_items: List, similarity_matrix: np.ndarray) -> float:
        """
        Calculate diversity of recommendations using similarity matrix
        
        Args:
            recommended_items: List of recommended item indices
            similarity_matrix: Item-item similarity matrix
            
        Returns:
            Diversity score (average dissimilarity)
        """
        if len(recommended_items) < 2:
            return 0.0
        
        dissimilarities = []
        for i, item1 in enumerate(recommended_items):
            for item2 in recommended_items[i+1:]:
                sim = similarity_matrix[item1, item2] if item1 < similarity_matrix.shape[0] and item2 < similarity_matrix.shape[1] else 0
                dissimilarities.append(1 - sim)
        
        diversity = np.mean(dissimilarities) if dissimilarities else 0.0
        return diversity


def evaluate_recommendations(test_interactions: List[Tuple],
                           user_recommendations: dict,
                           k_values: List[int] = [5, 10, 20]) -> dict:
    """
    Comprehensive evaluation of recommendation system
    
    Args:
        test_interactions: List of (user_id, item_id) tuples representing ground truth
        user_recommendations: Dict mapping user_id to list of recommended item_ids
        k_values: List of K values to evaluate at
        
    Returns:
        Dict containing metrics at different K values
    """
    metrics = RecommendationMetrics()
    results = {}
    
    # Group test interactions by user
    user_items = {}
    for user_id, item_id in test_interactions:
        if user_id not in user_items:
            user_items[user_id] = set()
        user_items[user_id].add(item_id)
    
    # Evaluate each K value
    for k in k_values:
        k_metrics = {
            'ndcg': [],
            'map': [],
            'precision': [],
            'recall': []
        }
        
        for user_id, relevant_items in user_items.items():
            if user_id in user_recommendations:
                recommendations = user_recommendations[user_id]
                
                k_metrics['ndcg'].append(metrics.ndcg_at_k(relevant_items, recommendations, k))
                k_metrics['map'].append(metrics.map_at_k(relevant_items, recommendations, k))
                k_metrics['precision'].append(metrics.precision_at_k(relevant_items, recommendations, k))
                k_metrics['recall'].append(metrics.recall_at_k(relevant_items, recommendations, k))
        
        # Average metrics
        results[f'@{k}'] = {
            'ndcg': np.mean(k_metrics['ndcg']) if k_metrics['ndcg'] else 0,
            'map': np.mean(k_metrics['map']) if k_metrics['map'] else 0,
            'precision': np.mean(k_metrics['precision']) if k_metrics['precision'] else 0,
            'recall': np.mean(k_metrics['recall']) if k_metrics['recall'] else 0,
        }
    
    return results
