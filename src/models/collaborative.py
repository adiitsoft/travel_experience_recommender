"""Collaborative filtering recommendation models"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from typing import Tuple, List, Dict


class UserBasedCollaborativeFilter:
    """User-based collaborative filtering model"""
    
    def __init__(self, n_neighbors: int = 10, similarity_metric: str = 'cosine'):
        """
        Initialize user-based collaborative filter
        
        Args:
            n_neighbors: Number of similar users to consider
            similarity_metric: 'cosine', 'euclidean', or 'pearson'
        """
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_similarity_matrix = None
        self.interaction_matrix = None
        self.user_id_map = None
        self.dest_id_map = None
        self.reverse_dest_map = None
        
    def fit(self, interaction_matrix: np.ndarray, user_id_map: Dict, 
            dest_id_map: Dict) -> 'UserBasedCollaborativeFilter':
        """
        Fit model on interaction matrix
        
        Args:
            interaction_matrix: User-destination rating matrix
            user_id_map: Mapping of user IDs to indices
            dest_id_map: Mapping of destination IDs to indices
            
        Returns:
            Self for chaining
        """
        self.interaction_matrix = interaction_matrix.copy()
        self.user_id_map = user_id_map
        self.dest_id_map = dest_id_map
        self.reverse_dest_map = {v: k for k, v in dest_id_map.items()}
        
        # Handle zero rows (new users)
        interaction_matrix_normalized = self.interaction_matrix.copy()
        for i in range(interaction_matrix_normalized.shape[0]):
            non_zero = interaction_matrix_normalized[i, interaction_matrix_normalized[i] > 0]
            if len(non_zero) > 0:
                mean_rating = non_zero.mean()
                interaction_matrix_normalized[i][interaction_matrix_normalized[i] == 0] = mean_rating
        
        # Compute user-user similarity
        if self.similarity_metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(interaction_matrix_normalized)
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(interaction_matrix_normalized)
            self.user_similarity_matrix = 1 / (1 + distances)
        elif self.similarity_metric == 'pearson':
            self.user_similarity_matrix = np.corrcoef(interaction_matrix_normalized)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return self
    
    def recommend(self, user_idx: int, n_recommendations: int = 10) -> List[Tuple]:
        """
        Generate recommendations for a user
        
        Args:
            user_idx: User index in the interaction matrix
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (destination_id, score) tuples
        """
        # Get similar users
        user_similarities = self.user_similarity_matrix[user_idx].copy()
        user_similarities[user_idx] = -1  # Exclude self
        
        similar_user_indices = np.argsort(user_similarities)[-self.n_neighbors:]
        similar_user_indices = similar_user_indices[similar_user_indices != user_idx]
        
        # Get destinations rated by similar users but not by target user
        user_visited = np.where(self.interaction_matrix[user_idx] > 0)[0]
        recommendations_scores = np.zeros(self.interaction_matrix.shape[1])
        
        for similar_user_idx in similar_user_indices:
            similarity_weight = user_similarities[similar_user_idx]
            similar_user_ratings = self.interaction_matrix[similar_user_idx]
            recommendations_scores += similarity_weight * similar_user_ratings
        
        # Zero out already visited destinations
        recommendations_scores[user_visited] = -np.inf
        
        # Get top recommendations
        top_dest_indices = np.argsort(recommendations_scores)[-n_recommendations:][::-1]
        
        recommendations = [
            (self.reverse_dest_map[idx], float(recommendations_scores[idx]))
            for idx in top_dest_indices
            if recommendations_scores[idx] > -np.inf
        ]
        
        return recommendations


class ItemBasedCollaborativeFilter:
    """Item-based collaborative filtering model"""
    
    def __init__(self, similarity_metric: str = 'cosine'):
        """
        Initialize item-based collaborative filter
        
        Args:
            similarity_metric: 'cosine', 'euclidean', or 'pearson'
        """
        self.similarity_metric = similarity_metric
        self.item_similarity_matrix = None
        self.interaction_matrix = None
        self.user_id_map = None
        self.dest_id_map = None
        self.reverse_dest_map = None
        
    def fit(self, interaction_matrix: np.ndarray, user_id_map: Dict,
            dest_id_map: Dict) -> 'ItemBasedCollaborativeFilter':
        """
        Fit model on interaction matrix
        
        Args:
            interaction_matrix: User-destination rating matrix
            user_id_map: Mapping of user IDs to indices
            dest_id_map: Mapping of destination IDs to indices
            
        Returns:
            Self for chaining
        """
        self.interaction_matrix = interaction_matrix.copy()
        self.user_id_map = user_id_map
        self.dest_id_map = dest_id_map
        self.reverse_dest_map = {v: k for k, v in dest_id_map.items()}
        
        # Compute destination-destination similarity
        interaction_matrix_T = self.interaction_matrix.T
        
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(interaction_matrix_T)
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(interaction_matrix_T)
            self.item_similarity_matrix = 1 / (1 + distances)
        elif self.similarity_metric == 'pearson':
            self.item_similarity_matrix = np.corrcoef(interaction_matrix_T)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return self
    
    def recommend(self, user_idx: int, n_recommendations: int = 10) -> List[Tuple]:
        """
        Generate recommendations for a user
        
        Args:
            user_idx: User index in the interaction matrix
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (destination_id, score) tuples
        """
        user_visited = np.where(self.interaction_matrix[user_idx] > 0)[0]
        user_ratings = self.interaction_matrix[user_idx]
        
        # Compute recommendation scores based on similar destinations
        recommendations_scores = np.zeros(self.interaction_matrix.shape[1])
        
        for visited_idx in user_visited:
            if user_ratings[visited_idx] > 0:
                similar_dest_scores = self.item_similarity_matrix[visited_idx]
                recommendations_scores += user_ratings[visited_idx] * similar_dest_scores
        
        # Zero out already visited destinations
        recommendations_scores[user_visited] = -np.inf
        
        # Get top recommendations
        top_dest_indices = np.argsort(recommendations_scores)[-n_recommendations:][::-1]
        
        recommendations = [
            (self.reverse_dest_map[idx], float(recommendations_scores[idx]))
            for idx in top_dest_indices
            if recommendations_scores[idx] > -np.inf
        ]
        
        return recommendations
