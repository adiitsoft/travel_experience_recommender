"""Content-based filtering recommendation models"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


class ContentBasedFilter:
    """Content-based recommendation model using destination and user features"""
    
    def __init__(self, similarity_metric: str = 'cosine'):
        """
        Initialize content-based filter
        
        Args:
            similarity_metric: Metric for computing similarity
        """
        self.similarity_metric = similarity_metric
        self.user_feature_matrix = None
        self.dest_feature_matrix = None
        self.interaction_matrix = None
        self.user_id_map = None
        self.dest_id_map = None
        self.reverse_dest_map = None
        
    def fit(self, user_feature_matrix: np.ndarray, dest_feature_matrix: np.ndarray,
            interaction_matrix: np.ndarray, user_id_map: Dict, 
            dest_id_map: Dict) -> 'ContentBasedFilter':
        """
        Fit model with feature matrices
        
        Args:
            user_feature_matrix: User feature vectors
            dest_feature_matrix: Destination feature vectors
            interaction_matrix: User-destination ratings
            user_id_map: Mapping of user IDs to indices
            dest_id_map: Mapping of destination IDs to indices
            
        Returns:
            Self for chaining
        """
        self.user_feature_matrix = user_feature_matrix
        self.dest_feature_matrix = dest_feature_matrix
        self.interaction_matrix = interaction_matrix
        self.user_id_map = user_id_map
        self.dest_id_map = dest_id_map
        self.reverse_dest_map = {v: k for k, v in dest_id_map.items()}
        
        return self
    
    def recommend(self, user_idx: int, n_recommendations: int = 10) -> List[Tuple]:
        """
        Generate recommendations based on content similarity
        
        Args:
            user_idx: User index in the feature matrix
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (destination_id, score) tuples
        """
        user_features = self.user_feature_matrix[user_idx].reshape(1, -1)
        
        # Compute similarity between user and all destinations
        if self.similarity_metric == 'cosine':
            similarity_scores = cosine_similarity(user_features, 
                                                 self.dest_feature_matrix)[0]
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Get user's visited destinations
        user_visited = np.where(self.interaction_matrix[user_idx] > 0)[0]
        similarity_scores[user_visited] = -np.inf
        
        # Get top recommendations
        top_dest_indices = np.argsort(similarity_scores)[-n_recommendations:][::-1]
        
        recommendations = [
            (self.reverse_dest_map[idx], float(similarity_scores[idx]))
            for idx in top_dest_indices
            if similarity_scores[idx] > -np.inf
        ]
        
        return recommendations
    
    def recommend_by_profile(self, user_profile: np.ndarray,
                            n_recommendations: int = 10,
                            exclude_destinations: List = None) -> List[Tuple]:
        """
        Generate recommendations for a new user profile (cold-start)
        
        Args:
            user_profile: Feature vector of user (1D array)
            n_recommendations: Number of recommendations
            exclude_destinations: List of destination indices to exclude
            
        Returns:
            List of (destination_id, score) tuples
        """
        user_profile = user_profile.reshape(1, -1)
        
        # Compute similarity
        if self.similarity_metric == 'cosine':
            similarity_scores = cosine_similarity(user_profile,
                                                 self.dest_feature_matrix)[0]
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Exclude specified destinations
        if exclude_destinations:
            for dest_idx in exclude_destinations:
                if dest_idx < len(similarity_scores):
                    similarity_scores[dest_idx] = -np.inf
        
        # Get top recommendations
        top_dest_indices = np.argsort(similarity_scores)[-n_recommendations:][::-1]
        
        recommendations = [
            (self.reverse_dest_map[idx], float(similarity_scores[idx]))
            for idx in top_dest_indices
            if similarity_scores[idx] > -np.inf
        ]
        
        return recommendations


class FeatureWeightedContentFilter(ContentBasedFilter):
    """Content-based filter with weighted feature importance"""
    
    def __init__(self, feature_weights: Dict[str, float], 
                 similarity_metric: str = 'cosine'):
        """
        Initialize with feature weights
        
        Args:
            feature_weights: Dict mapping feature names to weights
            similarity_metric: Similarity metric to use
        """
        super().__init__(similarity_metric)
        self.feature_weights = feature_weights
    
    def recommend(self, user_idx: int, n_recommendations: int = 10) -> List[Tuple]:
        """
        Generate weighted recommendations
        
        Args:
            user_idx: User index
            n_recommendations: Number of recommendations
            
        Returns:
            List of (destination_id, score) tuples
        """
        # Apply feature weights
        user_features = self.user_feature_matrix[user_idx].copy()
        dest_features = self.dest_feature_matrix.copy()
        
        # Normalize weights
        if self.feature_weights:
            n_features = len(user_features)
            weights = np.ones(n_features)
            
            # Apply custom weights if provided
            for i in range(min(len(weights), len(list(self.feature_weights.values())))):
                weights[i] *= list(self.feature_weights.values())[i]
            
            weights = weights / weights.sum()  # Normalize
            user_features = user_features * weights
            dest_features = dest_features * weights
        
        user_features = user_features.reshape(1, -1)
        
        # Compute weighted similarity
        if self.similarity_metric == 'cosine':
            similarity_scores = cosine_similarity(user_features, dest_features)[0]
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Get user's visited destinations
        user_visited = np.where(self.interaction_matrix[user_idx] > 0)[0]
        similarity_scores[user_visited] = -np.inf
        
        # Get top recommendations
        top_dest_indices = np.argsort(similarity_scores)[-n_recommendations:][::-1]
        
        recommendations = [
            (self.reverse_dest_map[idx], float(similarity_scores[idx]))
            for idx in top_dest_indices
            if similarity_scores[idx] > -np.inf
        ]
        
        return recommendations
