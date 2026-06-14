"""Feature engineering for travel recommender system"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class TravelFeatureEngineer:
    """Extracts and transforms features for recommendation models"""
    
    @staticmethod
    def engineer_user_features(users_df: pd.DataFrame, 
                              interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional user features from interaction history
        
        Args:
            users_df: User profile dataframe
            interactions_df: User-destination interaction dataframe
            
        Returns:
            Dataframe with engineered user features
        """
        user_features = users_df.copy()
        
        # Aggregate interaction statistics per user
        interaction_stats = interactions_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'destination_id': 'nunique'
        }).reset_index()
        
        interaction_stats.columns = ['user_id', 'trip_count', 'avg_rating', 
                                    'rating_variance', 'min_rating', 'max_rating', 
                                    'unique_destinations']
        
        # Merge features
        user_features = user_features.merge(interaction_stats, on='user_id', how='left')
        
        # Fill missing values for new users
        user_features['trip_count'] = user_features['trip_count'].fillna(0)
        user_features['avg_rating'] = user_features['avg_rating'].fillna(3.0)
        user_features['unique_destinations'] = user_features['unique_destinations'].fillna(0)
        user_features['rating_variance'] = user_features['rating_variance'].fillna(0)
        
        # Create user engagement level
        user_features['engagement_level'] = pd.cut(user_features['trip_count'], 
                                                   bins=[0, 2, 5, 10, np.inf],
                                                   labels=['low', 'medium', 'high', 'very_high'])
        
        return user_features
    
    @staticmethod
    def engineer_destination_features(destinations_df: pd.DataFrame,
                                     interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional destination features from ratings
        
        Args:
            destinations_df: Destination metadata dataframe
            interactions_df: User-destination interaction dataframe
            
        Returns:
            Dataframe with engineered destination features
        """
        destination_features = destinations_df.copy()
        
        # Aggregate ratings per destination
        rating_stats = interactions_df.groupby('destination_id').agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique'
        }).reset_index()
        
        rating_stats.columns = ['destination_id', 'num_reviews', 'avg_rating',
                               'rating_std', 'num_visitors']
        
        # Merge features
        destination_features = destination_features.merge(rating_stats, 
                                                         on='destination_id', 
                                                         how='left')
        
        # Fill missing values for new destinations
        destination_features['num_reviews'] = destination_features['num_reviews'].fillna(0)
        destination_features['avg_rating'] = destination_features['avg_rating'].fillna(3.0)
        destination_features['num_visitors'] = destination_features['num_visitors'].fillna(0)
        destination_features['rating_std'] = destination_features['rating_std'].fillna(0)
        
        # Create popularity score (normalized)
        max_visitors = destination_features['num_visitors'].max()
        if max_visitors > 0:
            destination_features['popularity_score'] = (
                destination_features['num_visitors'] / max_visitors
            )
        else:
            destination_features['popularity_score'] = 0.5
        
        return destination_features
    
    @staticmethod
    def compute_user_destination_similarity(user_features: pd.DataFrame,
                                           destination_features: pd.DataFrame,
                                           feature_cols: Dict[str, list]) -> np.ndarray:
        """
        Compute similarity matrix between users and destinations
        
        Args:
            user_features: Engineered user features
            destination_features: Engineered destination features
            feature_cols: Dict mapping 'user' and 'destination' to their feature columns
            
        Returns:
            Similarity matrix of shape (num_users, num_destinations)
        """
        user_feature_matrix = user_features[feature_cols['user']].values
        dest_feature_matrix = destination_features[feature_cols['destination']].values
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(user_feature_matrix, dest_feature_matrix)
        
        return similarity_matrix
    
    @staticmethod
    def create_interaction_matrix(interactions_df: pd.DataFrame,
                                 users_df: pd.DataFrame,
                                 destinations_df: pd.DataFrame) -> np.ndarray:
        """
        Create user-destination interaction matrix
        
        Args:
            interactions_df: User-destination interactions
            users_df: User data
            destinations_df: Destination data
            
        Returns:
            Interaction matrix of shape (num_users, num_destinations)
        """
        # Create mapping of IDs to indices
        user_id_map = {uid: idx for idx, uid in enumerate(users_df['user_id'].unique())}
        dest_id_map = {did: idx for idx, did in enumerate(destinations_df['destination_id'].unique())}
        
        # Initialize matrix
        interaction_matrix = np.zeros((len(user_id_map), len(dest_id_map)))
        
        # Fill matrix with ratings
        for _, row in interactions_df.iterrows():
            user_idx = user_id_map[row['user_id']]
            dest_idx = dest_id_map[row['destination_id']]
            interaction_matrix[user_idx, dest_idx] = row['rating']
        
        return interaction_matrix, user_id_map, dest_id_map
    
    @staticmethod
    def get_user_profiles(user_features: pd.DataFrame,
                         feature_cols: list) -> pd.DataFrame:
        """
        Extract user preference profiles
        
        Args:
            user_features: Engineered user features
            feature_cols: List of feature columns to include
            
        Returns:
            User profiles dataframe
        """
        profiles = user_features[['user_id'] + feature_cols].copy()
        return profiles
    
    @staticmethod
    def get_destination_profiles(destination_features: pd.DataFrame,
                                feature_cols: list) -> pd.DataFrame:
        """
        Extract destination characteristic profiles
        
        Args:
            destination_features: Engineered destination features
            feature_cols: List of feature columns to include
            
        Returns:
            Destination profiles dataframe
        """
        profiles = destination_features[['destination_id', 'name'] + feature_cols].copy()
        return profiles
