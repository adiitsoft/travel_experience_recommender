"""Data loading utilities for travel recommender system"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


class TravelDataLoader:
    """Loads and validates travel experience data"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path)
        
    def load_users(self) -> pd.DataFrame:
        """Load user profile data"""
        users_file = self.data_path / "users.csv"
        if not users_file.exists():
            raise FileNotFoundError(f"Users file not found: {users_file}")
        
        users_df = pd.read_csv(users_file)
        print(f"Loaded {len(users_df)} user profiles")
        return users_df
    
    def load_destinations(self) -> pd.DataFrame:
        """Load destination metadata"""
        destinations_file = self.data_path / "destinations.csv"
        if not destinations_file.exists():
            raise FileNotFoundError(f"Destinations file not found: {destinations_file}")
        
        destinations_df = pd.read_csv(destinations_file)
        print(f"Loaded {len(destinations_df)} destinations")
        return destinations_df
    
    def load_interactions(self) -> pd.DataFrame:
        """Load user-destination interactions (ratings, visits, reviews)"""
        interactions_file = self.data_path / "interactions.csv"
        if not interactions_file.exists():
            raise FileNotFoundError(f"Interactions file not found: {interactions_file}")
        
        interactions_df = pd.read_csv(interactions_file)
        print(f"Loaded {len(interactions_df)} user-destination interactions")
        return interactions_df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data files"""
        users = self.load_users()
        destinations = self.load_destinations()
        interactions = self.load_interactions()
        
        return users, destinations, interactions
    
    @staticmethod
    def validate_data(users: pd.DataFrame, destinations: pd.DataFrame, 
                     interactions: pd.DataFrame) -> bool:
        """
        Validate data integrity
        
        Args:
            users: User profile dataframe
            destinations: Destination metadata dataframe
            interactions: User-destination interaction dataframe
            
        Returns:
            bool: True if data is valid
        """
        # Check required columns
        required_user_cols = {'user_id', 'age_group', 'location'}
        required_dest_cols = {'destination_id', 'name', 'climate_zone'}
        required_interact_cols = {'user_id', 'destination_id', 'rating'}
        
        if not required_user_cols.issubset(users.columns):
            raise ValueError(f"Missing required user columns: {required_user_cols - set(users.columns)}")
        
        if not required_dest_cols.issubset(destinations.columns):
            raise ValueError(f"Missing required destination columns: {required_dest_cols - set(destinations.columns)}")
        
        if not required_interact_cols.issubset(interactions.columns):
            raise ValueError(f"Missing required interaction columns: {required_interact_cols - set(interactions.columns)}")
        
        # Check for missing values in key fields
        if users['user_id'].isnull().any():
            raise ValueError("User IDs contain missing values")
        
        if destinations['destination_id'].isnull().any():
            raise ValueError("Destination IDs contain missing values")
        
        if interactions[['user_id', 'destination_id']].isnull().any().any():
            raise ValueError("Interactions contain missing user_id or destination_id")
        
        print("✓ Data validation passed")
        return True


def load_travel_data(data_path: str = "data/raw/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all travel data
    
    Args:
        data_path: Path to raw data directory
        
    Returns:
        Tuple of (users, destinations, interactions) dataframes
    """
    loader = TravelDataLoader(data_path)
    users, destinations, interactions = loader.load_all()
    TravelDataLoader.validate_data(users, destinations, interactions)
    return users, destinations, interactions
