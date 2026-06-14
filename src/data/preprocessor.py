"""Data preprocessing and cleaning utilities"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Any


class TravelDataPreprocessor:
    """Handles data cleaning, encoding, and scaling for travel recommender system"""
    
    def __init__(self, scaling_method: str = "standard"):
        """
        Initialize preprocessor
        
        Args:
            scaling_method: Type of scaling - 'standard', 'minmax', or 'robust'
        """
        self.scaling_method = scaling_method
        self.scalers = {}
        self.encoders = {}
        
        if scaling_method == "standard":
            self.scaler_class = StandardScaler
        elif scaling_method == "minmax":
            self.scaler_class = MinMaxScaler
        elif scaling_method == "robust":
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Handle missing values in dataframe
        
        Args:
            df: Input dataframe
            strategy: 'mean', 'median', or 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "drop":
            df = df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Fill categorical with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: list, 
                          fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical column names
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                if col not in self.encoders:
                    raise ValueError(f"Encoder for {col} not fitted. Call with fit=True first.")
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_numeric(self, df: pd.DataFrame, numeric_cols: list,
                     fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df: Input dataframe
            numeric_cols: List of numeric column names
            fit: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            DataFrame with scaled numeric features
        """
        df = df.copy()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            if fit:
                self.scalers[col] = self.scaler_class()
                df[[col]] = self.scalers[col].fit_transform(df[[col]])
            else:
                if col not in self.scalers:
                    raise ValueError(f"Scaler for {col} not fitted. Call with fit=True first.")
                df[[col]] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def preprocess_users(self, users_df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess user data
        
        Args:
            users_df: User dataframe
            fit: Whether to fit encoders/scalers
            
        Returns:
            Preprocessed user dataframe
        """
        df = users_df.copy()
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy="mean")
        
        # Identify column types
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        if 'user_id' in categorical_cols:
            categorical_cols.remove('user_id')
        if 'user_id' in numeric_cols:
            numeric_cols.remove('user_id')
        
        # Encode and scale
        df = self.encode_categorical(df, categorical_cols, fit=fit)
        df = self.scale_numeric(df, numeric_cols, fit=fit)
        
        return df
    
    def preprocess_destinations(self, destinations_df: pd.DataFrame, 
                               fit: bool = True) -> pd.DataFrame:
        """
        Preprocess destination data
        
        Args:
            destinations_df: Destination dataframe
            fit: Whether to fit encoders/scalers
            
        Returns:
            Preprocessed destination dataframe
        """
        df = destinations_df.copy()
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy="mean")
        
        # Identify column types
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID and name columns
        if 'destination_id' in categorical_cols:
            categorical_cols.remove('destination_id')
        if 'name' in categorical_cols:
            categorical_cols.remove('name')
        
        # Encode and scale
        df = self.encode_categorical(df, categorical_cols, fit=fit)
        df = self.scale_numeric(df, numeric_cols, fit=fit)
        
        return df
    
    def preprocess_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess interaction data
        
        Args:
            interactions_df: Interactions dataframe
            
        Returns:
            Preprocessed interactions dataframe
        """
        df = interactions_df.copy()
        
        # Ensure rating is numeric and within valid range
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])
        df['rating'] = df['rating'].clip(1, 5)  # Clip to 1-5 range
        
        return df
