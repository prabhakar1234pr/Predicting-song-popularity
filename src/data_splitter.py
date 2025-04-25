import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

# Abstract base class
class DataSplitter(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets.
        
        Args:
            df: Input DataFrame containing features and target
            target_column: Name of the target column to predict
            
        Returns:
            Tuple containing:
            - X_train: Training features DataFrame
            - X_test: Testing features DataFrame
            - y_train: Training target Series
            - y_test: Testing target Series
        """
        pass

# Concrete class
class SklearnDataSplitter(DataSplitter):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data using scikit-learn's train_test_split.
        
        Args:
            df: Input DataFrame containing features and target
            target_column: Name of the target column to predict
            
        Returns:
            Tuple containing:
            - X_train: Training features DataFrame
            - X_test: Testing features DataFrame
            - y_train: Training target Series
            - y_test: Testing target Series
        """
        drop_cols = [
            "track_id", "track_name", "track_artist",
            "track_album_id", "track_album_name",
            "playlist_name", "playlist_id"
        ]
        df = df.drop(columns=drop_cols, errors="ignore")
        logger.info(f"🗑️ Dropped unnecessary columns: {drop_cols}")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        logger.info("✅ Separated features and target variable.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"🔀 Split data into train and test sets: {X_train.shape}, {X_test.shape}")

        return X_train, X_test, y_train, y_test


