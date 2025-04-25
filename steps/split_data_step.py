import pandas as pd
import numpy as np
import logging
from zenml import step
from src.data_splitter import SklearnDataSplitter
from typing import Tuple


@step
def split_data_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""
    splitter = SklearnDataSplitter()
    X_train, X_test, y_train, y_test = splitter.split(df, target_column=target_column)
    
    # Convert Series to numpy arrays for better ZenML serialization
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()
    
    return X_train, X_test, y_train_array, y_test_array

