# steps/clean_outliers_step.py

import pandas as pd
import numpy as np
from zenml import step
from typing import Tuple
from src.outlier_detector import clean_outliers_ohe_year

@step

def clean_outliers_step(X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    ZenML step to clean outliers using a hybrid IQR approach.
    Accepts and returns NumPy arrays for y_train to match ZenML expectations.
    """
    y_train_series = pd.Series(y_train, index=X_train.index)
    X_clean, y_clean = clean_outliers_ohe_year(X_train, y_train_series)
    return X_clean, y_clean.to_numpy()



