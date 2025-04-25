# src/outlier_cleaner.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

EXCLUDED_NUMERIC_CATEGORICALS = ["key", "mode", "release_year","release_day", "release_month"]  # Add any other numeric categorical columns to exclude

def clean_outliers_ohe_year(X: pd.DataFrame, y: pd.Series, threshold: int = 1000) -> tuple:
    """
    Hybrid IQR-based outlier cleaner:
    - Removes rows if outliers > threshold
    - Caps values if outliers <= threshold
    """
    X_clean = X.copy()
    y_clean = y.copy()
    
    numeric_cols = X_clean.select_dtypes(include='number').columns

# Exclude binary (0/1) one-hot columns + numeric categoricals like 'key' and 'mode'
    numeric_cols = [
    col for col in numeric_cols
    if col not in EXCLUDED_NUMERIC_CATEGORICALS and not set(X_clean[col].dropna().unique()).issubset({0, 1})
    ]

    for col in numeric_cols:

        Q1 = X_clean[col].quantile(0.25)
        Q3 = X_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        mask_outliers = (X_clean[col] < lower) | (X_clean[col] > upper)
        outlier_count = mask_outliers.sum()

        X_clean[col] = np.where(X_clean[col] < lower, lower, X_clean[col])
        X_clean[col] = np.where(X_clean[col] > upper, upper, X_clean[col])
        logger.info(f"🔧 Capped {outlier_count} outliers in '{col}'")

    logger.info(f"✅ Final shape after outlier cleanup: {X_clean.shape}")
    return X_clean, y_clean
