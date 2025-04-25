import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

EXCLUDED_NUMERIC_CATEGORICALS = ["key", "mode"]

def remove_outliers_iqr(df: pd.DataFrame, threshold: int = 1000) -> pd.DataFrame:
    """
    Apply hybrid outlier treatment using IQR:
    - Remove rows if outliers > threshold
    - Cap outliers if <= threshold
    """
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col not in EXCLUDED_NUMERIC_CATEGORICALS]
    df_clean = df.copy()

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        mask_outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
        outlier_count = mask_outliers.sum()

        if outlier_count > threshold:
            df_clean = df_clean[~mask_outliers]
            logger.info(f"🗑️ Removed {outlier_count} outliers from '{col}' using IQR.")
        else:
            original_col = df_clean[col].copy()
            df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
            logger.info(f"🔧 Capped {outlier_count} outliers in '{col}' using IQR bounds.")

    logger.info(f"✅ Hybrid outlier treatment complete. Final shape: {df_clean.shape}")
    return df_clean
