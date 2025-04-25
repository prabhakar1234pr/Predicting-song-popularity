import pandas as pd
import logging
from zenml import step
from src.outlier_detector import remove_outliers_iqr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def remove_outliers_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML step to detect and handle outliers using a hybrid IQR strategy:
    - For numeric columns (excluding known categorical-like numerics):
        • If number of outliers > 1000, remove them.
        • If number of outliers <= 1000, cap the values at IQR bounds.
    """
    cleaned_df = remove_outliers_iqr(df)
    logger.info(f"✅ Outlier handling done in step. Final shape: {cleaned_df.shape}")
    return cleaned_df

