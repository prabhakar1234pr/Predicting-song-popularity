import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_track_id_duplicates(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    initial_shape = df.shape
    duplicate_count = df.duplicated(subset=['track_id'], keep='first').sum()
    df_cleaned = df.drop_duplicates(subset=['track_id'], keep='first')

    if verbose:
        logger.info(f"🔍 Found {duplicate_count} duplicate track_id entries.")
        logger.info(f"✅ Removed duplicates. New shape: {df_cleaned.shape} (was {initial_shape})")

    return df_cleaned

