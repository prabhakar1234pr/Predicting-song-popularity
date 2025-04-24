from zenml import step
import pandas as pd
from src.handle_duplicates import handle_track_id_duplicates
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@step
def handle_duplicates_step(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🚀 Starting duplicate handling step...")
    cleaned_df = handle_track_id_duplicates(df, verbose=True)
    logger.info("✅ Duplicate handling completed.")
    return cleaned_df
