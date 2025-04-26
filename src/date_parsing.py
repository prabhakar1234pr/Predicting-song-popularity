import pandas as pd
import logging

logger = logging.getLogger(__name__)

def parse_release_date(df: pd.DataFrame, date_column: str = "track_album_release_date") -> pd.DataFrame:
    """
    Parse the track_album_release_date column into year, month, and day.
    Keeps original column intact.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")

    logger.info(f"📆 Parsing dates from '{date_column}'...")

    # Convert to datetime safely
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Extract components
    df["release_year"] = df[date_column].dt.year
    df["release_month"] = df[date_column].dt.month
    df["release_day"] = df[date_column].dt.day

    # 🛑 DROP the original datetime column
    df = df.drop(columns=[date_column])

    logger.info("✅ Date parsing complete.")
    return df
