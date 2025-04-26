import pandas as pd
from zenml import step
import logging
from typing import Annotated
from src.handle_missing_values import (
    DropMissingHandler,
    MeanFillHandler,
    MedianFillHandler,
    ModeFillHandler,
    FillConstantHandler,
    MissingHandlerChooser,
)

# 🔧 Default hard-coded strategy map for known columns
default_strategy = {
    # Numerical
    "track_popularity": "mean",
    "danceability": "mean",
    "energy": "mean",
    "key": "median",
    "loudness": "mean",
    "mode": "mode",
    "speechiness": "mean",
    "acousticness": "mean",
    "instrumentalness": "median",
    "liveness": "mean",
    "valence": "mean",
    "tempo": "mean",
    "duration_ms": "median",

    # Categorical
    "track_id": "drop",
    "track_name": "mode",
    "track_artist": "mode",
    "track_album_id": "mode",
    "track_album_name": "mode",
    "release_year": "median",
    "release_month": "median",
    "release_day": "median",
    "playlist_name": "mode",
    "playlist_id": "mode",
    "playlist_genre": "mode",
    "playlist_subgenre": "mode"
}

# 🧠 Dynamic strategy generator
def generate_strategy_map(df: pd.DataFrame, threshold: int = 100) -> dict:
    strategy_map = {}
    missing_counts = df.isnull().sum()

    for col in df.columns:
        if missing_counts[col] == 0:
            continue
        elif missing_counts[col] < threshold:
            strategy_map[col] = "drop"
        else:
            if col not in default_strategy:
                raise ValueError(f"Missing value handling strategy not defined for column: '{col}'")
            strategy_map[col] = default_strategy[col]

    logging.info(f"📋 Auto-generated strategy map: {strategy_map}")
    return strategy_map


@step
def handle_missing_step(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("🔍 Generating strategy map for handling missing values...")

    strategy_map = generate_strategy_map(df)

    for col, strategy_name in strategy_map.items():
        logger.info(f"🛠 Applying strategy '{strategy_name}' for column: {col}")

        if strategy_name == "drop":
            df = df[df[col].notnull()]
        else:
            if strategy_name == "mean":
                strategy = MeanFillHandler()
            elif strategy_name == "median":
                strategy = MedianFillHandler()
            elif strategy_name == "mode":
                strategy = ModeFillHandler()
            elif strategy_name == "constant":
                strategy = FillConstantHandler()
            else:
                raise ValueError(f"Unknown strategy '{strategy_name}' for column '{col}'")

            handler = MissingHandlerChooser(strategy)
            df[[col]] = handler.apply(df[[col]])

    logger.info("✅ Missing value handling complete.")
    return df
