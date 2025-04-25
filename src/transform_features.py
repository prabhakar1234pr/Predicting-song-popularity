import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Exclude these numeric-like categorical features
NUMERIC_CATEGORICALS = ["key", "mode"]

class FeatureTransformer(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class LogTransformer(FeatureTransformer):
    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold
        self.cols_to_transform = []

    def fit(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include='number').columns
        numeric_cols = [col for col in numeric_cols if col not in NUMERIC_CATEGORICALS]

        for col in numeric_cols:
            if set(df[col].dropna().unique()).issubset({0, 1}):
                continue  # Skip one-hot encoded columns
            skewness = df[col].skew()
            if skewness > self.skew_threshold:
                self.cols_to_transform.append(col)
                logger.info(f"📊 Will apply log1p to '{col}' (skewness={skewness:.2f})")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in self.cols_to_transform:
            df_copy[col] = np.log1p(df_copy[col])
            logger.info(f"🔁 Transformed '{col}' using log1p.")
        return df_copy

