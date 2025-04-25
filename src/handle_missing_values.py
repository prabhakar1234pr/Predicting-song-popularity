import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

#  Abstract base class for missing value handlers
class MissingValueHandler(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value strategy to a DataFrame."""
        pass

#  Drop rows with missing values in the column(s)
class DropMissingHandler(MissingValueHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(" Dropping rows with missing values.")
        return df.dropna()

#  Fill numeric columns with mean, fallback to mode for categorical
class MeanFillHandler(MissingValueHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                logger.info(f" Filled missing values in '{col}' with mean: {mean_val}")
            else:
                df = ModeFillHandler().handle(df)
        return df

#  Fill numeric columns with median, fallback to mode for categorical
class MedianFillHandler(MissingValueHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f" Filled missing values in '{col}' with median: {median_val}")
            else:
                df = ModeFillHandler().handle(df)
        return df

#  Fill all missing values with a constant
class FillConstantHandler(MissingValueHandler):
    def __init__(self, constant=0):
        self.constant = constant

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Filling missing values with constant: {self.constant}")
        return df.fillna(self.constant)

# 🔁 Fill categorical (object) columns with mode
class ModeFillHandler(MissingValueHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                try:
                    mode_val = df[col].mode().dropna().iloc[0]
                except IndexError:
                    mode_val = "missing"
                df[col] = df[col].fillna(mode_val)
                logger.info(f" Filled missing values in '{col}' with mode: {mode_val}")
        return df

# 🧩 Strategy executor
class MissingHandlerChooser:
    def __init__(self, strategy: MissingValueHandler):
        self.strategy = strategy

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.handle(df)
