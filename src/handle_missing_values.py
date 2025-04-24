import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# 🧠 Abstract Base Class for missing value strategies
class MissingValueHandler(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value strategy and return the cleaned DataFrame."""
        pass


# 🚮 Strategy 1: Drop all rows with any missing values
class DropMissingHandler(MissingValueHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Dropping rows with any missing values.")
        return df.dropna()


# 🧮 Strategy 2: Fill numeric columns with mean, categorical with mode
class MeanFillHandler(MissingValueHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filling numeric columns with mean, categorical with mode.")

        # Fill numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # Fill categorical columns with mode
        df = ModeFillHandler().handle(df)

        return df


# 📐 Strategy 3: Fill numeric columns with median, categorical with mode
class MedianFillHandler(MissingValueHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filling numeric columns with median, categorical with mode.")

        # Fill numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Fill categorical columns with mode
        df = ModeFillHandler().handle(df)

        return df


# 🧱 Strategy 4: Fill ALL missing values with a constant
class FillConstantHandler(MissingValueHandler):
    def __init__(self, constant=0):
        self.constant = constant

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Filling ALL missing values with constant: {self.constant}")
        return df.fillna(self.constant)


# 🎯 Strategy 5: Fill ONLY categorical features with mode
class ModeFillHandler(MissingValueHandler):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filling categorical (object) columns with mode.")

        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                try:
                    mode_val = df[col].mode().dropna().iloc[0]
                except IndexError:
                    mode_val = "missing"
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Filled missing values in '{col}' with mode: {mode_val}")

        return df


# 🧩 Utility to choose and apply any strategy
class MissingHandlerChooser:
    def __init__(self, strategy: MissingValueHandler):
        self.strategy = strategy

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.handle(df)
