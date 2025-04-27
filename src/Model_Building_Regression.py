# src/Model_Building_Regression.py

import warnings
import logging
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin

# 🚀 SETUP LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# 🛠️ SAFE IMPORTS
try:
    from xgboost import XGBRegressor
    logger.info("Successfully imported XGBoost.")
except ImportError:
    XGBRegressor = None
    logger.warning("XGBoost library not found. XGBoost models will be unavailable.")

try:
    from lightgbm import LGBMRegressor
    logger.info("Successfully imported LightGBM.")
except ImportError:
    LGBMRegressor = None
    logger.warning("LightGBM library not found. LightGBM models will be unavailable.")

# 🛠️ ABSTRACT BASE CLASS
class RegressionModel(ABC):
    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Build and return a sklearn pipeline."""
        pass

# 🚀 CONCRETE MODEL CLASSES
class LinearRegressionModel(RegressionModel):
    def build_pipeline(self) -> RegressorMixin:
        logger.info("Building Linear Regression pipeline.")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])

class RidgeRegressionModel(RegressionModel):
    def build_pipeline(self) -> RegressorMixin:
        logger.info("Building Ridge Regression pipeline.")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge())
        ])

class RandomForestModel(RegressionModel):
    def build_pipeline(self) -> RegressorMixin:
        logger.info("Building Random Forest pipeline.")
        return Pipeline([
            ("regressor", RandomForestRegressor())
        ])

class XGBoostModel(RegressionModel):
    def build_pipeline(self) -> RegressorMixin:
        if XGBRegressor is None:
            logger.error("Attempted to build XGBoost model but xgboost is not installed.")
            raise ImportError("XGBoost library is not installed. Please install xgboost.")

        logger.info("Building XGBoost Regression pipeline.")
        return Pipeline([
            ("regressor", XGBRegressor())
        ])

class LightGBMModel(RegressionModel):
    def build_pipeline(self) -> RegressorMixin:
        if LGBMRegressor is None:
            logger.error("Attempted to build LightGBM model but lightgbm is not installed.")
            raise ImportError("LightGBM library is not installed. Please install lightgbm.")

        logger.info("Building LightGBM Regression pipeline.")
        return Pipeline([
            ("regressor", LGBMRegressor())
        ])

# 🏗️ MODEL FACTORY CLASS
class ModelFactory:
    def __init__(self):
        self.models = {
            "linear_regression": LinearRegressionModel(),
            "ridge_regression": RidgeRegressionModel(),
            "random_forest": RandomForestModel(),
            "xgboost": XGBoostModel(),
            "lightgbm": LightGBMModel(),
        }
        logger.info("ModelFactory initialized with models: %s", list(self.models.keys()))

    def get_model_builder(self, model_name: str) -> RegressionModel:
        model_name = model_name.lower()
        if model_name not in self.models:
            logger.error(f"Requested unknown model '{model_name}'. Available models: {list(self.models.keys())}")
            raise ValueError(f"Model '{model_name}' not recognized. Available models are: {list(self.models.keys())}")

        logger.info(f"Fetching model builder for: {model_name}")
        return self.models[model_name]

