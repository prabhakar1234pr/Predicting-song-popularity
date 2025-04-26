# src/components/evaluator.py

import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# 🧠 ABSTRACT BASE CLASS
# ==========================
class BaseRegressionEvaluator(ABC):
    """Abstract base class for regression model evaluators."""

    @abstractmethod
    def evaluate(self, model, X_test, y_test):
        """Evaluate a model and log metrics."""
        pass

# ==========================
# 🚀 CONCRETE EVALUATOR CLASS
# ==========================
class MLFLOWRegressionEvaluator(BaseRegressionEvaluator):
    """Concrete evaluator that logs metrics to MLflow."""

    def evaluate(self, model, X_test, y_test):
        logger.info("Evaluating model performance...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"📈 Evaluation metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        logger.info("✅ Metrics logged to MLflow successfully.")

        return {"mse": mse, "mae": mae, "r2": r2}
