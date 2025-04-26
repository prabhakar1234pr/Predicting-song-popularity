# steps/evaluator_step.py

import logging
from zenml import step
from src.evaluator import MLFLOWRegressionEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# 🚀 ZENML EVALUATOR STEP
# ==========================
@step
def evaluate_regression_model(model, X_test, y_test) -> dict:
    """ZenML Step to evaluate a regression model and log metrics."""

    evaluator = MLFLOWRegressionEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)

    logger.info(f"🏆 Evaluation complete: {metrics}")

    return metrics
