# steps/leaderboard_step.py

import logging
from zenml import step
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def leaderboard_step(models_and_metrics: dict, X_test, y_test):
    """Leaderboard Step to auto-select and log the best model based on R2 score."""

    best_model_name = None
    best_r2_score = float('-inf')
    best_model = None

    for model_name, (model_artifact, metrics) in models_and_metrics.items():
        model = model_artifact.load()  # ✅ Load model INSIDE the step
        r2_score = metrics["r2_score"]

        logger.info(f"Model: {model_name} | R2 Score: {r2_score:.4f}")

        if r2_score > best_r2_score:
            best_model_name = model_name
            best_r2_score = r2_score
            best_model = model

    if best_model is None:
        raise ValueError("No valid model found.")

    logger.info(f"🏆 Best Model Selected: {best_model_name} with R2 Score: {best_r2_score:.4f}")

    # ✅ Log the best model to MLflow
    mlflow.set_experiment("Song_Popularity_Leaderboard")
    with mlflow.start_run(run_name=f"leaderboard_best_model_{best_model_name}"):
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_r2_score", best_r2_score)

        # You can also log the final model here if you want
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

