# steps/regression_model_step.py

import logging
from abc import ABC, abstractmethod
import mlflow
import optuna
from optuna.integration import OptunaSearchCV
from mlflow.models.signature import infer_signature
from zenml import step
from zenml.client import Client
from zenml import Model

from src.Model_Building_Regression import ModelFactory

# 🚀 SETUP LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🧩 ACTIVE EXPERIMENT TRACKER AND MODEL
experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="Song-Popularity-Predictor",
    version="1.0.0",
    license="Apache 2.0",
    description="Predicts song popularity before release using metadata and audio features."
)

# 🛠️ ABSTRACT BASE CLASS
class BaseRegressionTrainer(ABC):
    @abstractmethod
    def train(self, X_train, y_train, model_name: str):
        pass

# 🚀 CONCRETE TRAINER CLASS
class OptunaRegressionTrainer(BaseRegressionTrainer):
    def train(self, X_train, y_train, model_name: str = "random_forest"):
        logger.info(f"Starting training for model: {model_name}")

        mlflow.set_experiment("Song_Popularity_Prediction_Experiment")

        # Start new or nested MLflow run
        if not mlflow.active_run():
            mlflow.start_run(run_name=f"{model_name}_optuna_tuning")
        else:
            mlflow.start_run(run_name=f"{model_name}_optuna_tuning", nested=True)

        mlflow.sklearn.autolog()

        try:
            factory = ModelFactory()
            model_builder = factory.get_model_builder(model_name)
            base_pipeline = model_builder.build_pipeline()

            logger.info("Model pipeline built successfully.")

            param_distributions = self.get_param_distributions(model_name)

            search = OptunaSearchCV(
                base_pipeline,
                param_distributions,
                n_trials=30,
                cv=3,
                scoring='r2',
                random_state=42,
                n_jobs=-1
            )

            logger.info("Starting Optuna hyperparameter tuning...")
            search.fit(X_train, y_train)
            logger.info("Optuna tuning completed.")

            best_model = search.best_estimator_
            best_params = search.best_params_

            # 📝 Log hyperparameters (even though autolog does it, explicit is better)
            mlflow.log_params(best_params)

            # 🏷️ Log metadata tags
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("optuna_n_trials", len(search.study_.trials))

            # ✨ Log input features (for transparency)
            try:
                feature_names = X_train.columns.tolist()
                mlflow.log_text("\n".join(feature_names), artifact_file="input_features.txt")
                logger.info(f"Logged input features: {feature_names}")
            except Exception as e:
                logger.warning(f"Could not log feature names: {e}")

            # 💥 Log the model manually
            sample_input = X_train[:5]
            signature = infer_signature(X_train, best_model.predict(X_train))

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                input_example=sample_input,
                signature=signature
            )

            logger.info("✅ Best model logged to MLflow successfully.")

            return best_model

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise e

        finally:
            mlflow.end_run()

    def get_param_distributions(self, model_name: str):
        model_name = model_name.lower()

        if model_name == "linear_regression":
            return {}

        elif model_name == "ridge_regression":
            return {
                "regressor__alpha": optuna.distributions.FloatDistribution(1e-3, 1e2, log=True)
            }

        elif model_name == "random_forest":
            return {
                "regressor__n_estimators": optuna.distributions.IntDistribution(50, 300),
                "regressor__max_depth": optuna.distributions.IntDistribution(3, 20),
                "regressor__min_samples_split": optuna.distributions.IntDistribution(2, 10)
            }

        elif model_name == "xgboost":
            return {
                "regressor__n_estimators": optuna.distributions.IntDistribution(50, 300),
                "regressor__max_depth": optuna.distributions.IntDistribution(3, 12),
                "regressor__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                "regressor__subsample": optuna.distributions.FloatDistribution(0.5, 1.0)
            }

        elif model_name == "lightgbm":
            return {
                "regressor__n_estimators": optuna.distributions.IntDistribution(50, 300),
                "regressor__max_depth": optuna.distributions.IntDistribution(3, 12),
                "regressor__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                "regressor__num_leaves": optuna.distributions.IntDistribution(20, 150)
            }

        else:
            logger.error(f"No hyperparameter search space defined for model: {model_name}")
            raise ValueError(f"No hyperparameter search space defined for model: {model_name}")

# 🚀 ZENML STEP
@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def train_regression_step(X_train, y_train, model_name: str = "random_forest"):
    trainer = OptunaRegressionTrainer()
    best_model = trainer.train(X_train, y_train, model_name)
    return best_model


