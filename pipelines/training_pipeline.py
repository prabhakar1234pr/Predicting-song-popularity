import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from zenml import pipeline, Model

# Import your steps
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_duplicates_step import handle_duplicates_step
from steps.parse_dates_step import parse_date_step
from steps.handle_missing_values_step import handle_missing_step
from steps.split_data_step import split_data_step
from steps.encoder_step import encoder_step
from steps.transform_features_step import transform_features_step
from steps.remove_outliers_step import clean_outliers_step
from steps.regression_models_step import train_regression_step
from steps.diagnostic_step import diagnostics_step

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s"
)

@pipeline(
    model=Model(
        name="Song-Popularity-Predictor",
        version="1.0.0",
        description="A model to predict song popularity based on various audio and metadata features."
    )
)
def ml_pipeline():
    """
    A machine learning pipeline that orchestrates the steps for data ingestion,
    preprocessing, feature engineering, model training, and diagnostics.
    """

    # 1. Ingest raw data
    raw_data = data_ingestion_step(file_path="C:/predicting_song_popularity/Data/archive (7).zip")

    # 2. Handle duplicate rows
    data_no_duplicates = handle_duplicates_step(df=raw_data)

    # 3. Parse date columns into release_year, release_month, release_day
    date_parsed_data = parse_date_step(df=data_no_duplicates)

    # 4. Handle missing values (after parsing dates!)
    cleaned_data = handle_missing_step(df=date_parsed_data)

    # 5. Split into training and testing sets
    X_train, X_test, y_train, y_test = split_data_step(df=cleaned_data, target_column="track_popularity")

    # 6. Encode categorical features
    X_train_encoded, X_test_encoded = encoder_step(X_train=X_train, X_test=X_test)

    # 7. Feature transformation (handle skewness)
    X_train_transformed, X_test_transformed = transform_features_step(X_train=X_train_encoded, X_test=X_test_encoded)

    # 8. Quick diagnostics after feature transformation
    diagnostics_step(X=X_train_transformed)

    # 9. Clean outliers
    X_train_cleaned, y_train_cleaned = clean_outliers_step(X_train=X_train_transformed, y_train=y_train)

    # 10. Train different regression models with hyperparameter tuning
    best_model_rf = train_regression_step(X_train=X_train_cleaned, y_train=y_train_cleaned, model_name="random_forest")
    best_model_linear = train_regression_step(X_train=X_train_cleaned, y_train=y_train_cleaned, model_name="linear_regression")
    best_model_ridge = train_regression_step(X_train=X_train_cleaned, y_train=y_train_cleaned, model_name="ridge_regression")
    best_model_xgb = train_regression_step(X_train=X_train_cleaned, y_train=y_train_cleaned, model_name="xgboost")
    best_model_lgbm = train_regression_step(X_train=X_train_cleaned, y_train=y_train_cleaned, model_name="lightgbm")

if __name__ == "__main__":
    ml_pipeline()
