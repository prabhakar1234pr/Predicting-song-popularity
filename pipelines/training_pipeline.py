import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import logging
from zenml import pipeline, step, Model
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_duplicates_step import handle_duplicates_step
from steps.handle_missing_values_step import handle_missing_step
from steps.parse_dates_step import parse_dates_step
from steps.split_data_step import split_data_step
from steps.encoder_step import encoder_step
from steps.transform_features_step import transform_features_step
from steps.remove_outliers_step import clean_outliers_step



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s"
)


@pipeline(
    model=Model(name="Song-Popularity-Predictor", version="1.0.0", description="A model to predict song popularity based on various features.")
)

def ml_pipeline():
    """
    A machine learning pipeline that orchestrates the steps for data ingestion,
    preprocessing, model training, and evaluation.
    """
    # Define the steps in the pipeline
    #data_ingestion_step(
    raw_data = data_ingestion_step(file_path="C:/predicting_song_popularity/Data/archive (7).zip")  # Replace with your zip file path
    data_no_duplicates = handle_duplicates_step(df=raw_data)
    cleaned_data = handle_missing_step(df=data_no_duplicates)  # You can change the strategy as needed
    date_parse_data = parse_dates_step(df=cleaned_data)
    X_train, X_test, y_train, y_test = split_data_step(df=date_parse_data, target_column="track_popularity")
    X_train_encoded, X_test_encoded = encoder_step(X_train=X_train, X_test=X_test)
    X_train_transformed, X_test_tranformed = transform_features_step(X_train=X_train_encoded, X_test=X_test_encoded)
    X_train_cleaned, y_train_cleaned = clean_outliers_step(X_train=X_train_transformed, y_train=y_train)





if __name__ == "__main__":
    ml_pipeline()
    