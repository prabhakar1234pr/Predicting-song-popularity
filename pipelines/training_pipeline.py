import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import logging
from zenml import pipeline, step, Model
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_duplicates_step import handle_duplicates_step
from steps.handle_missing_values_step import handle_missing_step


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
    cleaned_data = handle_missing_step(df=data_no_duplicates, strategy_name="drop")  # You can change the strategy as needed


    