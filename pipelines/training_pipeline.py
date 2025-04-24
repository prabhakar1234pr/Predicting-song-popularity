from zenml import pipeline, step, Model
from steps.data_ingestion_step import data_ingestion_step

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
    raw_data = data_ingestion_step(file_path="C:\predicting_song_popularity\Data/archive (7).zip")  # Replace with your zip file path

