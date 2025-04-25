import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_ingestor(file_path)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)
    return df
