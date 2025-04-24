import os
import zipfile
import pandas as pd
from abc import ABC, abstractmethod

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame: #output type is DataFrame
        """
        Ingest data from a file and return a DataFrame.
        """
        pass


class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingest data from a zip file and return a DataFrame.
        """
        if not zipfile.is_zipfile(file_path):
            raise ValueError(f"{file_path} is not a valid zip file.") # Check if the file is a zip file

        with zipfile.ZipFile(file_path, 'r') as z:
            z.extractall("extracted_data") # Extract the contents of the zip file

        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith('.csv')] # Filter for CSV files
        if not csv_files:
            raise ValueError("No CSV files found in the zip archive.") # Check if there are any CSV files in the extracted contents
            
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found in the zip archive. Please provide a single CSV file.")# Ensure only one CSV file is present
            
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)# Read the CSV file into a DataFrame

        
        return df
    

class DataIngestorFactory:
    @staticmethod
    def get_ingestor(file_path: str) -> DataIngestor:
        """
        Factory method to get the appropriate DataIngestor based on the file type.
        """
        if zipfile.is_zipfile(file_path):
            return ZipDataIngestor()
        else:
            raise ValueError(f"No suitable ingestor found for {file_path}.") # Raise an error if no suitable ingestor is found for the file type
        

if __name__ == "__main__":
    # Example usage
    #file_path = "data.zip"  # Replace with your zip file path
    #ingestor = DataIngestorFactory.get_ingestor(file_path)
    #df = ingestor.ingest(file_path)
    #print(df.head())  # Display the first few rows of the DataFrame
    # Ensure the zip file contains a single CSV file    
    pass