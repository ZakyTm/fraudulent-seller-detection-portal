"""
Data Processor Module for Fraudulent Seller Detection Portal

This module handles all aspects of data input, processing, and management,
including:
- File upload system with drag-and-drop and preview
- Support for multiple file formats (CSV, Excel, JSON)
- Chunked file processing for large datasets
- Demo data management and synthetic data generation
- Data quality assessment and reporting

Author: Manus AI
Version: 1.0.0
"""

import pandas as pd
import json
import io
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import streamlit as st


class DataProcessor:
    """
    Manages data input, processing, and quality assessment for the
    Fraudulent Seller Detection Portal.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Data Processor with configuration settings.
        
        Args:
            config: Configuration dictionary for data processing.
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        self.chunk_size = self.config.get("chunk_size", 10000)  # Rows per chunk
        self.demo_datasets_path = self.config.get("demo_datasets_path", "data/samples")
        self._ensure_demo_datasets_directory()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the Data Processor."""
        logger = logging.getLogger("data_processor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("logs/data_processor.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _ensure_demo_datasets_directory(self):
        """Ensure the directory for demo datasets exists."""
        os.makedirs(self.demo_datasets_path, exist_ok=True)
        
    def process_uploaded_file(self, uploaded_file, file_metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Processes an uploaded file, handling different formats and large datasets.
        
        Args:
            uploaded_file: The file object uploaded via Streamlit.
            file_metadata: Metadata about the file from SecurityManager validation.
            
        Returns:
            A pandas DataFrame containing the processed data.
        """
        mime_type = file_metadata.get("mime_type")
        
        self.logger.info(f"Starting processing for file: {uploaded_file.name} (MIME: {mime_type})")
        
        try:
            if mime_type == "text/csv":
                df = self._read_csv_in_chunks(uploaded_file)
            elif mime_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                df = self._read_excel_in_chunks(uploaded_file)
            elif mime_type == "application/json":
                df = self._read_json_in_chunks(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type for processing: {mime_type}")
            
            self.logger.info(f"Successfully processed file {uploaded_file.name}. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing file {uploaded_file.name}: {e}")
            raise DataProcessingError(f"Failed to process file: {str(e)}")
    
    def _read_csv_in_chunks(self, uploaded_file) -> pd.DataFrame:
        """
        Reads a CSV file in chunks to handle large datasets efficiently.
        """
        chunks = []
        for chunk in pd.read_csv(uploaded_file, chunksize=self.chunk_size):
            chunks.append(chunk)
            # Optional: Update progress bar here if integrated with Streamlit UI
        uploaded_file.seek(0) # Reset pointer for potential re-reads
        return pd.concat(chunks, ignore_index=True)
    
    def _read_excel_in_chunks(self, uploaded_file) -> pd.DataFrame:
        """
        Reads an Excel file in chunks. Note: pandas read_excel doesn't directly
        support chunking like CSV, so we read the whole file into memory first
        if it's within limits, or suggest alternative for very large excel files.
        """
        # For very large Excel files, a different strategy (e.g., openpyxl for row-by-row) might be needed.
        # For now, assuming files within reasonable memory limits after initial size validation.
        df = pd.read_excel(uploaded_file)
        uploaded_file.seek(0)
        return df
    
    def _read_json_in_chunks(self, uploaded_file) -> pd.DataFrame:
        """
        Reads a JSON file in chunks. Similar to Excel, direct chunking isn't
        straightforward with pandas for JSON. Reads into memory.
        """
        content = uploaded_file.read().decode("utf-8")
        data = json.loads(content)
        uploaded_file.seek(0)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data]) # Wrap single dict in a list for DataFrame
        else:
            raise ValueError("JSON content is not a list of objects or a single object.")
            
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assesses the quality of the input DataFrame.
        
        Args:
            df: The DataFrame to assess.
            
        Returns:
            A dictionary containing data quality metrics.
        """
        self.logger.info("Assessing data quality...")
        quality_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_summary": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "unique_values_count": {col: df[col].nunique() for col in df.columns},
            "numeric_column_stats": {}
        }
        
        for col in df.select_dtypes(include=np.number).columns:
            quality_report["numeric_column_stats"][col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max()
            }
            
        self.logger.info("Data quality assessment complete.")
        return quality_report
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean", columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handles missing values in the DataFrame using specified strategy.
        
        Args:
            df: The DataFrame to process.
            strategy: 'mean', 'median', 'mode', 'drop', or 'fill_value'.
            columns: List of columns to apply the strategy to. If None, applies to all.
            
        Returns:
            DataFrame with missing values handled.
        """
        df_processed = df.copy()
        target_columns = columns if columns is not None else df_processed.columns
        
        self.logger.info(f"Handling missing values with strategy: {strategy} for columns: {target_columns}")
        
        for col in target_columns:
            if df_processed[col].isnull().any():
                if strategy == "mean":
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                elif strategy == "median":
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        df_processed[col].fillna(df_processed[col].median(), inplace=True)
                elif strategy == "mode":
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                elif strategy == "drop":
                    df_processed.dropna(subset=[col], inplace=True)
                elif strategy == "fill_value":
                    # Requires a value to be passed, not directly supported by this signature
                    # This would typically be handled by a separate parameter or a more complex strategy object
                    self.logger.warning(f"'fill_value' strategy requires a specific value for column {col}. Skipping.")
                else:
                    self.logger.warning(f"Unknown missing value strategy: {strategy}. Skipping column {col}.")
        
        if strategy == "drop_all_rows":
            df_processed.dropna(inplace=True)
            
        self.logger.info("Missing value handling complete.")
        return df_processed
    
    def generate_synthetic_data(self, num_rows: int, schema: Dict[str, str]) -> pd.DataFrame:
        """
        Generates synthetic data based on a provided schema.
        
        Args:
            num_rows: Number of rows to generate.
            schema: Dictionary where keys are column names and values are data types
                    (e.g., {"transaction_id": "int", "amount": "float", "is_fraud": "bool"}).
                    Supports 'int', 'float', 'str', 'bool', 'datetime'.
                    
        Returns:
            A pandas DataFrame with synthetic data.
        """
        self.logger.info(f"Generating {num_rows} rows of synthetic data with schema: {schema}")
        data = {}
        for col, dtype in schema.items():
            if dtype == "int":
                data[col] = np.random.randint(1, 100000, num_rows)
            elif dtype == "float":
                data[col] = np.random.rand(num_rows) * 1000
            elif dtype == "str":
                data[col] = [f"item_{i}" for i in range(num_rows)]
            elif dtype == "bool":
                data[col] = np.random.choice([True, False], num_rows)
            elif dtype == "datetime":
                start_date = datetime(2023, 1, 1)
                data[col] = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_rows)]
            else:
                data[col] = [None] * num_rows # Placeholder for unsupported types
                self.logger.warning(f"Unsupported data type for synthetic data generation: {dtype}")
                
        df = pd.DataFrame(data)
        self.logger.info("Synthetic data generation complete.")
        return df
    
    def load_demo_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Loads a pre-configured demo dataset.
        
        Args:
            dataset_name: Name of the demo dataset (e.g., "sample_transactions.csv").
            
        Returns:
            A pandas DataFrame if the dataset is found, otherwise None.
        """
        dataset_path = os.path.join(self.demo_datasets_path, dataset_name)
        self.logger.info(f"Attempting to load demo dataset from: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            self.logger.warning(f"Demo dataset not found: {dataset_path}")
            return None
            
        try:
            if dataset_name.endswith(".csv"):
                df = pd.read_csv(dataset_path)
            elif dataset_name.endswith(tuple([".xls", ".xlsx"])): # For Excel files
                df = pd.read_excel(dataset_path)
            elif dataset_name.endswith(".json"):
                df = pd.read_json(dataset_path)
            else:
                self.logger.warning(f"Unsupported demo dataset format: {dataset_name}")
                return None
                
            self.logger.info(f"Successfully loaded demo dataset {dataset_name}. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading demo dataset {dataset_name}: {e}")
            raise DataProcessingError(f"Failed to load demo dataset: {str(e)}")
            
    def get_available_demo_datasets(self) -> List[str]:
        """
        Lists all available demo datasets.
        
        Returns:
            A list of filenames of available demo datasets.
        """
        try:
            return [f for f in os.listdir(self.demo_datasets_path) if os.path.isfile(os.path.join(self.demo_datasets_path, f))]
        except Exception as e:
            self.logger.error(f"Error listing demo datasets: {e}")
            return []


class DataProcessingError(Exception):
    """Custom exception for data processing related errors."""
    pass


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Create a dummy CSV file for testing
    dummy_csv_content = "col1,col2,col3\n1,a,True\n2,b,False\n3,,True\n4,d,False"
    with open("test_data.csv", "w") as f:
        f.write(dummy_csv_content)
        
    # Simulate a Streamlit uploaded file object
    class MockUploadedFile:
        def __init__(self, name, content, size, mime_type):
            self.name = name
            self._content = content.encode("utf-8")
            self.size = size
            self.mime_type = mime_type
            self._io = io.BytesIO(self._content)
            
        def read(self, size=-1):
            return self._io.read(size)
            
        def seek(self, offset):
            self._io.seek(offset)
            
    mock_file = MockUploadedFile(
        name="test_data.csv",
        content=dummy_csv_content,
        size=len(dummy_csv_content),
        mime_type="text/csv"
    )
    
    # Initialize DataProcessor
    data_processor = DataProcessor()
    
    # Simulate file metadata from SecurityManager
    file_meta = {
        "mime_type": "text/csv",
        "name": "test_data.csv",
        "size": len(dummy_csv_content),
        "columns": ["col1", "col2", "col3"],
        "row_count": 4
    }
    
    # Test file processing
    processed_df = data_processor.process_uploaded_file(mock_file, file_meta)
    print("\nProcessed DataFrame:")
    print(processed_df)
    
    # Test data quality assessment
    quality_report = data_processor.assess_data_quality(processed_df)
    print("\nData Quality Report:")
    print(json.dumps(quality_report, indent=2))
    
    # Test handling missing values
    df_missing_handled = data_processor.handle_missing_values(processed_df, strategy="mode", columns=["col2"])
    print("\nDataFrame after handling missing values (mode for col2):")
    print(df_missing_handled)
    
    # Test synthetic data generation
    synthetic_schema = {
        "transaction_id": "int",
        "amount": "float",
        "description": "str",
        "is_fraud": "bool",
        "timestamp": "datetime"
    }
    synthetic_df = data_processor.generate_synthetic_data(num_rows=10, schema=synthetic_schema)
    print("\nSynthetic DataFrame:")
    print(synthetic_df.head())
    
    # Clean up dummy file
    os.remove("test_data.csv")
    
    # Create a dummy demo dataset
    dummy_demo_csv_content = "id,value\n1,100\n2,200"
    os.makedirs("data/samples", exist_ok=True)
    with open("data/samples/demo_data.csv", "w") as f:
        f.write(dummy_demo_csv_content)
        
    # Test loading demo dataset
    demo_df = data_processor.load_demo_dataset("demo_data.csv")
    print("\nLoaded Demo DataFrame:")
    print(demo_df)
    
    # Test listing demo datasets
    available_demos = data_processor.get_available_demo_datasets()
    print("\nAvailable Demo Datasets:", available_demos)
    
    # Clean up dummy demo dataset
    os.remove("data/samples/demo_data.csv")
    os.rmdir("data/samples")
    os.rmdir("data")
    
    # Clean up logs directory if created
    if os.path.exists("logs/data_processor.log"):
        os.remove("logs/data_processor.log")
    if os.path.exists("logs"):
        os.rmdir("logs")




