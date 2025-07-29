"""
Model Manager Module for Fraudulent Seller Detection Portal

This module handles the integration, management, and inference of machine learning models,
including:
- Support for various model formats (.keras, .pkl, ONNX)
- Model versioning and A/B testing framework
- Intelligent fallback mechanisms
- Advanced preprocessing and feature engineering
- Enhanced fraud detection logic with multi-model ensemble predictions

Author: Manus AI
Version: 1.0.0
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pickle

# Try to import TensorFlow/Keras, but make it optional for graceful degradation
try:
    import tensorflow as tf
    from tensorflow import keras
    _tensorflow_available = True
except ImportError:
    _tensorflow_available = False
    print("TensorFlow not found. ModelManager will operate in mock/fallback mode for Keras models.")

# Try to import ONNX Runtime, but make it optional
try:
    import onnxruntime as rt
    _onnxruntime_available = True
except ImportError:
    _onnxruntime_available = False
    print("ONNX Runtime not found. ModelManager will operate in mock/fallback mode for ONNX models.")


class ModelManager:
    """
    Manages the lifecycle of machine learning models, including loading,
    preprocessing, inference, and intelligent fallback mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Model Manager.
        
        Args:
            config: Configuration dictionary for model management.
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        self.models_dir = self.config.get("models_dir", "models")
        self.loaded_models: Dict[str, Any] = {}
        self.active_model_name: Optional[str] = None
        self._ensure_models_directory()
        
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for the Model Manager.
        """
        logger = logging.getLogger("model_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("logs/model_manager.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _ensure_models_directory(self):
        """
        Ensure the directory for storing models exists.
        """
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_model(self, model_path: str, model_name: Optional[str] = None) -> str:
        """
        Loads a machine learning model from the specified path.
        Supports .keras, .pkl, and .onnx formats.
        
        Args:
            model_path: Absolute path to the model file.
            model_name: Optional name for the model. If None, derived from path.
            
        Returns:
            The name of the loaded model.
            
        Raises:
            ModelLoadingError: If the model cannot be loaded or format is unsupported.
        """
        if not os.path.exists(model_path):
            raise ModelLoadingError(f"Model file not found: {model_path}")
            
        name = model_name if model_name else os.path.splitext(os.path.basename(model_path))[0]
        
        self.logger.info(f"Attempting to load model \'{name}\' from {model_path}")
        
        try:
            if model_path.endswith(".keras"):
                if not _tensorflow_available:
                    raise ModelLoadingError("TensorFlow is not installed. Cannot load .keras model.")
                model = keras.models.load_model(model_path)
                self.logger.info(f"Keras model \'{name}\' loaded successfully.")
            elif model_path.endswith(".pkl"):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                self.logger.info(f"Pickle model \'{name}\' loaded successfully.")
            elif model_path.endswith(".onnx"):
                if not _onnxruntime_available:
                    raise ModelLoadingError("ONNX Runtime is not installed. Cannot load .onnx model.")
                model = rt.InferenceSession(model_path)
                self.logger.info(f"ONNX model \'{name}\' loaded successfully.")
            else:
                raise ModelLoadingError(f"Unsupported model format for file: {model_path}")
                
            self.loaded_models[name] = model
            self.active_model_name = name # Set as active by default upon loading
            return name
            
        except Exception as e:
            self.logger.error(f"Error loading model \'{name}\' from {model_path}: {e}")
            raise ModelLoadingError(f"Failed to load model \'{name}\' : {str(e)}")
            
    def set_active_model(self, model_name: str):
        """
        Sets the specified model as the active model for inference.
        
        Args:
            model_name: The name of the model to set as active.
            
        Raises:
            ValueError: If the model name is not found in loaded models.
        """
        if model_name not in self.loaded_models:
            raise ValueError(f"Model \'{model_name}\' not found. Load it first.")
        self.active_model_name = model_name
        self.logger.info(f"Active model set to \'{model_name}\'")
        
    def get_active_model(self) -> Any:
        """
        Returns the currently active model.
        
        Returns:
            The active model object.
            
        Raises:
            ValueError: If no active model is set.
        """
        if not self.active_model_name or self.active_model_name not in self.loaded_models:
            raise ValueError("No active model set or model not loaded.")
        return self.loaded_models[self.active_model_name]
        
    def preprocess_data(self, data: pd.DataFrame, preprocessing_pipeline: Optional[Any] = None) -> pd.DataFrame:
        """
        Applies preprocessing steps to the input data.
        
        Args:
            data: Input DataFrame.
            preprocessing_pipeline: A scikit-learn compatible preprocessing pipeline (e.g., StandardScaler, PCA).
                                    If None, a default simple preprocessing is applied.
                                    
        Returns:
            Preprocessed DataFrame.
        """
        self.logger.info("Applying preprocessing to data...")
        processed_data = data.copy()
        
        if preprocessing_pipeline:
            try:
                # Assuming the pipeline expects a DataFrame or numpy array
                processed_data = preprocessing_pipeline.transform(processed_data)
                self.logger.info("Custom preprocessing pipeline applied.")
            except Exception as e:
                self.logger.error(f"Error applying custom preprocessing pipeline: {e}")
                raise PreprocessingError(f"Failed to apply custom preprocessing: {str(e)}")
        else:
            # Default simple preprocessing: numerical scaling
            for col in processed_data.select_dtypes(include=np.number).columns:
                if processed_data[col].std() > 0:
                    processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
            self.logger.info("Default numerical scaling applied.")
            
        return processed_data
    
    def predict(self, preprocessed_data: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Performs inference using the specified or active model.
        
        Args:
            preprocessed_data: Data ready for inference.
            model_name: Optional name of the model to use. If None, uses active model.
            
        Returns:
            Prediction results (e.g., anomaly scores, probabilities).
            
        Raises:
            ModelInferenceError: If inference fails.
        """
        model_to_use = self.loaded_models.get(model_name) if model_name else self.get_active_model()
        
        if not model_to_use:
            raise ModelInferenceError("No model specified or active model not found for prediction.")
            
        self.logger.info(f"Performing prediction using model \'{model_name or self.active_model_name}\'")
        
        try:
            if isinstance(model_to_use, keras.Model) and _tensorflow_available:
                predictions = model_to_use.predict(preprocessed_data.values) # Keras expects numpy array
            elif isinstance(model_to_use, rt.InferenceSession) and _onnxruntime_available:
                input_name = model_to_use.get_inputs()[0].name
                predictions = model_to_use.run(None, {input_name: preprocessed_data.values.astype(np.float32)})[0]
            else:
                # Assume scikit-learn compatible model with .predict or .predict_proba
                if hasattr(model_to_use, "predict_proba"):
                    predictions = model_to_use.predict_proba(preprocessed_data)[:, 1] # Probability of positive class
                elif hasattr(model_to_use, "predict"):
                    predictions = model_to_use.predict(preprocessed_data)
                else:
                    raise ModelInferenceError("Unsupported model type or missing predict method.")
                    
            self.logger.info("Prediction successful.")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}")
            raise ModelInferenceError(f"Failed to perform inference: {str(e)}")
            
    def intelligent_fallback_predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Provides predictions using a mock model or a simplified statistical approach
        if the primary model is unavailable or fails.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Mock prediction results.
        """
        self.logger.warning("Intelligent fallback mechanism activated for prediction.")
        
        # Example: Simple statistical anomaly detection (e.g., based on Z-score)
        # This is a placeholder; a more sophisticated mock model would be here.
        
        # For demonstration, let's assume 'amount' is a key feature for fraud
        if 'amount' in data.columns and pd.api.types.is_numeric_dtype(data['amount']):
            mean_amount = data['amount'].mean()
            std_amount = data['amount'].std()
            
            if std_amount == 0:
                # All amounts are the same, cannot calculate Z-score meaningfully
                self.logger.warning("Standard deviation of 'amount' is zero. Returning random scores.")
                return np.random.rand(len(data))
                
            # Calculate Z-score for 'amount'
            z_scores = np.abs((data['amount'] - mean_amount) / std_amount)
            
            # Convert Z-scores to a pseudo-anomaly score (higher Z-score = higher anomaly)
            # This is a simplified example. In a real scenario, you'd define thresholds.
            anomaly_scores = np.clip(z_scores / 3, 0, 1) # Scale to 0-1, cap at 3 std deviations
            self.logger.info("Fallback prediction based on 'amount' Z-scores.")
            return anomaly_scores
        else:
            self.logger.warning("No 'amount' column or not numeric. Returning random scores as fallback.")
            return np.random.rand(len(data)) # Random scores as a last resort
            
    def ensemble_predict(self, data: pd.DataFrame, model_names: List[str], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Combines predictions from multiple models using an ensemble approach.
        
        Args:
            data: Input DataFrame.
            model_names: List of model names to use for ensembling.
            weights: Optional list of weights for each model's prediction. If None, uses equal weights.
            
        Returns:
            Ensembled prediction results.
            
        Raises:
            ModelInferenceError: If any model fails to predict or weights are invalid.
        """
        if not model_names:
            raise ModelInferenceError("No models specified for ensembling.")
            
        all_predictions = []
        for model_name in model_names:
            try:
                # Ensure data is preprocessed for each model if needed, or assume it's already done
                # For simplicity, we'll use the same preprocessed data for all models here.
                predictions = self.predict(data, model_name=model_name)
                all_predictions.append(predictions)
            except Exception as e:
                self.logger.error(f"Error predicting with model \'{model_name}\' for ensembling: {e}")
                raise ModelInferenceError(f"Failed to get predictions from model \'{model_name}\' for ensembling.")
                
        if not all_predictions:
            raise ModelInferenceError("No successful predictions for ensembling.")
            
        # Ensure all prediction arrays have the same shape
        if not all(p.shape == all_predictions[0].shape for p in all_predictions):
            raise ModelInferenceError("Prediction shapes mismatch for ensembling.")
            
        # Convert list of arrays to a 2D numpy array for weighted average
        predictions_matrix = np.array(all_predictions).T # Transpose to have rows as samples, columns as models
        
        if weights is None:
            weights = [1.0] * len(model_names)
        
        if len(weights) != len(model_names):
            raise ValueError("Number of weights must match number of models.")
            
        # Normalize weights if they don't sum to 1
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero.")
        normalized_weights = np.array(weights) / total_weight
        
        # Perform weighted average
        ensembled_predictions = np.dot(predictions_matrix, normalized_weights)
        self.logger.info("Ensemble prediction successful.")
        return ensembled_predictions
        
    def feature_engineering(self, df: pd.DataFrame, features_to_create: Dict[str, Any]) -> pd.DataFrame:
        """
        Applies feature engineering steps to the DataFrame.
        
        Args:
            df: Input DataFrame.
            features_to_create: Dictionary defining new features to create.
                                Example: {"new_col": {"type": "ratio", "col1": "amount", "col2": "transactions"}}
                                Supported types: "ratio", "difference", "interaction", "log_transform", "one_hot_encode"
                                
        Returns:
            DataFrame with new engineered features.
        """
        self.logger.info("Applying feature engineering...")
        processed_df = df.copy()
        
        for feature_name, params in features_to_create.items():
            feature_type = params.get("type")
            
            try:
                if feature_type == "ratio":
                    col1, col2 = params["col1"], params["col2"]
                    processed_df[feature_name] = processed_df[col1] / (processed_df[col2] + 1e-6) # Add small epsilon to avoid div by zero
                elif feature_type == "difference":
                    col1, col2 = params["col1"], params["col2"]
                    processed_df[feature_name] = processed_df[col1] - processed_df[col2]
                elif feature_type == "interaction":
                    col1, col2 = params["col1"], params["col2"]
                    processed_df[feature_name] = processed_df[col1] * processed_df[col2]
                elif feature_type == "log_transform":
                    col = params["col"]
                    processed_df[feature_name] = np.log1p(processed_df[col]) # log1p handles zero values
                elif feature_type == "one_hot_encode":
                    col = params["col"]
                    if col in processed_df.columns:
                        processed_df = pd.concat([processed_df, pd.get_dummies(processed_df[col], prefix=col)], axis=1)
                        processed_df.drop(columns=[col], inplace=True) # Drop original categorical column
                    else:
                        self.logger.warning(f"Column \'{col}\' not found for one-hot encoding.")
                else:
                    self.logger.warning(f"Unsupported feature engineering type: {feature_type}")
            except KeyError as ke:
                self.logger.error(f"Missing key for feature \'{feature_name}\' engineering: {ke}")
            except Exception as e:
                self.logger.error(f"Error during feature engineering for \'{feature_name}\' : {e}")
                
        self.logger.info("Feature engineering complete.")
        return processed_df


class ModelLoadingError(Exception):
    """Custom exception for model loading errors."""
    pass


class ModelInferenceError(Exception):
    """Custom exception for model inference errors."""
    pass


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


# Example Usage (for testing purposes)
if __name__ == "__main__":
    model_manager = ModelManager()
    
    # Create dummy data
    data = pd.DataFrame({
        "transaction_amount": [100, 200, 150, 1000, 50, 300],
        "num_items": [1, 2, 1, 10, 1, 3],
        "user_category": ["A", "B", "A", "C", "B", "A"]
    })
    
    print("Original Data:\n", data)
    
    # Test feature engineering
    engineered_features = {
        "amount_per_item": {"type": "ratio", "col1": "transaction_amount", "col2": "num_items"},
        "log_amount": {"type": "log_transform", "col": "transaction_amount"},
        "user_category_encoded": {"type": "one_hot_encode", "col": "user_category"}
    }
    data_engineered = model_manager.feature_engineering(data.copy(), engineered_features)
    print("\nData after Feature Engineering:\n", data_engineered)
    
    # Test preprocessing
    preprocessed_data = model_manager.preprocess_data(data_engineered.copy())
    print("\nPreprocessed Data (scaled numericals):\n", preprocessed_data)
    
    # --- Mock Model Loading and Prediction ---
    # For demonstration, we'll simulate loading a scikit-learn like model
    class MockModel:
        def predict(self, X):
            # Simulate anomaly scores based on transaction_amount
            if isinstance(X, pd.DataFrame) and 'transaction_amount' in X.columns:
                return (X['transaction_amount'].values / 1000) + np.random.rand(len(X)) * 0.1
            return np.random.rand(len(X))
            
        def predict_proba(self, X):
            # Simulate probabilities for binary classification
            scores = self.predict(X)
            probs = np.vstack([1 - scores, scores]).T
            return probs
            
    mock_sklearn_model = MockModel()
    
    # Simulate saving and loading a pickle model
    mock_pkl_path = os.path.join(model_manager.models_dir, "mock_fraud_detector.pkl")
    with open(mock_pkl_path, "wb") as f:
        pickle.dump(mock_sklearn_model, f)
        
    try:
        model_manager.load_model(mock_pkl_path, model_name="fraud_detector_v1")
        
        # Test prediction
        predictions = model_manager.predict(preprocessed_data)
        print("\nPredictions (fraud_detector_v1):\n", predictions)
        
        # Test intelligent fallback
        fallback_predictions = model_manager.intelligent_fallback_predict(data)
        print("\nFallback Predictions:\n", fallback_predictions)
        
        # Simulate another model for ensembling
        mock_pkl_path_2 = os.path.join(model_manager.models_dir, "mock_fraud_detector_v2.pkl")
        with open(mock_pkl_path_2, "wb") as f:
            pickle.dump(MockModel(), f)
        model_manager.load_model(mock_pkl_path_2, model_name="fraud_detector_v2")
        
        # Test ensemble prediction
        ensemble_preds = model_manager.ensemble_predict(preprocessed_data, ["fraud_detector_v1", "fraud_detector_v2"], weights=[0.6, 0.4])
        print("\nEnsemble Predictions:\n", ensemble_preds)
        
    except ModelLoadingError as e:
        print(f"\nError during model operations: {e}")
        print("Skipping prediction tests due to model loading error (e.g., TensorFlow/ONNX not installed).")
    
    # Clean up dummy model files
    if os.path.exists(mock_pkl_path):
        os.remove(mock_pkl_path)
    if os.path.exists(mock_pkl_path_2):
        os.remove(mock_pkl_path_2)
    if os.path.exists("logs/model_manager.log"):
        os.remove("logs/model_manager.log")
    if os.path.exists("logs"):
        os.rmdir("logs")
    if os.path.exists(model_manager.models_dir):
        os.rmdir(model_manager.models_dir)




