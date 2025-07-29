"""
UI Components Module for Fraudulent Seller Detection Portal

This module provides Streamlit UI components for interactive controls and settings,
including:
- Anomaly detection configuration (thresholds, statistical calculation)
- Model configuration panel (hyperparameter adjustment, feature selection)
- Advanced settings (processing optimization, data retention, notifications)

Author: Manus AI
Version: 1.0.0
"""

import streamlit as st
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


class UIComponents:
    """
    A collection of Streamlit UI components for configuring various aspects
    of the Fraudulent Seller Detection Portal.
    """
    
    def __init__(self):
        pass
        
    def anomaly_detection_config(self, current_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Renders UI for anomaly detection configuration.
        
        Args:
            current_thresholds: Dictionary of current threshold values (e.g., {"warning": 0.5, "alert": 0.7, "critical": 0.9}).
            
        Returns:
            Updated dictionary of threshold values.
        """
        st.header("Anomaly Detection Configuration")
        st.write("Define thresholds for different anomaly levels.")
        
        new_thresholds = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_thresholds["warning"] = st.slider(
                "Warning Threshold",
                min_value=0.0, max_value=1.0, value=current_thresholds.get("warning", 0.5),
                step=0.01,
                help="Anomaly scores above this value will be flagged as warning."
            )
        with col2:
            new_thresholds["alert"] = st.slider(
                "Alert Threshold",
                min_value=0.0, max_value=1.0, value=current_thresholds.get("alert", 0.7),
                step=0.01,
                help="Anomaly scores above this value will be flagged as alert."
            )
        with col3:
            new_thresholds["critical"] = st.slider(
                "Critical Threshold",
                min_value=0.0, max_value=1.0, value=current_thresholds.get("critical", 0.9),
                step=0.01,
                help="Anomaly scores above this value will be flagged as critical."
            )
            
        if not (new_thresholds["warning"] <= new_thresholds["alert"] <= new_thresholds["critical"]):
            st.warning("Thresholds must be in increasing order (Warning <= Alert <= Critical).")
            
        st.subheader("Statistical Threshold Calculation")
        stat_method = st.selectbox(
            "Method for Statistical Thresholds",
            ["None", "Percentile-based", "Standard Deviation-based"],
            help="Choose a method to automatically calculate thresholds based on data distribution."
        )
        
        if stat_method == "Percentile-based":
            p_warning = st.slider("Warning Percentile", 0, 100, 90, help="e.g., 90th percentile for warning.")
            p_alert = st.slider("Alert Percentile", 0, 100, 95, help="e.g., 95th percentile for alert.")
            p_critical = st.slider("Critical Percentile", 0, 100, 99, help="e.g., 99th percentile for critical.")
            st.info(f"Thresholds will be calculated based on {p_warning}th, {p_alert}th, and {p_critical}th percentiles of anomaly scores.")
            # In a real app, you'd pass these percentiles to a backend function to calculate actual values
            
        elif stat_method == "Standard Deviation-based":
            std_warning = st.number_input("Warning Std Dev Multiplier", value=1.5, step=0.1, help="e.g., 1.5 standard deviations from mean.")
            std_alert = st.number_input("Alert Std Dev Multiplier", value=2.0, step=0.1, help="e.g., 2.0 standard deviations from mean.")
            std_critical = st.number_input("Critical Std Dev Multiplier", value=3.0, step=0.1, help="e.g., 3.0 standard deviations from mean.")
            st.info(f"Thresholds will be calculated based on {std_warning}, {std_alert}, and {std_critical} standard deviations from the mean anomaly score.")
            # Similar to percentiles, these would be used by a backend function
            
        return new_thresholds
        
    def model_configuration_panel(self, available_models: List[str], current_model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Renders UI for model configuration, including selection, hyperparameters, and feature selection.
        
        Args:
            available_models: List of names of models that can be selected.
            current_model_config: Dictionary of current model configuration.
            
        Returns:
            Updated dictionary of model configuration.
        """
        st.header("Model Configuration Panel")
        
        new_config = current_model_config.copy()
        
        # Model Selection
        selected_model = st.selectbox(
            "Select Active Model",
            options=available_models,
            index=available_models.index(current_model_config.get("active_model", available_models[0])) if available_models else 0,
            help="Choose the machine learning model to use for fraud detection."
        )
        new_config["active_model"] = selected_model
        
        st.subheader("Hyperparameter Adjustment")
        st.info("Note: Hyperparameters shown here are examples. Actual parameters depend on the selected model.")
        
        # Example hyperparameters (these would be dynamically loaded based on `selected_model` in a real app)
        if selected_model == "fraud_detector_v1":
            new_config["learning_rate"] = st.number_input(
                "Learning Rate",
                value=current_model_config.get("learning_rate", 0.01),
                min_value=0.0001, max_value=0.1, step=0.001,
                format="%.4f"
            )
            new_config["n_estimators"] = st.slider(
                "Number of Estimators",
                value=current_model_config.get("n_estimators", 100),
                min_value=10, max_value=500, step=10
            )
        elif selected_model == "mock_model":
            new_config["random_seed"] = st.number_input(
                "Random Seed",
                value=current_model_config.get("random_seed", 42),
                min_value=0, step=1
            )
            
        st.subheader("Feature Selection and Weighting")
        all_features = ["transaction_amount", "num_items", "user_category", "time_of_day"]
        selected_features = st.multiselect(
            "Select Features for Model Input",
            options=all_features,
            default=current_model_config.get("selected_features", all_features),
            help="Choose which features to feed into the model."
        )
        new_config["selected_features"] = selected_features
        
        if selected_features:
            st.write("Adjust weights for selected features (sum does not need to be 1):")
            feature_weights = {}
            for feature in selected_features:
                feature_weights[feature] = st.number_input(
                    f"Weight for {feature}",
                    value=current_model_config.get("feature_weights", {}).get(feature, 1.0),
                    min_value=0.0, step=0.1
                )
            new_config["feature_weights"] = feature_weights
            
        st.subheader("Prediction Confidence Calibration")
        new_config["confidence_calibration_enabled"] = st.checkbox(
            "Enable Confidence Calibration",
            value=current_model_config.get("confidence_calibration_enabled", False),
            help="Adjust model outputs to better reflect true probabilities."
        )
        
        return new_config
        
    def advanced_settings(self, current_advanced_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Renders UI for advanced application settings.
        
        Args:
            current_advanced_settings: Dictionary of current advanced settings.
            
        Returns:
            Updated dictionary of advanced settings.
        """
        st.header("Advanced Settings")
        
        new_settings = current_advanced_settings.copy()
        
        st.subheader("Processing Performance Optimization")
        new_settings["parallel_processing_enabled"] = st.checkbox(
            "Enable Parallel Processing",
            value=current_advanced_settings.get("parallel_processing_enabled", True),
            help="Utilize multiple CPU cores for faster data processing."
        )
        new_settings["batch_size"] = st.number_input(
            "Batch Size for Processing",
            value=current_advanced_settings.get("batch_size", 1000),
            min_value=100, max_value=10000, step=100,
            help="Number of records to process in each batch."
        )
        
        st.subheader("Data Retention and Privacy Settings")
        data_retention_days = st.slider(
            "Data Retention Period (days)",
            min_value=7, max_value=365, value=current_advanced_settings.get("data_retention_days", 30),
            help="Number of days to retain processed data and logs."
        )
        new_settings["data_retention_days"] = data_retention_days
        
        new_settings["anonymize_data_on_retention"] = st.checkbox(
            "Anonymize Data on Retention",
            value=current_advanced_settings.get("anonymize_data_on_retention", True),
            help="Anonymize sensitive data after the retention period instead of full deletion."
        )
        
        st.subheader("Notification and Alerting Configurations")
        new_settings["email_notifications_enabled"] = st.checkbox(
            "Enable Email Notifications",
            value=current_advanced_settings.get("email_notifications_enabled", False),
            help="Receive email alerts for critical events."
        )
        if new_settings["email_notifications_enabled"]:
            new_settings["notification_email"] = st.text_input(
                "Notification Email Address",
                value=current_advanced_settings.get("notification_email", ""),
                help="Enter the email address to send notifications to."
            )
            
        new_settings["slack_notifications_enabled"] = st.checkbox(
            "Enable Slack Notifications",
            value=current_advanced_settings.get("slack_notifications_enabled", False),
            help="Receive Slack alerts for critical events."
        )
        
        st.subheader("Custom Risk Level Definitions")
        st.info("Define custom labels and colors for risk levels.")
        
        num_custom_levels = st.number_input(
            "Number of Custom Risk Levels",
            min_value=0, max_value=5, value=len(current_advanced_settings.get("custom_risk_levels", [])),
            step=1
        )
        
        custom_risk_levels = []
        for i in range(num_custom_levels):
            st.write(f"--- Risk Level {i+1} ---")
            level_name = st.text_input(f"Name for Level {i+1}", value=current_advanced_settings.get("custom_risk_levels", [{}])[-1].get("name", f"Level {i+1}") if i < len(current_advanced_settings.get("custom_risk_levels", [])) else f"Level {i+1}", key=f"level_name_{i}")
            level_color = st.color_picker(f"Color for Level {i+1}", value=current_advanced_settings.get("custom_risk_levels", [{}])[-1].get("color", "#FF0000") if i < len(current_advanced_settings.get("custom_risk_levels", [])) else "#FF0000", key=f"level_color_{i}")
            level_threshold = st.slider(f"Threshold for Level {i+1}", min_value=0.0, max_value=1.0, value=current_advanced_settings.get("custom_risk_levels", [{}])[-1].get("threshold", 0.5) if i < len(current_advanced_settings.get("custom_risk_levels", [])) else 0.5, step=0.01, key=f"level_threshold_{i}")
            custom_risk_levels.append({"name": level_name, "color": level_color, "threshold": level_threshold})
            
        new_settings["custom_risk_levels"] = custom_risk_levels
        
        return new_settings


# Example Usage (for testing purposes)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    ui_components = UIComponents()
    
    st.title("UI Components Demo")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Anomaly Config", "Model Config", "Advanced Settings"])
    
    if page == "Anomaly Config":
        current_anomaly_thresholds = st.session_state.get("anomaly_thresholds", {"warning": 0.5, "alert": 0.7, "critical": 0.9})
        updated_anomaly_thresholds = ui_components.anomaly_detection_config(current_anomaly_thresholds)
        st.session_state["anomaly_thresholds"] = updated_anomaly_thresholds
        st.write("\nUpdated Anomaly Thresholds:", updated_anomaly_thresholds)
        
    elif page == "Model Config":
        available_models = ["fraud_detector_v1", "mock_model", "ensemble_model"]
        current_model_config = st.session_state.get("model_config", {"active_model": "fraud_detector_v1"})
        updated_model_config = ui_components.model_configuration_panel(available_models, current_model_config)
        st.session_state["model_config"] = updated_model_config
        st.write("\nUpdated Model Configuration:", updated_model_config)
        
    elif page == "Advanced Settings":
        current_advanced_settings = st.session_state.get("advanced_settings", {})
        updated_advanced_settings = ui_components.advanced_settings(current_advanced_settings)
        st.session_state["advanced_settings"] = updated_advanced_settings
        st.write("\nUpdated Advanced Settings:", updated_advanced_settings)




