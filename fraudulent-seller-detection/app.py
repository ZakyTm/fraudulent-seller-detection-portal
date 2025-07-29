"""
Main Streamlit Application for Fraudulent Seller Detection Portal

This is the main entry point for the application, integrating all modules
and providing the complete user interface.

Author: Manus AI
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.config_manager import ConfigManager
from core.security_manager import SecurityManager, GDPRCompliance
from core.data_processor import DataProcessor
from core.model_manager import ModelManager
from core.performance_manager import PerformanceManager
from core.monitoring_service import MonitoringService
from ui.components import UIComponents
from ui.dashboard import DashboardUI
from utils.error_handler import IntelligentErrorHandler, UserExperienceManager


class FraudDetectionApp:
    """
    Main application class that orchestrates all components.
    """
    
    def __init__(self):
        """Initialize the application with all necessary components."""
        self.config_manager = ConfigManager()
        self.security_manager = SecurityManager(self.config_manager.all())
        self.gdpr_compliance = GDPRCompliance(self.security_manager)
        self.data_processor = DataProcessor(self.config_manager.all())
        self.model_manager = ModelManager(self.config_manager.all())
        self.performance_manager = PerformanceManager(self.config_manager.all())
        self.monitoring_service = MonitoringService(self.config_manager.all())
        self.error_handler = IntelligentErrorHandler(self.monitoring_service)
        self.ux_manager = UserExperienceManager()
        self.ui_components = UIComponents()
        
        # Initialize session state
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = self.security_manager.generate_session_token()
            st.session_state.session_start_time = datetime.now()
            
        if 'data' not in st.session_state:
            st.session_state.data = None
            
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
            
        if 'anomaly_thresholds' not in st.session_state:
            st.session_state.anomaly_thresholds = {"warning": 0.5, "alert": 0.7, "critical": 0.9}
            
        if 'model_config' not in st.session_state:
            st.session_state.model_config = {"active_model": "default_fraud_model"}
            
        if 'advanced_settings' not in st.session_state:
            st.session_state.advanced_settings = {}
            
    def run(self):
        """Main application entry point."""
        st.set_page_config(
            page_title="Fraudulent Seller Detection Portal",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        try:
            # Check session validity
            if not self.security_manager.validate_session_token(
                st.session_state.session_id, 
                st.session_state.session_start_time
            ):
                st.error("Session expired. Please refresh the page.")
                return
                
            # Main UI
            self._render_header()
            self._render_sidebar()
            self._render_main_content()
            
        except Exception as e:
            self.error_handler.handle_exception(e, "main_app", st.session_state.session_id)
            
    def _render_header(self):
        """Render the application header."""
        st.title("🔍 Fraudulent Seller Detection Portal")
        st.markdown("---")
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("System Status", "🟢 Online")
        with col2:
            resource_usage = self.performance_manager.get_system_resource_usage()
            st.metric("CPU Usage", f"{resource_usage['cpu_percent']:.1f}%")
        with col3:
            st.metric("Memory Usage", f"{resource_usage['memory_percent']:.1f}%")
        with col4:
            st.metric("Session", f"Active ({st.session_state.session_id[:8]}...)")
            
    def _render_sidebar(self):
        """Render the sidebar with controls and settings."""
        st.sidebar.title("Control Panel")
        
        # Data Input Section
        st.sidebar.header("📁 Data Input")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Data File",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload your transaction data for fraud analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Validate file
                is_valid, error_msg, file_metadata = self.security_manager.validate_file_upload(uploaded_file)
                
                if is_valid:
                    st.sidebar.success("File validated successfully!")
                    
                    # Process file
                    with st.sidebar.spinner("Processing file..."):
                        processed_data = self.data_processor.process_uploaded_file(uploaded_file, file_metadata)
                        st.session_state.data = processed_data
                        
                        # Log user activity
                        self.monitoring_service.log_user_activity(
                            st.session_state.session_id,
                            "file_upload",
                            {"filename": uploaded_file.name, "rows": len(processed_data)}
                        )
                        
                    st.sidebar.success(f"Loaded {len(processed_data)} rows!")
                else:
                    st.sidebar.error(f"File validation failed: {error_msg}")
                    
            except Exception as e:
                self.error_handler.handle_exception(e, "file_upload", st.session_state.session_id)
                
        # Demo datasets
        st.sidebar.subheader("📊 Demo Datasets")
        demo_datasets = self.data_processor.get_available_demo_datasets()
        
        if demo_datasets:
            selected_demo = st.sidebar.selectbox("Select Demo Dataset", ["None"] + demo_datasets)
            if selected_demo != "None" and st.sidebar.button("Load Demo Dataset"):
                try:
                    demo_data = self.data_processor.load_demo_dataset(selected_demo)
                    if demo_data is not None:
                        st.session_state.data = demo_data
                        st.sidebar.success(f"Loaded demo dataset: {selected_demo}")
                except Exception as e:
                    self.error_handler.handle_exception(e, "demo_load", st.session_state.session_id)
        else:
            st.sidebar.info("No demo datasets available")
            
        # Model Configuration
        st.sidebar.header("🤖 Model Configuration")
        available_models = ["default_fraud_model", "ensemble_model", "mock_model"]
        
        updated_model_config = self.ui_components.model_configuration_panel(
            available_models, 
            st.session_state.model_config
        )
        st.session_state.model_config = updated_model_config
        
        # Advanced Settings
        st.sidebar.header("⚙️ Advanced Settings")
        updated_advanced_settings = self.ui_components.advanced_settings(
            st.session_state.advanced_settings
        )
        st.session_state.advanced_settings = updated_advanced_settings
        
    def _render_main_content(self):
        """Render the main content area."""
        if st.session_state.data is None:
            self._render_welcome_screen()
        else:
            self._render_analysis_interface()
            
    def _render_welcome_screen(self):
        """Render the welcome screen when no data is loaded."""
        st.header("Welcome to the Fraudulent Seller Detection Portal")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Getting Started
            
            This portal helps you detect fraudulent sellers and suspicious transactions using advanced machine learning techniques.
            
            **To begin:**
            1. Upload your transaction data using the file uploader in the sidebar
            2. Or select a demo dataset to explore the features
            3. Configure your fraud detection model settings
            4. Analyze the results in the interactive dashboard
            
            **Supported file formats:**
            - CSV files (.csv)
            - Excel files (.xlsx, .xls)
            - JSON files (.json)
            
            **Key Features:**
            - Real-time anomaly detection
            - Interactive visualizations
            - Comprehensive reporting
            - Advanced security and privacy controls
            """)
            
        with col2:
            st.info("💡 **Tip**: Start with a demo dataset to explore the features before uploading your own data.")
            
            if st.button("Show Interactive Tutorial"):
                self.ux_manager.display_interactive_tutorial("Getting Started")
                
            self.ux_manager.display_contextual_help("Data Upload")
            
    def _render_analysis_interface(self):
        """Render the main analysis interface when data is loaded."""
        data = st.session_state.data
        
        # Data overview
        st.header("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            st.metric("Data Types", len(data.dtypes.unique()))
            
        # Anomaly Detection Configuration
        st.header("🎯 Anomaly Detection Configuration")
        updated_thresholds = self.ui_components.anomaly_detection_config(
            st.session_state.anomaly_thresholds
        )
        st.session_state.anomaly_thresholds = updated_thresholds
        
        # Run Analysis Button
        if st.button("🚀 Run Fraud Detection Analysis", type="primary"):
            self._run_fraud_analysis()
            
        # Display Results
        if st.session_state.analysis_results:
            self._render_analysis_results()
            
    def _run_fraud_analysis(self):
        """Run the fraud detection analysis."""
        try:
            with st.spinner("Running fraud detection analysis..."):
                data = st.session_state.data
                
                # Log analysis start
                self.monitoring_service.log_user_activity(
                    st.session_state.session_id,
                    "analysis_start",
                    {"data_shape": data.shape}
                )
                
                # Preprocess data
                quality_report = self.data_processor.assess_data_quality(data)
                
                # Handle missing values if needed
                if quality_report["missing_values_summary"]:
                    data = self.data_processor.handle_missing_values(data, strategy="mean")
                
                # Feature engineering (example)
                engineered_features = {
                    "log_amount": {"type": "log_transform", "col": "transaction_amount"}
                } if "transaction_amount" in data.columns else {}
                
                if engineered_features:
                    data = self.model_manager.feature_engineering(data, engineered_features)
                
                # Preprocess for model
                preprocessed_data = self.model_manager.preprocess_data(data)
                
                # Generate mock anomaly scores (in real app, this would use actual models)
                anomaly_scores = np.random.rand(len(data))
                
                # Apply thresholds
                thresholds = st.session_state.anomaly_thresholds
                risk_levels = []
                for score in anomaly_scores:
                    if score >= thresholds["critical"]:
                        risk_levels.append("Critical")
                    elif score >= thresholds["alert"]:
                        risk_levels.append("Alert")
                    elif score >= thresholds["warning"]:
                        risk_levels.append("Warning")
                    else:
                        risk_levels.append("Normal")
                
                # Store results
                st.session_state.analysis_results = {
                    "anomaly_scores": anomaly_scores,
                    "risk_levels": risk_levels,
                    "quality_report": quality_report,
                    "total_anomalies": sum(1 for level in risk_levels if level != "Normal"),
                    "high_risk_sellers": len(set(data.get("seller_id", []))),
                    "avg_anomaly_score": np.mean(anomaly_scores),
                    "risk_distribution": pd.Series(risk_levels).value_counts(),
                    "anomalous_transactions": data[np.array(risk_levels) != "Normal"],
                    "seller_risk_profiles": pd.DataFrame({
                        "seller_id": ["S001", "S002", "S003"],
                        "risk_score": [0.9, 0.5, 0.2],
                        "total_transactions": [10, 50, 100],
                        "anomaly_count": [8, 10, 5]
                    })
                }
                
                # Log analysis completion
                self.monitoring_service.log_user_activity(
                    st.session_state.session_id,
                    "analysis_complete",
                    {"anomalies_detected": st.session_state.analysis_results["total_anomalies"]}
                )
                
                st.success("Analysis completed successfully!")
                
        except Exception as e:
            self.error_handler.handle_exception(e, "fraud_analysis", st.session_state.session_id)
            
    def _render_analysis_results(self):
        """Render the analysis results using the dashboard."""
        st.header("📈 Analysis Results")
        
        # Create and render dashboard
        dashboard = DashboardUI(st.session_state.data, st.session_state.analysis_results)
        dashboard.render()


def main():
    """Main function to run the application."""
    app = FraudDetectionApp()
    app.run()


if __name__ == "__main__":
    main()

