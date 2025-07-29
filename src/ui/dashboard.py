"""
Dashboard and Visualization System for Fraudulent Seller Detection Portal

This module provides the main dashboard UI, including multiple tabs for:
- Executive Overview
- Statistical Analysis
- Seller Risk Management
- Transaction Investigation
- Pattern Analytics
- Reporting & Export

It uses Streamlit for the UI and Plotly for interactive visualizations.

Author: Manus AI
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List


class DashboardUI:
    """
    Manages the display of the main dashboard with its various analytical tabs.
    """
    
    def __init__(self, data: pd.DataFrame, analysis_results: Dict[str, Any]):
        """
        Initialize the Dashboard UI.
        
        Args:
            data: The processed DataFrame containing the data to be visualized.
            analysis_results: A dictionary containing results from backend analysis
                              (e.g., anomaly scores, risk profiles).
        """
        self.data = data
        self.analysis_results = analysis_results
        
    def render(self):
        """
        Renders the main dashboard with tabbed navigation.
        """
        st.title("Fraudulent Seller Detection Dashboard")
        
        tabs = st.tabs([
            "Executive Overview", 
            "Statistical Analysis", 
            "Seller Risk Management", 
            "Transaction Investigation", 
            "Pattern Analytics", 
            "Reporting & Export"
        ])
        
        with tabs[0]:
            self.executive_overview_tab()
        with tabs[1]:
            self.statistical_analysis_tab()
        with tabs[2]:
            self.seller_risk_management_tab()
        with tabs[3]:
            self.transaction_investigation_tab()
        with tabs[4]:
            self.pattern_analytics_tab()
        with tabs[5]:
            self.reporting_export_tab()
            
    def executive_overview_tab(self):
        """
        Renders the Executive Overview tab with KPIs and high-level visualizations.
        """
        st.header("Executive Overview")
        
        # Example KPIs - these would be calculated in the backend
        total_transactions = len(self.data)
        total_anomalies = self.analysis_results.get("total_anomalies", 0)
        anomaly_rate = (total_anomalies / total_transactions) * 100 if total_transactions > 0 else 0
        high_risk_sellers = self.analysis_results.get("high_risk_sellers", 0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total_transactions:,}")
        col2.metric("Detected Anomalies", f"{total_anomalies:,}", f"{anomaly_rate:.2f}%")
        col3.metric("High-Risk Sellers", f"{high_risk_sellers:,}")
        col4.metric("Average Anomaly Score", f"{self.analysis_results.get("avg_anomaly_score", 0):.3f}")
        
        st.subheader("Risk Distribution")
        risk_distribution = self.analysis_results.get("risk_distribution", pd.Series([10, 20, 70], index=["High", "Medium", "Low"]))
        fig = px.pie(risk_distribution, values=risk_distribution.values, names=risk_distribution.index, title="Seller Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Anomaly Trend Over Time")
        anomaly_trend = self.analysis_results.get("anomaly_trend", pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "anomalies": [5, 8, 6]
        }))
        if not anomaly_trend.empty:
            fig_trend = px.line(anomaly_trend, x="date", y="anomalies", title="Daily Anomaly Count")
            st.plotly_chart(fig_trend, use_container_width=True)
            
    def statistical_analysis_tab(self):
        """
        Renders the Statistical Analysis tab with detailed distributions and correlations.
        """
        st.header("Statistical Analysis")
        
        st.subheader("Reconstruction Error Distribution")
        anomaly_scores = self.analysis_results.get("anomaly_scores", np.random.rand(100))
        fig = px.histogram(x=anomaly_scores, nbins=50, title="Distribution of Anomaly Scores")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Analysis")
        # Use a subset of numeric columns for correlation heatmap
        numeric_data = self.data.select_dtypes(include=np.number)
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="Viridis"
            ))
            fig_corr.update_layout(title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric data available for correlation analysis.")
            
    def seller_risk_management_tab(self):
        """
        Renders the Seller Risk Management tab with seller profiles and risk matrix.
        """
        st.header("Seller Risk Management")
        
        st.subheader("Seller Risk Profiles")
        seller_risk_data = self.analysis_results.get("seller_risk_profiles", pd.DataFrame({
            "seller_id": ["S001", "S002", "S003"],
            "risk_score": [0.9, 0.5, 0.2],
            "total_transactions": [10, 50, 100],
            "anomaly_count": [8, 10, 5]
        }))
        st.dataframe(seller_risk_data)
        
        st.subheader("Interactive Risk Matrix")
        if not seller_risk_data.empty:
            fig = px.scatter(
                seller_risk_data,
                x="total_transactions",
                y="risk_score",
                size="anomaly_count",
                color="risk_score",
                hover_name="seller_id",
                title="Seller Risk Matrix",
                labels={"total_transactions": "Total Transactions", "risk_score": "Risk Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No seller risk data to display.")
            
    def transaction_investigation_tab(self):
        """
        Renders the Transaction Investigation tab for deep-diving into specific transactions.
        """
        st.header("Transaction Investigation")
        
        st.subheader("Anomalous Transactions")
        anomalous_transactions = self.analysis_results.get("anomalous_transactions", self.data.head())
        st.dataframe(anomalous_transactions)
        
        st.subheader("Transaction Clustering")
        # Example of clustering visualization
        if "cluster" in self.data.columns:
            fig = px.scatter(
                self.data,
                x="feature1", # Replace with actual features
                y="feature2", # Replace with actual features
                color="cluster",
                title="Transaction Clusters",
                hover_name="transaction_id"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Clustering analysis has not been performed.")
            
    def pattern_analytics_tab(self):
        """
        Renders the Pattern Analytics tab for discovering fraud patterns.
        """
        st.header("Pattern Analytics")
        
        st.subheader("Geographical Fraud Distribution")
        # Example map visualization (requires lat/lon data)
        if "latitude" in self.data.columns and "longitude" in self.data.columns:
            st.map(self.data[["latitude", "longitude"]])
        else:
            st.info("No geographical data available for map visualization.")
            
        st.subheader("Behavioral Pattern Analysis")
        # Example: Heatmap of fraud by time of day and day of week
        if "hour" in self.data.columns and "day_of_week" in self.data.columns:
            behavioral_heatmap = self.data.pivot_table(index="hour", columns="day_of_week", values="anomaly_score", aggfunc="mean")
            fig = px.imshow(behavioral_heatmap, title="Average Anomaly Score by Time and Day")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Time-based features (hour, day_of_week) not available for behavioral analysis.")
            
    def reporting_export_tab(self):
        """
        Renders the Reporting & Export tab for generating reports and exporting data.
        """
        st.header("Reporting & Export")
        
        st.subheader("Custom Report Builder")
        st.info("This is a placeholder for a custom report builder UI.")
        report_title = st.text_input("Report Title", "Fraud Analysis Report")
        include_overview = st.checkbox("Include Executive Overview", True)
        include_stats = st.checkbox("Include Statistical Analysis", True)
        
        if st.button("Generate Report Preview"):
            st.success(f"Generating report: {report_title}")
            # In a real app, this would trigger a PDF/HTML generation process
            
        st.subheader("Export Data")
        export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "JSON"])
        
        if st.button(f"Export Data as {export_format}"):
            if export_format == "CSV":
                csv = self.data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="fraud_data.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                # Requires openpyxl
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine=\"openpyxl\") as writer:
                    self.data.to_excel(writer, index=False, sheet_name="Data")
                excel_data = output.getvalue()
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name="fraud_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = self.data.to_json(orient="records").encode("utf-8")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="fraud_data.json",
                    mime="application/json"
                )


# Example Usage (for testing purposes)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    # Create dummy data and analysis results
    dummy_data = pd.DataFrame({
        "transaction_id": range(100),
        "seller_id": [f"S{i % 10:03d}" for i in range(100)],
        "transaction_amount": np.random.lognormal(mean=3, sigma=1, size=100),
        "anomaly_score": np.random.rand(100),
        "risk_level": np.random.choice(["Low", "Medium", "High"], 100, p=[0.7, 0.2, 0.1]),
        "transaction_date": pd.to_datetime(np.random.randint(1672531200, 1675209600, size=100), unit=\"s\"),
        "latitude": np.random.uniform(34, 40, 100),
        "longitude": np.random.uniform(-122, -118, 100),
        "hour": np.random.randint(0, 24, 100),
        "day_of_week": np.random.randint(0, 7, 100)
    })
    
    dummy_analysis_results = {
        "total_anomalies": (dummy_data["anomaly_score"] > 0.8).sum(),
        "high_risk_sellers": len(dummy_data[dummy_data["risk_level"] == "High"]["seller_id"].unique()),
        "avg_anomaly_score": dummy_data["anomaly_score"].mean(),
        "risk_distribution": dummy_data["risk_level"].value_counts(),
        "anomaly_trend": dummy_data.groupby(dummy_data["transaction_date"].dt.date)["anomaly_score"].count().reset_index().rename(columns={\"transaction_date\": \"date\", \"anomaly_score\": \"anomalies\"}),
        "anomaly_scores": dummy_data["anomaly_score"],
        "seller_risk_profiles": dummy_data.groupby("seller_id").agg(
            risk_score=("anomaly_score", "mean"),
            total_transactions=("transaction_id", "count"),
            anomaly_count=("anomaly_score", lambda x: (x > 0.8).sum())
        ).reset_index(),
        "anomalous_transactions": dummy_data[dummy_data["anomaly_score"] > 0.8]
    }
    
    dashboard = DashboardUI(dummy_data, dummy_analysis_results)
    dashboard.render()



