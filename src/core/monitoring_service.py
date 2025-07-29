"""
Monitoring Service Module for Fraudulent Seller Detection Portal

This module provides comprehensive monitoring, logging, and analytics capabilities,
including:
- Real-time performance metrics dashboard
- Error tracking and alerting system
- User behavior analytics and usage patterns
- Comprehensive audit trail for all operations
- Security event logging and compliance reporting

Author: Manus AI
Version: 1.0.0
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd


class MonitoringService:
    """
    Provides services for application monitoring, logging, and analytics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Monitoring Service.
        
        Args:
            config: Configuration dictionary for monitoring settings.
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        self.performance_log_path = self.config.get("performance_log_path", "logs/performance.log")
        self.error_log_path = self.config.get("error_log_path", "logs/errors.log")
        self.user_activity_log_path = self.config.get("user_activity.log_path", "logs/user_activity.log")
        self.security_event_log_path = self.config.get("security_event_log_path", "logs/security_events.log")
        self._ensure_log_directories()
        
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for the Monitoring Service.
        """
        logger = logging.getLogger("monitoring_service")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for persistent logs
            file_handler = logging.FileHandler("logs/monitoring_service.log")
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def _ensure_log_directories(self):
        """
        Ensure all necessary log directories exist.
        """
        log_dir = os.path.dirname(self.performance_log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_dir = os.path.dirname(self.error_log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_dir = os.path.dirname(self.user_activity_log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_dir = os.path.dirname(self.security_event_log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def log_performance_metric(self, metric_name: str, value: Any, tags: Optional[Dict[str, Any]] = None):
        """
        Logs a performance metric.
        
        Args:
            metric_name: Name of the metric (e.g., "cpu_usage", "memory_usage").
            value: The value of the metric.
            tags: Optional dictionary of additional tags (e.g., {"component": "data_processor"}).
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance",
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {}
        }
        self._write_log(self.performance_log_path, log_entry)
        self.logger.info(f"Performance metric logged: {metric_name}={value}")
        
    def log_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Logs an error event.
        
        Args:
            error_type: Type of error (e.g., "FileNotFound", "DatabaseError").
            message: A descriptive error message.
            details: Optional dictionary of additional error details (e.g., stack trace).
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": error_type,
            "message": message,
            "details": details or {}
        }
        self._write_log(self.error_log_path, log_entry)
        self.logger.error(f"Error logged: {error_type} - {message}")
        
    def log_user_activity(self, user_id: str, activity_type: str, details: Optional[Dict[str, Any]] = None):
        """
        Logs user activity for behavior analytics.
        
        Args:
            user_id: Identifier for the user.
            activity_type: Type of activity (e.g., "file_upload", "model_run", "report_export").
            details: Optional dictionary of additional activity details.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "user_activity",
            "user_id": user_id,
            "activity_type": activity_type,
            "details": details or {}
        }
        self._write_log(self.user_activity_log_path, log_entry)
        self.logger.info(f"User activity logged for {user_id}: {activity_type}")
        
    def log_security_event(self, event_type: str, message: str, severity: str, details: Optional[Dict[str, Any]] = None):
        """
        Logs a security-related event.
        
        Args:
            event_type: Type of security event (e.g., "LoginAttempt", "DataBreachAttempt").
            message: A descriptive message about the event.
            severity: Severity level (e.g., "low", "medium", "high", "critical").
            details: Optional dictionary of additional event details.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "security_event",
            "event_type": event_type,
            "message": message,
            "severity": severity,
            "details": details or {}
        }
        self._write_log(self.security_event_log_path, log_entry)
        self.logger.warning(f"Security event logged ({severity}): {event_type} - {message}")
        
    def _write_log(self, log_file_path: str, entry: Dict[str, Any]):
        """
        Helper method to write a log entry to a specified file.
        """
        try:
            with open(log_file_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write log to {log_file_path}: {e}")
            
    def get_logs(self, log_file_path: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Retrieves log entries from a specified log file within a time range.
        
        Args:
            log_file_path: Path to the log file.
            start_time: Optional start datetime for filtering logs.
            end_time: Optional end datetime for filtering logs.
            
        Returns:
            A list of log entries.
        """
        logs = []
        if not os.path.exists(log_file_path):
            return logs
            
        try:
            with open(log_file_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_timestamp = datetime.fromisoformat(entry["timestamp"])
                        
                        if (start_time is None or entry_timestamp >= start_time) and \
                           (end_time is None or entry_timestamp <= end_time):
                            logs.append(entry)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping malformed log entry in {log_file_path}: {line.strip()}")
                        continue
        except Exception as e:
            self.logger.error(f"Error reading log file {log_file_path}: {e}")
            
        return logs
    
    def generate_performance_report(self, duration_days: int = 7) -> Dict[str, Any]:
        """
        Generates a summary report of application performance metrics.
        
        Args:
            duration_days: Number of past days to include in the report.
            
        Returns:
            A dictionary containing performance statistics.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=duration_days)
        
        performance_logs = self.get_logs(self.performance_log_path, start_time, end_time)
        
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "duration_days": duration_days,
            "total_performance_logs": len(performance_logs),
            "metrics_summary": {}
        }
        
        df = pd.DataFrame(performance_logs)
        if not df.empty:
            # Convert 'value' column to numeric where possible
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            for metric_name in df["metric_name"].unique():
                metric_df = df[df["metric_name"] == metric_name]["value"].dropna()
                if not metric_df.empty:
                    report["metrics_summary"][metric_name] = {
                        "average": metric_df.mean(),
                        "min": metric_df.min(),
                        "max": metric_df.max(),
                        "std_dev": metric_df.std()
                    }
        
        self.logger.info("Performance report generated.")
        return report
    
    def generate_error_report(self, duration_days: int = 7) -> Dict[str, Any]:
        """
        Generates a summary report of error events.
        
        Args:
            duration_days: Number of past days to include in the report.
            
        Returns:
            A dictionary containing error statistics.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=duration_days)
        
        error_logs = self.get_logs(self.error_log_path, start_time, end_time)
        
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "duration_days": duration_days,
            "total_errors": len(error_logs),
            "error_type_counts": pd.DataFrame(error_logs)["error_type"].value_counts().to_dict() if error_logs else {},
            "recent_errors": error_logs[-5:] # Last 5 errors
        }
        
        self.logger.info("Error report generated.")
        return report
    
    def generate_user_activity_report(self, duration_days: int = 7) -> Dict[str, Any]:
        """
        Generates a summary report of user activities.
        
        Args:
            duration_days: Number of past days to include in the report.
            
        Returns:
            A dictionary containing user activity statistics.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=duration_days)
        
        user_activity_logs = self.get_logs(self.user_activity_log_path, start_time, end_time)
        
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "duration_days": duration_days,
            "total_activities": len(user_activity_logs),
            "activity_type_counts": pd.DataFrame(user_activity_logs)["activity_type"].value_counts().to_dict() if user_activity_logs else {},
            "top_users": pd.DataFrame(user_activity_logs)["user_id"].value_counts().head(5).to_dict() if user_activity_logs else {},
            "recent_activities": user_activity_logs[-5:]
        }
        
        self.logger.info("User activity report generated.")
        return report
    
    def generate_security_event_report(self, duration_days: int = 7) -> Dict[str, Any]:
        """
        Generates a summary report of security events.
        
        Args:
            duration_days: Number of past days to include in the report.
            
        Returns:
            A dictionary containing security event statistics.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=duration_days)
        
        security_event_logs = self.get_logs(self.security_event_log_path, start_time, end_time)
        
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "duration_days": duration_days,
            "total_security_events": len(security_event_logs),
            "event_type_counts": pd.DataFrame(security_event_logs)["event_type"].value_counts().to_dict() if security_event_logs else {},
            "severity_counts": pd.DataFrame(security_event_logs)["severity"].value_counts().to_dict() if security_event_logs else {},
            "recent_security_events": security_event_logs[-5:]
        }
        
        self.logger.info("Security event report generated.")
        return report


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Clean up old logs for a fresh test run
    for log_file in ["logs/performance.log", "logs/errors.log", "logs/user_activity.log", "logs/security_events.log", "logs/monitoring_service.log"]:
        if os.path.exists(log_file):
            os.remove(log_file)
    if os.path.exists("logs"):
        os.rmdir("logs")
            
    monitor = MonitoringService()
    
    print("\n--- Logging Performance Metrics ---")
    monitor.log_performance_metric("cpu_usage", 25.5, {"component": "main_app"})
    monitor.log_performance_metric("memory_usage", 60.2, {"component": "data_processor"})
    monitor.log_performance_metric("response_time_ms", 150, {"endpoint": "/api/predict"})
    
    print("\n--- Logging Errors ---")
    monitor.log_error("ValueError", "Invalid input data format", {"input": "abc.txt"})
    monitor.log_error("DatabaseError", "Failed to connect to DB", {"db_host": "localhost"})
    
    print("\n--- Logging User Activity ---")
    monitor.log_user_activity("user_123", "file_upload", {"filename": "data.csv", "size": "10MB"})
    monitor.log_user_activity("user_456", "model_run", {"model": "fraud_detector_v1"})
    
    print("\n--- Logging Security Events ---")
    monitor.log_security_event("LoginAttempt", "Failed login from unknown IP", "medium", {"ip": "192.168.1.1"})
    monitor.log_security_event("DataExfiltration", "Large data download detected", "critical", {"user": "admin", "data_size": "500MB"})
    
    # Give some time for logs to be written if there's any buffer
    import time
    time.sleep(0.1)
    
    print("\n--- Generating Reports ---")
    perf_report = monitor.generate_performance_report()
    print("Performance Report:", json.dumps(perf_report, indent=2))
    
    error_report = monitor.generate_error_report()
    print("Error Report:", json.dumps(error_report, indent=2))
    
    user_activity_report = monitor.generate_user_activity_report()
    print("User Activity Report:", json.dumps(user_activity_report, indent=2))
    
    security_event_report = monitor.generate_security_event_report()
    print("Security Event Report:", json.dumps(security_event_report, indent=2))
    
    # Clean up logs directory after testing
    for log_file in ["logs/performance.log", "logs/errors.log", "logs/user_activity.log", "logs/security_events.log", "logs/monitoring_service.log"]:
        if os.path.exists(log_file):
            os.remove(log_file)
    if os.path.exists("logs"):
        os.rmdir("logs")




