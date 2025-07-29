"""
Config Manager Module for Fraudulent Seller Detection Portal

This module provides centralized configuration management for the application.
It handles loading, validation, and access to application settings from various sources.

Author: Manus AI
Version: 1.0.0
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """
    Manages application configuration, allowing settings to be loaded from
    files (JSON, YAML), environment variables, or provided directly.
    """
    
    def __init__(self, config_file: Optional[str] = None, env_prefix: str = "APP_"):
        """
        Initialize the ConfigManager.
        
        Args:
            config_file: Path to a configuration file (e.g., config.json, config.yaml).
            env_prefix: Prefix for environment variables to load (e.g., "APP_" for APP_DEBUG).
        """
        self.logger = self._setup_logging()
        self._config: Dict[str, Any] = {}
        self.env_prefix = env_prefix
        
        # Load default configurations
        self._load_defaults()
        
        # Load from file if provided
        if config_file:
            self._load_from_file(config_file)
            
        # Load from environment variables (overrides file and defaults)
        self._load_from_environment()
        
        self.logger.info("ConfigManager initialized.")
        
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for the ConfigManager.
        """
        logger = logging.getLogger("config_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("logs/config_manager.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_defaults(self):
        """
        Loads default configuration settings.
        """
        self._config.update({
            "debug_mode": False,
            "log_level": "INFO",
            "database": {
                "host": "localhost",
                "port": 5432,
                "user": "admin",
                "password": "password",
                "name": "fraud_db"
            },
            "model_settings": {
                "active_model": "default_fraud_model",
                "threshold": 0.7,
                "retrain_interval_days": 30
            },
            "security": {
                "max_file_size_mb": 100,
                "session_timeout_hours": 24
            }
        })
        self.logger.info("Default configurations loaded.")
        
    def _load_from_file(self, config_file: str):
        """
        Loads configuration from a specified file (JSON or YAML).
        
        Args:
            config_file: Path to the configuration file.
        """
        if not os.path.exists(config_file):
            self.logger.warning(f"Config file not found: {config_file}")
            return
            
        try:
            with open(config_file, "r") as f:
                if config_file.endswith(".json"):
                    file_config = json.load(f)
                elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    file_config = yaml.safe_load(f)
                else:
                    self.logger.warning(f"Unsupported config file format: {config_file}")
                    return
                    
            self._config = self._deep_merge(self._config, file_config)
            self.logger.info(f"Configuration loaded from file: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading config from {config_file}: {e}")
            raise ConfigError(f"Failed to load configuration from file: {str(e)}")
            
    def _load_from_environment(self):
        """
        Loads configuration from environment variables.
        Environment variables should be prefixed (e.g., APP_DEBUG, APP_DATABASE_HOST).
        Nested keys are separated by underscores (e.g., APP_MODEL_SETTINGS_THRESHOLD).
        """
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Convert APP_DATABASE_HOST to database.host
                clean_key = key[len(self.env_prefix):].lower()
                parts = clean_key.split("_")
                
                current_dict = env_config
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        # Attempt to convert value to appropriate type
                        if value.lower() == "true":
                            current_dict[part] = True
                        elif value.lower() == "false":
                            current_dict[part] = False
                        elif value.isdigit():
                            current_dict[part] = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            current_dict[part] = float(value)
                        else:
                            current_dict[part] = value
                    else:
                        if part not in current_dict:
                            current_dict[part] = {}
                        current_dict = current_dict[part]
                        
        self._config = self._deep_merge(self._config, env_config)
        self.logger.info("Configuration loaded from environment variables.")
        
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """
        Recursively merges dict2 into dict1.
        """
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                dict1[key] = self._deep_merge(dict1[key], value)
            else:
                dict1[key] = value
        return dict1
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key (e.g., "database.host").
        
        Args:
            key: Dot-separated string representing the configuration path.
            default: Default value to return if the key is not found.
            
        Returns:
            The configuration value or the default value.
        """
        parts = key.split(".")
        current_value = self._config
        for part in parts:
            if isinstance(current_value, dict) and part in current_value:
                current_value = current_value[part]
            else:
                return default
        return current_value
        
    def set(self, key: str, value: Any):
        """
        Sets a configuration value using a dot-separated key.
        
        Args:
            key: Dot-separated string representing the configuration path.
            value: The value to set.
        """
        parts = key.split(".")
        current_dict = self._config
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current_dict[part] = value
            else:
                if part not in current_dict or not isinstance(current_dict[part], dict):
                    current_dict[part] = {}
                current_dict = current_dict[part]
        self.logger.info(f"Configuration key \'{key}\' set to \'{value}\'")
        
    def all(self) -> Dict[str, Any]:
        """
        Returns the entire configuration dictionary.
        """
        return self._config
        
    def validate_config(self, schema: Dict[str, Any]) -> bool:
        """
        Validates the current configuration against a provided schema.
        This is a basic validation; for complex schemas, consider libraries like Cerberus or Pydantic.
        
        Args:
            schema: A dictionary defining the expected structure and types.
                    Example: {"debug_mode": bool, "database.host": str}
                    
        Returns:
            True if configuration is valid, False otherwise.
        """
        self.logger.info("Validating configuration against schema...")
        is_valid = True
        for key, expected_type in schema.items():
            value = self.get(key)
            if value is None:
                self.logger.error(f"Validation Error: Missing required configuration key: {key}")
                is_valid = False
            elif not isinstance(value, expected_type):
                self.logger.error(f"Validation Error: Key \'{key}\' has type {type(value).__name__}, expected {expected_type.__name__}")
                is_valid = False
                
        if is_valid:
            self.logger.info("Configuration validation successful.")
        else:
            self.logger.warning("Configuration validation failed.")
        return is_valid


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Clean up old logs for a fresh test run
    if os.path.exists("logs/config_manager.log"):
        os.remove("logs/config_manager.log")
    if os.path.exists("logs"):
        os.rmdir("logs")
        
    # Test 1: Default configuration
    print("\n--- Test 1: Default Configuration ---")
    config_manager = ConfigManager()
    print(f"Debug Mode: {config_manager.get("debug_mode")}")
    print(f"DB Host: {config_manager.get("database.host")}")
    print(f"Model Threshold: {config_manager.get("model_settings.threshold")}")
    
    # Test 2: Load from JSON file
    print("\n--- Test 2: Load from JSON File ---")
    json_config_content = {
        "debug_mode": True,
        "database": {"port": 5433},
        "new_setting": "test_value"
    }
    with open("test_config.json", "w") as f:
        json.dump(json_config_content, f)
        
    config_manager_json = ConfigManager(config_file="test_config.json")
    print(f"Debug Mode (from JSON): {config_manager_json.get("debug_mode")}")
    print(f"DB Port (merged from JSON): {config_manager_json.get("database.port")}")
    print(f"New Setting (from JSON): {config_manager_json.get("new_setting")}")
    os.remove("test_config.json")
    
    # Test 3: Load from YAML file
    print("\n--- Test 3: Load from YAML File ---")
    yaml_config_content = """
    debug_mode: false
    model_settings:
      threshold: 0.85
      new_model_param: 123
    """
    with open("test_config.yaml", "w") as f:
        f.write(yaml_config_content)
        
    config_manager_yaml = ConfigManager(config_file="test_config.yaml")
    print(f"Debug Mode (from YAML): {config_manager_yaml.get("debug_mode")}")
    print(f"Model Threshold (from YAML): {config_manager_yaml.get("model_settings.threshold")}")
    print(f"New Model Param (from YAML): {config_manager_yaml.get("model_settings.new_model_param")}")
    os.remove("test_config.yaml")
    
    # Test 4: Load from Environment Variables
    print("\n--- Test 4: Load from Environment Variables ---")
    os.environ["APP_DEBUG_MODE"] = "true"
    os.environ["APP_DATABASE_USER"] = "prod_user"
    os.environ["APP_MODEL_SETTINGS_THRESHOLD"] = "0.95"
    os.environ["APP_SECURITY_MAX_FILE_SIZE_MB"] = "200"
    
    config_manager_env = ConfigManager()
    print(f"Debug Mode (from ENV): {config_manager_env.get("debug_mode")}")
    print(f"DB User (from ENV): {config_manager_env.get("database.user")}")
    print(f"Model Threshold (from ENV): {config_manager_env.get("model_settings.threshold")}")
    print(f"Max File Size (from ENV): {config_manager_env.get("security.max_file_size_mb")}")
    
    del os.environ["APP_DEBUG_MODE"]
    del os.environ["APP_DATABASE_USER"]
    del os.environ["APP_MODEL_SETTINGS_THRESHOLD"]
    del os.environ["APP_SECURITY_MAX_FILE_SIZE_MB"]
    
    # Test 5: Set and Get
    print("\n--- Test 5: Set and Get ---")
    config_manager.set("new_runtime_setting", "dynamic_value")
    config_manager.set("database.timeout", 60)
    print(f"New Runtime Setting: {config_manager.get("new_runtime_setting")}")
    print(f"DB Timeout: {config_manager.get("database.timeout")}")
    
    # Test 6: Validation
    print("\n--- Test 6: Validation ---")
    schema = {
        "debug_mode": bool,
        "database.host": str,
        "database.port": int,
        "model_settings.threshold": float,
        "non_existent_key": str # This will cause validation to fail
    }
    is_valid = config_manager.validate_config(schema)
    print(f"Configuration is valid: {is_valid}")
    
    # Clean up logs directory if created
    if os.path.exists("logs/config_manager.log"):
        os.remove("logs/config_manager.log")
    if os.path.exists("logs"):
        os.rmdir("logs")




