"""
Security Manager Module for Fraudulent Seller Detection Portal

This module provides comprehensive security and data privacy controls including:
- Input validation and sanitization
- Data encryption and protection
- GDPR compliance features
- Audit logging
- Session management

Author: Manus AI
Version: 1.0.0
"""

import hashlib
import hmac
import secrets
import logging
import json
import os
import re
import tempfile
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pandas as pd
import streamlit as st
from email_validator import validate_email, EmailNotValidError
import magic
import yara


class SecurityManager:
    """
    Comprehensive security manager for handling data protection, validation,
    and privacy compliance in the fraud detection portal.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Security Manager with configuration settings.
        
        Args:
            config: Configuration dictionary containing security settings
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.session_timeout = timedelta(hours=24)
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB
        self.allowed_file_types = self.config.get('allowed_file_types', [
            'text/csv', 'application/vnd.ms-excel', 
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/json'
        ])
        self.audit_log_path = self.config.get('audit_log_path', 'logs/audit.log')
        self._ensure_log_directory()
        self._initialize_malware_scanner()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up secure logging configuration."""
        logger = logging.getLogger('security_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler('logs/security.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _ensure_log_directory(self):
        """Ensure log directory exists."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for data protection."""
        key_file = Path('.encryption_key')
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key
    
    def _initialize_malware_scanner(self):
        """Initialize YARA rules for malware detection."""
        try:
            # Basic YARA rules for common malware patterns
            rules_content = """
            rule SuspiciousExecutable {
                strings:
                    $exe = { 4D 5A }  // MZ header
                    $script = "eval("
                    $script2 = "exec("
                condition:
                    $exe at 0 or any of ($script*)
            }
            
            rule SuspiciousArchive {
                strings:
                    $zip = { 50 4B 03 04 }  // ZIP header
                    $rar = { 52 61 72 21 }  // RAR header
                condition:
                    any of them
            }
            """
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yar', delete=False) as f:
                f.write(rules_content)
                self.yara_rules_file = f.name
                
            self.yara_rules = yara.compile(filepath=self.yara_rules_file)
            
        except Exception as e:
            self.logger.warning(f"Could not initialize YARA scanner: {e}")
            self.yara_rules = None
    
    def validate_file_upload(self, uploaded_file) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive file validation including size, type, and malware scanning.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message, file_metadata)
        """
        try:
            # File size validation
            if uploaded_file.size > self.max_file_size:
                return False, f"File size ({uploaded_file.size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)", {}
            
            # File type validation using python-magic
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            mime_type = magic.from_buffer(file_content, mime=True)
            
            if mime_type not in self.allowed_file_types:
                return False, f"File type '{mime_type}' is not allowed. Allowed types: {', '.join(self.allowed_file_types)}", {}
            
            # Malware scanning
            if self.yara_rules:
                matches = self.yara_rules.match(data=file_content)
                if matches:
                    self.logger.warning(f"Malware detected in file {uploaded_file.name}: {[m.rule for m in matches]}")
                    return False, "File contains suspicious content and cannot be processed", {}
            
            # File structure validation for CSV/Excel
            validation_result = self._validate_file_structure(uploaded_file, mime_type)
            if not validation_result['valid']:
                return False, validation_result['error'], {}
            
            # Generate file metadata
            file_metadata = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'mime_type': mime_type,
                'upload_time': datetime.now().isoformat(),
                'hash': hashlib.sha256(file_content).hexdigest(),
                'columns': validation_result.get('columns', []),
                'row_count': validation_result.get('row_count', 0)
            }
            
            # Log successful validation
            self._audit_log('file_upload_validated', {
                'filename': uploaded_file.name,
                'size': uploaded_file.size,
                'mime_type': mime_type,
                'hash': file_metadata['hash']
            })
            
            return True, "File validation successful", file_metadata
            
        except Exception as e:
            self.logger.error(f"File validation error: {e}")
            return False, f"File validation failed: {str(e)}", {}
    
    def _validate_file_structure(self, uploaded_file, mime_type: str) -> Dict[str, Any]:
        """Validate the internal structure of uploaded files."""
        try:
            if mime_type == 'text/csv':
                df = pd.read_csv(uploaded_file, nrows=5)  # Read first 5 rows for validation
                uploaded_file.seek(0)
                
                return {
                    'valid': True,
                    'columns': df.columns.tolist(),
                    'row_count': len(df)
                }
                
            elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                df = pd.read_excel(uploaded_file, nrows=5)
                uploaded_file.seek(0)
                
                return {
                    'valid': True,
                    'columns': df.columns.tolist(),
                    'row_count': len(df)
                }
                
            elif mime_type == 'application/json':
                content = uploaded_file.read().decode('utf-8')
                uploaded_file.seek(0)
                
                data = json.loads(content)
                if isinstance(data, list) and len(data) > 0:
                    columns = list(data[0].keys()) if isinstance(data[0], dict) else []
                else:
                    columns = []
                
                return {
                    'valid': True,
                    'columns': columns,
                    'row_count': len(data) if isinstance(data, list) else 1
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Invalid file structure: {str(e)}"
            }
    
    def sanitize_input(self, input_data: Any) -> Any:
        """
        Sanitize user input to prevent XSS and injection attacks.
        
        Args:
            input_data: Input data to sanitize
            
        Returns:
            Sanitized input data
        """
        if isinstance(input_data, str):
            # Remove potentially dangerous characters and patterns
            sanitized = re.sub(r'[<>"\']', '', input_data)
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
            sanitized = sanitized.strip()
            
            # Log if sanitization occurred
            if sanitized != input_data:
                self._audit_log('input_sanitized', {
                    'original_length': len(input_data),
                    'sanitized_length': len(sanitized)
                })
            
            return sanitized
            
        elif isinstance(input_data, dict):
            return {key: self.sanitize_input(value) for key, value in input_data.items()}
            
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
            
        return input_data
    
    def encrypt_sensitive_data(self, data: Union[str, bytes, Dict]) -> str:
        """
        Encrypt sensitive data for secure storage.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.cipher_suite.encrypt(data)
            
            self._audit_log('data_encrypted', {
                'data_size': len(data),
                'timestamp': datetime.now().isoformat()
            })
            
            return encrypted_data.decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise SecurityError(f"Failed to encrypt data: {str(e)}")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, Dict]:
        """
        Decrypt previously encrypted data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = encrypted_data.encode('utf-8')
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            
            # Try to parse as JSON first
            try:
                return json.loads(decrypted_data.decode('utf-8'))
            except json.JSONDecodeError:
                return decrypted_data.decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise SecurityError(f"Failed to decrypt data: {str(e)}")
    
    def validate_email(self, email: str) -> Tuple[bool, str]:
        """
        Validate email address format.
        
        Args:
            email: Email address to validate
            
        Returns:
            Tuple of (is_valid, normalized_email)
        """
        try:
            validated_email = validate_email(email)
            return True, validated_email.email
        except EmailNotValidError:
            return False, ""
    
    def generate_session_token(self) -> str:
        """Generate a secure session token."""
        token = secrets.token_urlsafe(32)
        
        self._audit_log('session_token_generated', {
            'token_length': len(token),
            'timestamp': datetime.now().isoformat()
        })
        
        return token
    
    def validate_session_token(self, token: str, created_time: datetime) -> bool:
        """
        Validate session token and check expiration.
        
        Args:
            token: Session token to validate
            created_time: When the token was created
            
        Returns:
            True if token is valid and not expired
        """
        if not token or len(token) < 32:
            return False
        
        if datetime.now() - created_time > self.session_timeout:
            self._audit_log('session_expired', {
                'token_age': str(datetime.now() - created_time),
                'timestamp': datetime.now().isoformat()
            })
            return False
        
        return True
    
    def _audit_log(self, action: str, details: Dict[str, Any]):
        """
        Log security-related actions for audit purposes.
        
        Args:
            action: Action being logged
            details: Additional details about the action
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
            'session_id': getattr(st.session_state, 'session_id', 'unknown')
        }
        
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    def get_audit_logs(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """
        Retrieve audit logs for a specified date range.
        
        Args:
            start_date: Start date for log retrieval
            end_date: End date for log retrieval
            
        Returns:
            List of audit log entries
        """
        logs = []
        
        try:
            if not Path(self.audit_log_path).exists():
                return logs
            
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        
                        if start_date and entry_time < start_date:
                            continue
                        if end_date and entry_time > end_date:
                            continue
                            
                        logs.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
        except Exception as e:
            self.logger.error(f"Failed to read audit logs: {e}")
        
        return logs
    
    def gdpr_data_export(self, session_id: str) -> Dict[str, Any]:
        """
        Export all data associated with a session for GDPR compliance.
        
        Args:
            session_id: Session ID to export data for
            
        Returns:
            Dictionary containing all user data
        """
        export_data = {
            'session_id': session_id,
            'export_timestamp': datetime.now().isoformat(),
            'audit_logs': [],
            'processed_files': [],
            'user_preferences': {}
        }
        
        # Get audit logs for this session
        all_logs = self.get_audit_logs()
        export_data['audit_logs'] = [
            log for log in all_logs 
            if log.get('session_id') == session_id
        ]
        
        self._audit_log('gdpr_data_export', {
            'session_id': session_id,
            'records_exported': len(export_data['audit_logs'])
        })
        
        return export_data
    
    def gdpr_data_deletion(self, session_id: str) -> bool:
        """
        Delete all data associated with a session for GDPR compliance.
        
        Args:
            session_id: Session ID to delete data for
            
        Returns:
            True if deletion was successful
        """
        try:
            # Remove session data from audit logs
            if Path(self.audit_log_path).exists():
                with open(self.audit_log_path, 'r') as f:
                    lines = f.readlines()
                
                filtered_lines = []
                deleted_count = 0
                
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get('session_id') != session_id:
                            filtered_lines.append(line)
                        else:
                            deleted_count += 1
                    except json.JSONDecodeError:
                        filtered_lines.append(line)
                
                with open(self.audit_log_path, 'w') as f:
                    f.writelines(filtered_lines)
                
                self._audit_log('gdpr_data_deletion', {
                    'session_id': session_id,
                    'records_deleted': deleted_count
                })
                
                return True
                
        except Exception as e:
            self.logger.error(f"GDPR deletion failed: {e}")
            return False
    
    def check_data_retention_policy(self):
        """Check and enforce data retention policies."""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)  # 30-day retention
            
            if Path(self.audit_log_path).exists():
                with open(self.audit_log_path, 'r') as f:
                    lines = f.readlines()
                
                retained_lines = []
                deleted_count = 0
                
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        
                        if entry_time >= cutoff_date:
                            retained_lines.append(line)
                        else:
                            deleted_count += 1
                    except (json.JSONDecodeError, KeyError):
                        retained_lines.append(line)  # Keep malformed entries
                
                if deleted_count > 0:
                    with open(self.audit_log_path, 'w') as f:
                        f.writelines(retained_lines)
                    
                    self._audit_log('data_retention_cleanup', {
                        'records_deleted': deleted_count,
                        'cutoff_date': cutoff_date.isoformat()
                    })
                    
        except Exception as e:
            self.logger.error(f"Data retention cleanup failed: {e}")
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'audit_log_entries': 0,
            'recent_activities': [],
            'security_events': [],
            'file_uploads': 0,
            'data_encryptions': 0,
            'session_activities': 0
        }
        
        try:
            # Analyze recent audit logs
            recent_logs = self.get_audit_logs(
                start_date=datetime.now() - timedelta(days=7)
            )
            
            report['audit_log_entries'] = len(recent_logs)
            
            # Categorize activities
            activity_counts = {}
            for log in recent_logs:
                action = log.get('action', 'unknown')
                activity_counts[action] = activity_counts.get(action, 0) + 1
            
            report['recent_activities'] = [
                {'action': action, 'count': count}
                for action, count in activity_counts.items()
            ]
            
            # Identify security events
            security_actions = [
                'malware_detected', 'file_validation_failed', 
                'input_sanitized', 'session_expired'
            ]
            
            report['security_events'] = [
                log for log in recent_logs 
                if log.get('action') in security_actions
            ]
            
            report['file_uploads'] = activity_counts.get('file_upload_validated', 0)
            report['data_encryptions'] = activity_counts.get('data_encrypted', 0)
            report['session_activities'] = activity_counts.get('session_token_generated', 0)
            
        except Exception as e:
            self.logger.error(f"Security report generation failed: {e}")
            report['error'] = str(e)
        
        return report


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class GDPRCompliance:
    """
    GDPR compliance utilities for data protection and user rights.
    """
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.consent_log_path = 'logs/consent.log'
        self._ensure_consent_log()
    
    def _ensure_consent_log(self):
        """Ensure consent log file exists."""
        Path('logs').mkdir(exist_ok=True)
        if not Path(self.consent_log_path).exists():
            with open(self.consent_log_path, 'w') as f:
                f.write('')  # Create empty file
    
    def record_consent(self, session_id: str, consent_type: str, granted: bool):
        """
        Record user consent for data processing.
        
        Args:
            session_id: User session ID
            consent_type: Type of consent (e.g., 'data_processing', 'analytics')
            granted: Whether consent was granted
        """
        consent_record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'consent_type': consent_type,
            'granted': granted,
            'ip_hash': hashlib.sha256(
                st.session_state.get('client_ip', 'unknown').encode()
            ).hexdigest()
        }
        
        try:
            with open(self.consent_log_path, 'a') as f:
                f.write(json.dumps(consent_record) + '\n')
        except Exception as e:
            self.security_manager.logger.error(f"Failed to record consent: {e}")
    
    def get_user_consents(self, session_id: str) -> List[Dict]:
        """Get all consent records for a user session."""
        consents = []
        
        try:
            with open(self.consent_log_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record.get('session_id') == session_id:
                            consents.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.security_manager.logger.error(f"Failed to read consent log: {e}")
        
        return consents
    
    def has_valid_consent(self, session_id: str, consent_type: str) -> bool:
        """Check if user has valid consent for a specific type of processing."""
        consents = self.get_user_consents(session_id)
        
        # Get the most recent consent for this type
        relevant_consents = [
            c for c in consents 
            if c.get('consent_type') == consent_type
        ]
        
        if not relevant_consents:
            return False
        
        # Sort by timestamp and get the most recent
        relevant_consents.sort(key=lambda x: x['timestamp'], reverse=True)
        latest_consent = relevant_consents[0]
        
        return latest_consent.get('granted', False)


# Example usage and testing
if __name__ == "__main__":
    # Initialize security manager
    config = {
        'max_file_size': 50 * 1024 * 1024,  # 50MB
        'allowed_file_types': ['text/csv', 'application/json'],
        'audit_log_path': 'logs/test_audit.log'
    }
    
    security_manager = SecurityManager(config)
    
    # Test input sanitization
    malicious_input = "<script>alert('xss')</script>Hello World"
    sanitized = security_manager.sanitize_input(malicious_input)
    print(f"Original: {malicious_input}")
    print(f"Sanitized: {sanitized}")
    
    # Test encryption
    sensitive_data = {"user_id": "12345", "transaction_amount": 1000.50}
    encrypted = security_manager.encrypt_sensitive_data(sensitive_data)
    decrypted = security_manager.decrypt_sensitive_data(encrypted)
    print(f"Original: {sensitive_data}")
    print(f"Decrypted: {decrypted}")
    
    # Generate security report
    report = security_manager.generate_security_report()
    print(f"Security Report: {json.dumps(report, indent=2)}")

