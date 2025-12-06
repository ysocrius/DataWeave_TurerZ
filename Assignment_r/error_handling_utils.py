#!/usr/bin/env python3
"""
error_handling_utils.py
-----------------------
Comprehensive error handling utilities for the AI Document Processor.

Provides graceful failure recovery, error categorization, and system resilience
for all processing components including deduplication, LLM processing, and data validation.
"""

import logging
import traceback
import time
from typing import Dict, List, Any, Optional, Callable, Union
from functools import wraps
from enum import Enum


class ErrorCategory(Enum):
    """Error categories for better error handling and recovery"""
    VALIDATION_ERROR = "validation_error"
    DATA_ERROR = "data_error"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorHandler:
    """
    Comprehensive error handler with graceful failure recovery and detailed logging
    """
    
    def __init__(self, component_name: str = "DocumentProcessor"):
        self.component_name = component_name
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "critical_failures": 0
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging for error handling"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{self.component_name}_ErrorHandler")
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for appropriate handling strategy"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # API and network errors
        if any(term in error_str for term in ['api', 'openai', 'rate limit', '429', '503', '502']):
            return ErrorCategory.API_ERROR
        
        if any(term in error_str for term in ['connection', 'network', 'timeout', 'dns']):
            return ErrorCategory.NETWORK_ERROR
        
        # Data validation errors
        if any(term in error_str for term in ['validation', 'invalid', 'null', 'none', 'empty']):
            return ErrorCategory.VALIDATION_ERROR
        
        # Data processing errors
        if any(term in error_str for term in ['json', 'parse', 'decode', 'format', 'unicode']):
            return ErrorCategory.DATA_ERROR
        
        # Processing errors
        if any(term in error_str for term in ['processing', 'chunk', 'deduplication', 'normalization']):
            return ErrorCategory.PROCESSING_ERROR
        
        # System errors
        if any(term in error_type for term in ['memory', 'system', 'os', 'io']):
            return ErrorCategory.SYSTEM_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "", 
                    recovery_function: Optional[Callable] = None,
                    fallback_value: Any = None) -> Dict[str, Any]:
        """
        Handle error with categorization, logging, and optional recovery
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            recovery_function: Optional function to attempt recovery
            fallback_value: Fallback value to return if recovery fails
            
        Returns:
            Dictionary with error information and recovery status
        """
        self.error_stats["total_errors"] += 1
        category = self.categorize_error(error)
        
        # Update category statistics
        if category.value not in self.error_stats["errors_by_category"]:
            self.error_stats["errors_by_category"][category.value] = 0
        self.error_stats["errors_by_category"][category.value] += 1
        
        # Log error with context
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.value,
            "context": context,
            "timestamp": time.time(),
            "recovery_attempted": False,
            "recovery_successful": False,
            "fallback_used": False
        }
        
        self.logger.error(f"Error in {context}: {error_info['error_type']} - {error_info['error_message']}")
        
        # Attempt recovery if function provided
        if recovery_function:
            try:
                self.error_stats["recovery_attempts"] += 1
                error_info["recovery_attempted"] = True
                
                recovery_result = recovery_function()
                if recovery_result is not None:
                    self.error_stats["successful_recoveries"] += 1
                    error_info["recovery_successful"] = True
                    error_info["recovery_result"] = recovery_result
                    self.logger.info(f"Recovery successful for {context}")
                    return error_info
                    
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {context}: {recovery_error}")
                error_info["recovery_error"] = str(recovery_error)
        
        # Use fallback value if provided
        if fallback_value is not None:
            error_info["fallback_used"] = True
            error_info["fallback_value"] = fallback_value
            self.logger.info(f"Using fallback value for {context}")
        else:
            self.error_stats["critical_failures"] += 1
            self.logger.critical(f"Critical failure in {context} - no recovery or fallback available")
        
        return error_info
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = self.error_stats["total_errors"]
        if total_errors > 0:
            recovery_rate = (self.error_stats["successful_recoveries"] / self.error_stats["recovery_attempts"]) * 100 if self.error_stats["recovery_attempts"] > 0 else 0
            critical_rate = (self.error_stats["critical_failures"] / total_errors) * 100
        else:
            recovery_rate = 0
            critical_rate = 0
        
        return {
            **self.error_stats,
            "recovery_rate_percent": round(recovery_rate, 1),
            "critical_failure_rate_percent": round(critical_rate, 1)
        }


def safe_execute(func: Callable, 
                error_handler: ErrorHandler,
                context: str = "",
                recovery_function: Optional[Callable] = None,
                fallback_value: Any = None,
                *args, **kwargs) -> Any:
    """
    Safely execute a function with comprehensive error handling
    
    Args:
        func: Function to execute
        error_handler: ErrorHandler instance
        context: Context description
        recovery_function: Optional recovery function
        fallback_value: Fallback value if execution fails
        *args, **kwargs: Arguments for the function
        
    Returns:
        Function result or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_info = error_handler.handle_error(e, context, recovery_function, fallback_value)
        
        if error_info.get("recovery_successful"):
            return error_info.get("recovery_result")
        elif error_info.get("fallback_used"):
            return error_info.get("fallback_value")
        else:
            # Re-raise if no recovery or fallback
            raise


def resilient_function(error_handler: ErrorHandler, 
                      context: str = "",
                      recovery_function: Optional[Callable] = None,
                      fallback_value: Any = None):
    """
    Decorator to make functions resilient with automatic error handling
    
    Args:
        error_handler: ErrorHandler instance
        context: Context description
        recovery_function: Optional recovery function
        fallback_value: Fallback value if execution fails
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return safe_execute(
                func, error_handler, context or func.__name__, 
                recovery_function, fallback_value, *args, **kwargs
            )
        return wrapper
    return decorator


class DataValidator:
    """
    Comprehensive data validator with error recovery for document processing
    """
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    def validate_entry(self, entry: Any, index: int = None) -> Dict[str, Any]:
        """
        Validate a single entry with comprehensive error handling
        
        Args:
            entry: Entry to validate
            index: Optional index for error reporting
            
        Returns:
            Validation result with cleaned entry or error information
        """
        context = f"entry_validation_{index}" if index is not None else "entry_validation"
        
        try:
            # Null value protection
            if entry is None:
                raise ValueError("Entry is None")
            
            if not isinstance(entry, dict):
                raise ValueError(f"Entry must be dict, got {type(entry)}")
            
            # Key validation
            key = entry.get('Key')
            if key is None:
                raise ValueError("Entry missing 'Key' field")
            
            if not isinstance(key, str):
                key = str(key) if key is not None else ""
            
            key = key.strip()
            if not key or len(key) < 1:
                raise ValueError("Entry has empty or invalid key")
            
            # Value validation
            value = entry.get('Value')
            if value is None:
                raise ValueError("Entry missing 'Value' field")
            
            if not isinstance(value, str):
                value = str(value) if value is not None else ""
            
            value = value.strip()
            if not value:
                raise ValueError("Entry has empty value")
            
            # Create validated entry
            validated_entry = {
                'Key': key,
                'Value': value,
                'Comments': str(entry.get('Comments', '')).strip()
            }
            
            # Preserve additional fields
            for field_name, field_value in entry.items():
                if field_name not in ['Key', 'Value', 'Comments']:
                    validated_entry[field_name] = field_value
            
            return {
                'valid': True,
                'entry': validated_entry,
                'errors': []
            }
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context, fallback_value={
                'valid': False,
                'entry': None,
                'errors': [str(e)]
            })
            
            return error_info.get('fallback_value', {
                'valid': False,
                'entry': None,
                'errors': [str(e)]
            })
    
    def validate_entries_list(self, entries: List[Any]) -> Dict[str, Any]:
        """
        Validate a list of entries with comprehensive error handling
        
        Args:
            entries: List of entries to validate
            
        Returns:
            Validation result with valid entries and error statistics
        """
        try:
            if not isinstance(entries, list):
                raise ValueError(f"Entries must be list, got {type(entries)}")
            
            valid_entries = []
            validation_errors = []
            
            for i, entry in enumerate(entries):
                validation_result = self.validate_entry(entry, i)
                
                if validation_result.get('valid'):
                    valid_entries.append(validation_result['entry'])
                else:
                    validation_errors.extend(validation_result.get('errors', []))
            
            return {
                'valid_entries': valid_entries,
                'total_input': len(entries),
                'total_valid': len(valid_entries),
                'total_invalid': len(entries) - len(valid_entries),
                'validation_errors': validation_errors,
                'success_rate': (len(valid_entries) / len(entries)) * 100 if entries else 0
            }
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, "entries_list_validation", fallback_value={
                'valid_entries': [],
                'total_input': 0,
                'total_valid': 0,
                'total_invalid': 0,
                'validation_errors': [str(e)],
                'success_rate': 0
            })
            
            return error_info.get('fallback_value')


# Global error handler instance
global_error_handler = ErrorHandler("DocumentProcessor")
global_data_validator = DataValidator(global_error_handler)


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    return global_error_handler


def get_data_validator() -> DataValidator:
    """Get the global data validator instance"""
    return global_data_validator


def print_error_summary():
    """Print a summary of all errors encountered"""
    stats = global_error_handler.get_error_stats()
    
    print("\n" + "="*50)
    print("📊 ERROR HANDLING SUMMARY")
    print("="*50)
    print(f"Total Errors: {stats['total_errors']}")
    print(f"Recovery Attempts: {stats['recovery_attempts']}")
    print(f"Successful Recoveries: {stats['successful_recoveries']}")
    print(f"Critical Failures: {stats['critical_failures']}")
    print(f"Recovery Rate: {stats['recovery_rate_percent']}%")
    print(f"Critical Failure Rate: {stats['critical_failure_rate_percent']}%")
    
    if stats['errors_by_category']:
        print("\nErrors by Category:")
        for category, count in stats['errors_by_category'].items():
            print(f"  {category}: {count}")
    
    print("="*50)