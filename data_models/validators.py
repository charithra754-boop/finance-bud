"""
FinPilot Multi-Agent System - Data Validation Utilities

This module provides validation utilities and custom validators for the Pydantic data models.
Includes financial calculations, constraint checking, and data quality validation.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from pydantic import validator


class FinancialValidators:
    """Collection of financial validation utilities"""
    
    @staticmethod
    def validate_positive_amount(value: Decimal) -> Decimal:
        """Validate that a financial amount is positive"""
        if value < 0:
            raise ValueError("Amount must be positive")
        return value
    
    @staticmethod
    def validate_percentage(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate that a value is within percentage bounds"""
        if not min_val <= value <= max_val:
            raise ValueError(f"Value must be between {min_val} and {max_val}")
        return value
    
    @staticmethod
    def validate_risk_score(value: float) -> float:
        """Validate risk score is between 0 and 1"""
        return FinancialValidators.validate_percentage(value, 0.0, 1.0)
    
    @staticmethod
    def validate_confidence_score(value: float) -> float:
        """Validate confidence score is between 0 and 1"""
        return FinancialValidators.validate_percentage(value, 0.0, 1.0)


class TimeValidators:
    """Collection of time-related validation utilities"""
    
    @staticmethod
    def validate_future_date(value: datetime) -> datetime:
        """Validate that a date is in the future"""
        if value <= datetime.utcnow():
            raise ValueError("Date must be in the future")
        return value
    
    @staticmethod
    def validate_reasonable_horizon(months: int) -> int:
        """Validate that time horizon is reasonable (1 month to 50 years)"""
        if not 1 <= months <= 600:
            raise ValueError("Time horizon must be between 1 and 600 months")
        return months


class DataQualityValidators:
    """Collection of data quality validation utilities"""
    
    @staticmethod
    def validate_correlation_id_format(value: str) -> str:
        """Validate correlation ID format"""
        if not value or len(value) < 10:
            raise ValueError("Correlation ID must be at least 10 characters")
        return value
    
    @staticmethod
    def validate_agent_id_format(value: str) -> str:
        """Validate agent ID format"""
        if not value or not value.endswith('_agent') and not value.endswith('_001'):
            raise ValueError("Agent ID must follow naming convention")
        return value
    
    @staticmethod
    def validate_non_empty_dict(value: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that dictionary is not empty when required"""
        if not value:
            raise ValueError("Dictionary cannot be empty")
        return value


class ComplianceValidators:
    """Collection of compliance-related validation utilities"""
    
    @staticmethod
    def validate_tax_rate(value: float) -> float:
        """Validate tax rate is reasonable (0% to 50%)"""
        if not 0.0 <= value <= 0.5:
            raise ValueError("Tax rate must be between 0% and 50%")
        return value
    
    @staticmethod
    def validate_regulatory_requirement(requirement: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regulatory requirement structure"""
        required_fields = ['regulation_name', 'description', 'compliance_level']
        for field in required_fields:
            if field not in requirement:
                raise ValueError(f"Regulatory requirement missing required field: {field}")
        return requirement


# Custom validator functions for use in Pydantic models
def validate_financial_amount(cls, v):
    """Pydantic validator for financial amounts"""
    return FinancialValidators.validate_positive_amount(v)

def validate_risk_percentage(cls, v):
    """Pydantic validator for risk percentages"""
    return FinancialValidators.validate_risk_score(v)

def validate_confidence_percentage(cls, v):
    """Pydantic validator for confidence scores"""
    return FinancialValidators.validate_confidence_score(v)

def validate_future_datetime(cls, v):
    """Pydantic validator for future dates"""
    return TimeValidators.validate_future_date(v)

def validate_time_horizon_months(cls, v):
    """Pydantic validator for time horizons"""
    return TimeValidators.validate_reasonable_horizon(v)