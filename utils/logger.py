"""
FinPilot Multi-Agent System - Structured Logging Utilities

Provides comprehensive logging infrastructure with:
- Correlation ID tracking
- Performance metrics
- Structured JSON logging
- Agent-specific context
- Audit trail support

Requirements: 9.4, 10.1, 40.3
"""

import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import contextvars

# Context variable for correlation ID tracking
correlation_id_var = contextvars.ContextVar('correlation_id', default=None)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

DEFAULT_LOG_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(correlation_id)s | %(message)s'
JSON_LOG_FORMAT = True  # Set to False for plain text logs


# ============================================================================
# CUSTOM FORMATTER
# ============================================================================

class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""

    def filter(self, record):
        record.correlation_id = correlation_id_var.get() or 'N/A'
        return True


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for logs.

    Outputs logs as JSON for easy parsing and analysis.
    """

    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'correlation_id': getattr(record, 'correlation_id', 'N/A'),
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data

        # Add performance metrics if present
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms

        return json.dumps(log_data)


# ============================================================================
# LOGGER SETUP
# ============================================================================

def setup_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    use_json: bool = True
) -> logging.Logger:
    """
    Set up a logger with structured logging support.

    Args:
        name: Logger name (typically agent name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        use_json: Use JSON formatting (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level, logging.INFO))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    logger.addFilter(correlation_filter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVELS.get(level, logging.INFO))

    if use_json:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVELS.get(level, logging.INFO))

        if use_json:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

        logger.addHandler(file_handler)

    return logger


# ============================================================================
# AGENT LOGGER
# ============================================================================

class AgentLogger:
    """
    Enhanced logger for multi-agent system with correlation tracking.

    Provides structured logging with:
    - Automatic correlation ID tracking
    - Performance metrics
    - Agent context
    - Audit trail support
    """

    def __init__(self, agent_name: str, level: str = 'INFO', log_file: Optional[str] = None):
        """
        Initialize agent logger.

        Args:
            agent_name: Name of the agent
            level: Log level
            log_file: Optional log file path
        """
        self.agent_name = agent_name
        self.logger = setup_logger(
            name=f'FinPilot.{agent_name}',
            level=level,
            log_file=log_file
        )

    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current context"""
        correlation_id_var.set(correlation_id)

    def clear_correlation_id(self):
        """Clear correlation ID"""
        correlation_id_var.set(None)

    def _log(self, level: str, message: str, extra_data: Optional[Dict[str, Any]] = None,
             duration_ms: Optional[float] = None):
        """Internal log method with extra data support"""
        log_method = getattr(self.logger, level.lower())

        # Create log record with extra data
        if extra_data or duration_ms:
            extra = {
                'extra_data': extra_data or {},
            }
            if duration_ms is not None:
                extra['duration_ms'] = duration_ms

            log_method(message, extra=extra)
        else:
            log_method(message)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log('DEBUG', message, kwargs.get('extra_data'), kwargs.get('duration_ms'))

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log('INFO', message, kwargs.get('extra_data'), kwargs.get('duration_ms'))

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log('WARNING', message, kwargs.get('extra_data'), kwargs.get('duration_ms'))

    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log('ERROR', message, kwargs.get('extra_data'), kwargs.get('duration_ms'))

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log('CRITICAL', message, kwargs.get('extra_data'), kwargs.get('duration_ms'))

    def log_api_call(self, provider: str, endpoint: str, duration_ms: float,
                     success: bool, status_code: Optional[int] = None):
        """
        Log external API call.

        Args:
            provider: API provider name
            endpoint: API endpoint called
            duration_ms: Call duration in milliseconds
            success: Whether call succeeded
            status_code: HTTP status code if applicable
        """
        extra_data = {
            'api_provider': provider,
            'endpoint': endpoint,
            'success': success,
            'status_code': status_code
        }

        level = 'INFO' if success else 'ERROR'
        message = f"API call to {provider}/{endpoint} {'succeeded' if success else 'failed'}"

        self._log(level, message, extra_data, duration_ms)

    def log_agent_communication(self, from_agent: str, to_agent: str,
                               message_type: str, success: bool = True):
        """
        Log inter-agent communication.

        Args:
            from_agent: Sending agent
            to_agent: Receiving agent
            message_type: Type of message
            success: Whether communication succeeded
        """
        extra_data = {
            'from_agent': from_agent,
            'to_agent': to_agent,
            'message_type': message_type,
            'success': success
        }

        message = f"Agent communication: {from_agent} -> {to_agent} ({message_type})"
        self._log('INFO' if success else 'ERROR', message, extra_data)

    def log_performance(self, operation: str, duration_ms: float, metadata: Optional[Dict] = None):
        """
        Log performance metrics.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            metadata: Additional metadata
        """
        extra_data = metadata or {}
        extra_data['operation'] = operation

        message = f"Performance: {operation} completed in {duration_ms:.2f}ms"
        self._log('INFO', message, extra_data, duration_ms)

    def log_trigger_event(self, event_type: str, severity: str, description: str):
        """
        Log trigger event detection.

        Args:
            event_type: Type of trigger event
            severity: Event severity
            description: Event description
        """
        extra_data = {
            'event_type': event_type,
            'severity': severity,
            'description': description
        }

        message = f"Trigger detected: {event_type} (severity: {severity})"
        self._log('WARNING' if severity in ['HIGH', 'CRITICAL'] else 'INFO',
                 message, extra_data)

    def log_plan_generation(self, plan_id: str, goal: str, steps_count: int,
                           success_probability: float):
        """
        Log plan generation.

        Args:
            plan_id: Generated plan ID
            goal: Plan goal
            steps_count: Number of steps in plan
            success_probability: Success probability
        """
        extra_data = {
            'plan_id': plan_id,
            'goal': goal,
            'steps_count': steps_count,
            'success_probability': success_probability
        }

        message = f"Plan generated: {plan_id} with {steps_count} steps"
        self._log('INFO', message, extra_data)

    def log_verification(self, plan_id: str, approved: bool, confidence: float,
                        violations: Optional[list] = None):
        """
        Log plan verification result.

        Args:
            plan_id: Plan being verified
            approved: Whether plan approved
            confidence: Verification confidence
            violations: List of violations if any
        """
        extra_data = {
            'plan_id': plan_id,
            'approved': approved,
            'confidence': confidence,
            'violations': violations or []
        }

        message = f"Plan verification: {plan_id} {'APPROVED' if approved else 'REJECTED'}"
        self._log('INFO' if approved else 'WARNING', message, extra_data)

    def log_audit(self, action: str, user_id: str, details: Dict[str, Any]):
        """
        Log audit trail entry.

        Args:
            action: Action performed
            user_id: User performing action
            details: Action details
        """
        extra_data = {
            'action': action,
            'user_id': user_id,
            'details': details,
            'audit': True
        }

        message = f"AUDIT: {action} by user {user_id}"
        self._log('INFO', message, extra_data)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_logger(agent_name: str, level: str = 'INFO', log_file: Optional[str] = None) -> AgentLogger:
    """
    Get or create an agent logger.

    Args:
        agent_name: Name of the agent
        level: Log level
        log_file: Optional log file path

    Returns:
        AgentLogger instance
    """
    return AgentLogger(agent_name, level, log_file)


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class LoggingStandards:
    """
    Backward compatibility class for existing code.

    Provides create_system_logger method used by existing modules.
    """

    @staticmethod
    def create_system_logger(name: str, level: str = 'INFO') -> AgentLogger:
        """
        Create a system logger with standard configuration.

        Args:
            name: Logger name
            level: Log level

        Returns:
            AgentLogger instance
        """
        return get_logger(name, level=level)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'AgentLogger',
    'get_logger',
    'setup_logger',
    'correlation_id_var',
    'LoggingStandards'  # For backward compatibility
]
