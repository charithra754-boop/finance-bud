"""
Application Settings

Centralized configuration using pydantic-settings.
All configuration loaded from environment variables or .env file.

NO HARDCODED VALUES - everything configurable.

Created: Phase 1 - Foundation & Safety Net

Usage:
    from backend.src.config import settings

    # Access configuration
    print(settings.api_host)
    print(settings.database_url)
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables can be prefixed with FINPILOT_
    Example: FINPILOT_API_HOST=0.0.0.0
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FINPILOT_",
        case_sensitive=False,
        extra="ignore"  # Ignore unknown env vars
    )

    # =================================================================
    # API CONFIGURATION
    # =================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )

    api_port: int = Field(
        default=8000,
        description="API server port",
        ge=1024,
        le=65535
    )

    api_title: str = Field(
        default="FinPilot VP-MAS API",
        description="API title for OpenAPI docs"
    )

    api_version: str = Field(
        default="1.0.0",
        description="API version"
    )

    api_debug: bool = Field(
        default=False,
        description="Enable debug mode (DO NOT use in production)"
    )

    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="Allowed CORS origins"
    )

    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )

    # =================================================================
    # AGENT CONFIGURATION
    # =================================================================
    agents_enabled: List[str] = Field(
        default=[
            "orchestration",
            "planning",
            "information_retrieval",
            "verification",
            "execution",
            "conversational"
        ],
        description="List of agents to enable"
    )

    agent_timeout_seconds: float = Field(
        default=30.0,
        description="Default agent operation timeout",
        ge=1.0,
        le=300.0
    )

    agent_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for agent operations",
        ge=0,
        le=10
    )

    # =================================================================
    # LLM CONFIGURATION
    # =================================================================
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider (ollama|openai|local)"
    )

    ollama_enabled: bool = Field(
        default=True,
        description="Enable Ollama for local LLM"
    )

    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )

    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model to use"
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (if using OpenAI)"
    )

    openai_model: str = Field(
        default="gpt-4",
        description="OpenAI model to use"
    )

    # =================================================================
    # DATABASE CONFIGURATION
    # =================================================================
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL"
    )

    database_pool_size: int = Field(
        default=10,
        description="Database connection pool size",
        ge=1,
        le=100
    )

    database_max_overflow: int = Field(
        default=20,
        description="Database max overflow connections",
        ge=0,
        le=100
    )

    # =================================================================
    # REDIS / CACHE CONFIGURATION
    # =================================================================
    redis_enabled: bool = Field(
        default=False,
        description="Enable Redis for caching"
    )

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    cache_ttl_seconds: int = Field(
        default=300,
        description="Default cache TTL in seconds",
        ge=1
    )

    # =================================================================
    # EXTERNAL API CONFIGURATION
    # =================================================================
    barchart_api_key: Optional[str] = Field(
        default=None,
        description="Barchart API key for market data"
    )

    alpha_vantage_api_key: Optional[str] = Field(
        default=None,
        description="Alpha Vantage API key"
    )

    market_data_provider: str = Field(
        default="mock",
        description="Market data provider (mock|barchart|alpha_vantage|yfinance)"
    )

    use_mock_apis: bool = Field(
        default=True,
        description="Use mock APIs when real APIs unavailable"
    )

    # =================================================================
    # MESSAGE BUS / COMMUNICATION
    # =================================================================
    message_bus_type: str = Field(
        default="in_memory",
        description="Message bus implementation (in_memory|redis|rabbitmq)"
    )

    message_queue_size: int = Field(
        default=10000,
        description="Maximum message queue size",
        ge=100,
        le=1000000
    )

    message_retention_seconds: int = Field(
        default=3600,
        description="How long to retain message history",
        ge=60
    )

    correlation_tracker_max_size: int = Field(
        default=10000,
        description="Maximum correlation tracker entries before cleanup",
        ge=100,
        le=1000000
    )

    correlation_tracker_ttl_seconds: int = Field(
        default=3600,
        description="TTL for correlation tracker entries",
        ge=60
    )

    # =================================================================
    # CIRCUIT BREAKER CONFIGURATION
    # =================================================================
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )

    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Failures before opening circuit",
        ge=1,
        le=100
    )

    circuit_breaker_timeout_seconds: int = Field(
        default=60,
        description="Circuit breaker timeout",
        ge=1,
        le=300
    )

    circuit_breaker_half_open_max_calls: int = Field(
        default=3,
        description="Max calls in half-open state",
        ge=1,
        le=10
    )

    # =================================================================
    # LOGGING CONFIGURATION
    # =================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL)"
    )

    log_format: str = Field(
        default="json",
        description="Log format (json|text)"
    )

    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (None for stdout only)"
    )

    structured_logging: bool = Field(
        default=True,
        description="Use structured logging with correlation IDs"
    )

    # =================================================================
    # SECURITY CONFIGURATION
    # =================================================================
    secret_key: str = Field(
        default="dev-secret-key-CHANGE-IN-PRODUCTION",
        description="Secret key for JWT and encryption"
    )

    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )

    jwt_expiration_minutes: int = Field(
        default=60,
        description="JWT token expiration in minutes",
        ge=1,
        le=10080  # 1 week
    )

    # =================================================================
    # FEATURE FLAGS
    # =================================================================
    feature_new_planning_agent: bool = Field(
        default=False,
        description="Use new refactored planning agent"
    )

    feature_clean_architecture: bool = Field(
        default=False,
        description="Enable clean architecture implementation"
    )

    feature_dependency_injection: bool = Field(
        default=False,
        description="Use dependency injection container"
    )

    feature_new_message_bus: bool = Field(
        default=False,
        description="Use new message bus implementation"
    )

    # =================================================================
    # MONITORING / METRICS
    # =================================================================
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )

    metrics_port: int = Field(
        default=9090,
        description="Prometheus metrics port",
        ge=1024,
        le=65535
    )

    tracing_enabled: bool = Field(
        default=False,
        description="Enable distributed tracing (OpenTelemetry)"
    )

    tracing_endpoint: Optional[str] = Field(
        default=None,
        description="Tracing collector endpoint"
    )

    # =================================================================
    # PERFORMANCE / OPTIMIZATION
    # =================================================================
    max_concurrent_workflows: int = Field(
        default=100,
        description="Maximum concurrent workflows",
        ge=1,
        le=10000
    )

    planning_timeout_seconds: int = Field(
        default=120,
        description="Maximum time for planning operations",
        ge=10,
        le=600
    )

    verification_timeout_seconds: int = Field(
        default=30,
        description="Maximum time for verification",
        ge=5,
        le=300
    )

    # =================================================================
    # CMVL (Continuous Monitoring) CONFIGURATION
    # =================================================================
    cmvl_enabled: bool = Field(
        default=True,
        description="Enable CMVL workflow"
    )

    cmvl_check_interval_seconds: int = Field(
        default=60,
        description="CMVL monitoring interval",
        ge=10,
        le=3600
    )

    cmvl_volatility_threshold: float = Field(
        default=0.25,
        description="Volatility threshold for CMVL triggers",
        ge=0.0,
        le=1.0
    )

    # =================================================================
    # TESTING CONFIGURATION
    # =================================================================
    testing_mode: bool = Field(
        default=False,
        description="Enable testing mode (uses mocks, faster timeouts)"
    )

    mock_mode: bool = Field(
        default=False,
        description="Use mock agents for testing"
    )

    # =================================================================
    # METHODS
    # =================================================================
    def is_production(self) -> bool:
        """Check if running in production"""
        return not self.api_debug and not self.testing_mode

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.api_debug or self.testing_mode

    def get_database_url(self) -> str:
        """Get database URL with fallback"""
        if self.database_url:
            return self.database_url
        # Default SQLite for development
        return "sqlite:///./finpilot.db"

    def get_cors_config(self) -> dict:
        """Get CORS configuration"""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_allow_credentials,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }


# Global settings instance
# Import this in your code: from backend.src.config import settings
settings = Settings()
