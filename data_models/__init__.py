"""
FinPilot Data Models Package

Comprehensive Pydantic schemas for multi-agent system communication.
"""

from data_models.schemas import (
    # Enums
    MarketEventType,
    SeverityLevel,
    LifeEventType,
    PlanStatus,
    RiskTolerance,

    # Market models
    MarketData,
    MarketContext,

    # Trigger models
    TriggerEvent,

    # Financial models
    TaxContext,
    RegulatoryRequirement,
    RiskProfile,
    FinancialState,
    Constraint,

    # Planning models
    PlanStep,
    PlanRequest,
    FinancialPlan,

    # Verification models
    VerificationReport,
    ComplianceStatus,

    # Communication models
    AgentMessage,
    ExecutionLog,

    # Reasoning models
    ReasoningTrace,
    SearchPath,

    # Metrics
    PerformanceMetrics,
)

__all__ = [
    # Enums
    'MarketEventType',
    'SeverityLevel',
    'LifeEventType',
    'PlanStatus',
    'RiskTolerance',

    # Market models
    'MarketData',
    'MarketContext',

    # Trigger models
    'TriggerEvent',

    # Financial models
    'TaxContext',
    'RegulatoryRequirement',
    'RiskProfile',
    'FinancialState',
    'Constraint',

    # Planning models
    'PlanStep',
    'PlanRequest',
    'FinancialPlan',

    # Verification models
    'VerificationReport',
    'ComplianceStatus',

    # Communication models
    'AgentMessage',
    'ExecutionLog',

    # Reasoning models
    'ReasoningTrace',
    'SearchPath',

    # Metrics
    'PerformanceMetrics',
]
