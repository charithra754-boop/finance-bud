"""
FinPilot Multi-Agent System - Comprehensive Data Schemas

This module defines all Pydantic data contracts shared across agents.
These schemas ensure type safety and validation for inter-agent communication.

Requirements: 6.1, 6.2, 9.1, 9.3, 28.1, 28.4, 28.5
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, ConfigDict


# ============================================================================
# ENUMS - Standard types across the system
# ============================================================================

class MarketEventType(str, Enum):
    """Types of market events that can trigger replanning"""
    CRASH = "crash"
    RECOVERY = "recovery"
    VOLATILITY_SPIKE = "volatility_spike"
    SECTOR_ROTATION = "sector_rotation"
    REGULATORY_CHANGE = "regulatory_change"
    INTEREST_RATE_CHANGE = "interest_rate_change"
    ECONOMIC_INDICATOR = "economic_indicator"


class SeverityLevel(str, Enum):
    """Severity assessment for triggers and events"""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"          # Urgent attention needed
    MEDIUM = "medium"      # Should be addressed soon
    LOW = "low"            # Informational/monitoring


class LifeEventType(str, Enum):
    """Types of life events that can trigger replanning"""
    JOB_LOSS = "job_loss"
    JOB_CHANGE = "job_change"
    MARRIAGE = "marriage"
    DIVORCE = "divorce"
    CHILD_BIRTH = "child_birth"
    FAMILY_EMERGENCY = "family_emergency"
    HEALTH_ISSUE = "health_issue"
    INHERITANCE = "inheritance"
    MAJOR_PURCHASE = "major_purchase"
    RETIREMENT = "retirement"


class PlanStatus(str, Enum):
    """Status of a financial plan"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class RiskTolerance(str, Enum):
    """User's risk tolerance level"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


# ============================================================================
# MARKET DATA MODELS
# ============================================================================

class MarketData(BaseModel):
    """
    Comprehensive market data from multiple sources.

    Used by IRA to provide enriched market context to other agents.
    Includes real-time prices, volatility indicators, and predictive analytics.
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "NIFTY50",
            "price": 19500.50,
            "change_percent": -2.3,
            "volume": 1250000,
            "volatility": 18.5,
            "timestamp": "2025-11-17T10:30:00Z"
        }
    })

    symbol: str = Field(..., description="Stock/index symbol")
    price: float = Field(..., description="Current price")
    change_percent: float = Field(..., description="Percentage change from previous close")
    volume: Optional[int] = Field(None, description="Trading volume")
    volatility: Optional[float] = Field(None, description="Volatility indicator (VIX-style)")

    # Market context
    market_sentiment: Optional[str] = Field(None, description="Bullish/Bearish/Neutral")
    sector_trend: Optional[str] = Field(None, description="Sector performance trend")
    correlation_indices: Optional[Dict[str, float]] = Field(None, description="Correlation with other indices")

    # Predictive indicators
    predicted_volatility: Optional[float] = Field(None, description="Predicted volatility next period")
    risk_score: Optional[float] = Field(None, description="Risk score 0-100")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")
    source: str = Field("unknown", description="Data source (barchart/alphavantage/massive)")

    @validator('volatility', 'predicted_volatility')
    def validate_volatility(cls, v):
        if v is not None and (v < 0 or v > 200):
            raise ValueError('Volatility must be between 0 and 200')
        return v


class MarketContext(BaseModel):
    """
    Enriched market context with multiple data points.

    Returned by IRA's get_market_context() function.
    Provides comprehensive view of market conditions for planning.
    """
    # Core indicators
    market_volatility: float = Field(..., description="Overall market volatility index")
    interest_rate: float = Field(..., description="Current benchmark interest rate")
    inflation_rate: Optional[float] = Field(None, description="Current inflation rate")

    # Market state
    economic_sentiment: str = Field(..., description="Overall economic sentiment")
    sector_trends: Dict[str, str] = Field(default_factory=dict, description="Trends by sector")

    # Specific asset data
    indices: Dict[str, MarketData] = Field(default_factory=dict, description="Major index data")
    commodities: Optional[Dict[str, MarketData]] = Field(None, description="Commodity prices")

    # Regulatory and news
    regulatory_changes: List[str] = Field(default_factory=list, description="Recent regulatory updates")
    major_events: List[str] = Field(default_factory=list, description="Major market events")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(0.95, description="Data confidence score 0-1")

    @validator('market_volatility')
    def validate_market_volatility(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Market volatility must be between 0 and 100')
        return v


# ============================================================================
# TRIGGER EVENT MODELS
# ============================================================================

class TriggerEvent(BaseModel):
    """
    Event that triggers CMVL (Continuous Monitoring and Verification Loop).

    Can be market events, life events, or scheduled reviews.
    """
    event_id: str = Field(..., description="Unique event identifier")
    event_type: Union[MarketEventType, LifeEventType, str] = Field(..., description="Type of trigger event")
    severity: SeverityLevel = Field(..., description="Event severity level")

    # Event details
    description: str = Field(..., description="Human-readable event description")
    impact_assessment: Optional[str] = Field(None, description="Expected impact on plan")
    confidence: float = Field(0.9, description="Detection confidence 0-1")

    # Market-specific fields
    market_data: Optional[MarketData] = Field(None, description="Related market data if applicable")
    affected_assets: List[str] = Field(default_factory=list, description="Assets affected by event")

    # Life event-specific fields
    financial_impact: Optional[float] = Field(None, description="Estimated financial impact (positive/negative)")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = Field(..., description="Source of trigger detection")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")

    # Action recommendations
    recommended_actions: List[str] = Field(default_factory=list, description="Suggested actions")
    requires_immediate_action: bool = Field(False, description="Requires immediate replanning")


# ============================================================================
# FINANCIAL STATE MODELS
# ============================================================================

class TaxContext(BaseModel):
    """Tax context for financial planning"""
    tax_bracket: str = Field(..., description="Current tax bracket")
    annual_income: float = Field(..., description="Annual taxable income")
    tax_deductions: Dict[str, float] = Field(default_factory=dict, description="Available deductions")
    tax_credits: Dict[str, float] = Field(default_factory=dict, description="Available tax credits")
    tax_jurisdiction: str = Field("India", description="Tax jurisdiction")

    # Tax optimization
    tax_saving_instruments: List[str] = Field(default_factory=list, description="80C, 80D, etc.")
    estimated_tax_liability: Optional[float] = Field(None, description="Estimated annual tax")


class RegulatoryRequirement(BaseModel):
    """Regulatory compliance requirement"""
    requirement_id: str = Field(..., description="Requirement identifier")
    regulation_name: str = Field(..., description="Name of regulation")
    jurisdiction: str = Field(..., description="Applicable jurisdiction")
    compliance_status: str = Field("unknown", description="Current compliance status")
    deadline: Optional[datetime] = Field(None, description="Compliance deadline")
    description: str = Field(..., description="Requirement description")


class RiskProfile(BaseModel):
    """User's risk profile and preferences"""
    risk_tolerance: RiskTolerance = Field(..., description="Overall risk tolerance")
    investment_horizon: int = Field(..., description="Investment horizon in years")

    # Risk preferences
    max_drawdown_tolerance: float = Field(20.0, description="Maximum acceptable drawdown %")
    volatility_tolerance: float = Field(15.0, description="Acceptable volatility level")

    # Preferences
    ethical_investing: bool = Field(False, description="Prefer ESG investments")
    sector_preferences: List[str] = Field(default_factory=list, description="Preferred sectors")
    sector_exclusions: List[str] = Field(default_factory=list, description="Excluded sectors")


class FinancialState(BaseModel):
    """
    Complete financial state of the user.

    Used by all agents to understand user's current financial position.
    """
    user_id: str = Field(..., description="User identifier")

    # Assets and liabilities
    total_assets: float = Field(..., description="Total asset value")
    total_liabilities: float = Field(0.0, description="Total liabilities")
    net_worth: float = Field(..., description="Net worth (assets - liabilities)")

    # Income and expenses
    monthly_income: float = Field(..., description="Monthly income")
    monthly_expenses: float = Field(..., description="Monthly expenses")
    savings_rate: float = Field(..., description="Savings rate as percentage")

    # Portfolio breakdown
    portfolio: Dict[str, float] = Field(default_factory=dict, description="Asset allocation by type")
    emergency_fund: float = Field(0.0, description="Emergency fund balance")

    # Context
    risk_profile: RiskProfile = Field(..., description="User's risk profile")
    tax_context: TaxContext = Field(..., description="Tax context")

    # Goals
    financial_goals: List[str] = Field(default_factory=list, description="User's financial goals")
    goal_timeline: Optional[Dict[str, int]] = Field(None, description="Timeline for goals in years")

    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)

    @validator('net_worth')
    def calculate_net_worth(cls, v, values):
        if 'total_assets' in values and 'total_liabilities' in values:
            calculated = values['total_assets'] - values['total_liabilities']
            if abs(calculated - v) > 0.01:  # Allow small floating point differences
                return calculated
        return v


class Constraint(BaseModel):
    """Financial constraint for planning"""
    constraint_id: str = Field(..., description="Constraint identifier")
    constraint_type: str = Field(..., description="Type: budget/regulatory/risk/timeline")
    description: str = Field(..., description="Constraint description")

    # Constraint parameters
    max_value: Optional[float] = Field(None, description="Maximum allowed value")
    min_value: Optional[float] = Field(None, description="Minimum required value")

    # Compliance
    is_hard_constraint: bool = Field(True, description="Hard vs soft constraint")
    priority: int = Field(1, description="Priority level 1-10")

    # Metadata
    source: str = Field("user", description="Source of constraint")
    active: bool = Field(True, description="Is constraint active")


# ============================================================================
# PLANNING MODELS
# ============================================================================

class PlanStep(BaseModel):
    """Individual step in a financial plan"""
    step_id: str = Field(..., description="Step identifier")
    sequence_number: int = Field(..., description="Order in plan")

    # Action details
    action: str = Field(..., description="Action to take")
    description: str = Field(..., description="Detailed description")
    amount: Optional[float] = Field(None, description="Amount involved")

    # Timing
    execute_at: Optional[datetime] = Field(None, description="Execution date")
    duration_months: Optional[int] = Field(None, description="Duration in months")

    # Impact
    expected_return: Optional[float] = Field(None, description="Expected return %")
    risk_level: Optional[str] = Field(None, description="Risk level")
    tax_implication: Optional[str] = Field(None, description="Tax implications")

    # Validation
    constraints_satisfied: List[str] = Field(default_factory=list, description="Satisfied constraints")
    regulatory_compliant: bool = Field(True, description="Meets regulatory requirements")

    # Metadata
    rationale: Optional[str] = Field(None, description="Reasoning for this step")
    alternatives_considered: List[str] = Field(default_factory=list, description="Alternative approaches")


class PlanRequest(BaseModel):
    """Request to create or update a financial plan"""
    request_id: str = Field(..., description="Request identifier")
    correlation_id: str = Field(..., description="Correlation ID for tracking")

    # User context
    user_id: str = Field(..., description="User identifier")
    financial_state: FinancialState = Field(..., description="Current financial state")

    # Planning parameters
    goal: str = Field(..., description="Primary goal")
    constraints: List[Constraint] = Field(default_factory=list, description="Planning constraints")
    time_horizon: int = Field(..., description="Planning horizon in years")

    # Trigger context (if replanning)
    trigger_event: Optional[TriggerEvent] = Field(None, description="Event triggering replanning")
    previous_plan_id: Optional[str] = Field(None, description="Previous plan if updating")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: str = Field("normal", description="Request priority")


class FinancialPlan(BaseModel):
    """Complete financial plan generated by Planning Agent"""
    plan_id: str = Field(..., description="Plan identifier")
    correlation_id: str = Field(..., description="Correlation ID")

    # Plan details
    goal: str = Field(..., description="Plan goal")
    steps: List[PlanStep] = Field(..., description="Ordered plan steps")

    # Performance metrics
    expected_final_value: float = Field(..., description="Expected portfolio value at end")
    probability_of_success: float = Field(..., description="Success probability 0-1")
    risk_adjusted_return: float = Field(..., description="Risk-adjusted return metric")

    # Compliance
    constraints_satisfied: List[str] = Field(default_factory=list, description="Satisfied constraints")
    regulatory_compliance: bool = Field(True, description="Meets all regulations")

    # Tax optimization
    tax_efficiency_score: Optional[float] = Field(None, description="Tax efficiency 0-100")
    estimated_tax_impact: Optional[float] = Field(None, description="Estimated tax impact")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field("PlanningAgent", description="Agent that created plan")
    status: PlanStatus = Field(PlanStatus.PENDING, description="Plan status")

    # Reasoning trace
    reasoning_summary: Optional[str] = Field(None, description="Summary of reasoning")
    alternatives_count: int = Field(0, description="Number of alternatives explored")


# ============================================================================
# VERIFICATION MODELS
# ============================================================================

class VerificationReport(BaseModel):
    """Verification report from Verification Agent"""
    report_id: str = Field(..., description="Report identifier")
    correlation_id: str = Field(..., description="Correlation ID")
    plan_id: str = Field(..., description="Plan being verified")

    # Verification results
    approved: bool = Field(..., description="Is plan approved")
    confidence_score: float = Field(..., description="Verification confidence 0-1")

    # Detailed findings
    constraint_violations: List[str] = Field(default_factory=list, description="Constraint violations found")
    risk_warnings: List[str] = Field(default_factory=list, description="Risk warnings")
    regulatory_issues: List[str] = Field(default_factory=list, description="Regulatory compliance issues")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")
    rejection_reason: Optional[str] = Field(None, description="Reason if rejected")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    verified_by: str = Field("VerificationAgent", description="Verifying agent")


class ComplianceStatus(BaseModel):
    """Compliance status for regulatory requirements"""
    requirement_id: str = Field(..., description="Requirement identifier")
    compliant: bool = Field(..., description="Is compliant")
    compliance_score: float = Field(..., description="Compliance score 0-1")
    issues: List[str] = Field(default_factory=list, description="Compliance issues")
    last_checked: datetime = Field(default_factory=datetime.now)


# ============================================================================
# AGENT COMMUNICATION MODELS
# ============================================================================

class AgentMessage(BaseModel):
    """Message passed between agents"""
    message_id: str = Field(..., description="Message identifier")
    correlation_id: str = Field(..., description="Correlation ID for request tracking")

    # Routing
    from_agent: str = Field(..., description="Sending agent")
    to_agent: str = Field(..., description="Receiving agent")
    message_type: str = Field(..., description="Message type")

    # Payload
    payload: Dict[str, Any] = Field(..., description="Message payload")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: str = Field("normal", description="Message priority")
    requires_response: bool = Field(False, description="Requires acknowledgment")


class ExecutionLog(BaseModel):
    """Log entry for plan execution"""
    log_id: str = Field(..., description="Log identifier")
    plan_id: str = Field(..., description="Plan being executed")
    step_id: str = Field(..., description="Step being executed")

    # Execution details
    action_taken: str = Field(..., description="Action executed")
    amount: Optional[float] = Field(None, description="Amount transacted")
    status: str = Field(..., description="Execution status")

    # Results
    success: bool = Field(..., description="Was execution successful")
    error_message: Optional[str] = Field(None, description="Error if failed")

    # Audit trail
    executed_at: datetime = Field(default_factory=datetime.now)
    executed_by: str = Field("ExecutionAgent", description="Executing agent")
    transaction_id: Optional[str] = Field(None, description="Transaction ID if applicable")


# ============================================================================
# REASONING & VISUALIZATION MODELS
# ============================================================================

class ReasoningTrace(BaseModel):
    """Trace of reasoning steps for visualization"""
    trace_id: str = Field(..., description="Trace identifier")
    plan_id: str = Field(..., description="Associated plan")

    # Reasoning steps
    step_number: int = Field(..., description="Step number in reasoning")
    description: str = Field(..., description="What was considered")
    decision: str = Field(..., description="Decision made")
    rationale: str = Field(..., description="Why this decision")

    # Alternatives
    alternatives_considered: List[str] = Field(default_factory=list)
    rejected_alternatives: List[str] = Field(default_factory=list)

    # Metrics
    confidence_score: float = Field(..., description="Confidence in decision 0-1")
    information_gain: Optional[float] = Field(None, description="Information gain from this step")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    agent: str = Field(..., description="Agent making decision")


class SearchPath(BaseModel):
    """Search path explored during planning"""
    path_id: str = Field(..., description="Path identifier")
    correlation_id: str = Field(..., description="Correlation ID")

    # Path details
    steps: List[str] = Field(..., description="Steps in this path")
    score: float = Field(..., description="Path score")

    # Evaluation metrics
    constraint_satisfaction: float = Field(..., description="How well constraints satisfied 0-1")
    expected_return: float = Field(..., description="Expected return")
    risk_score: float = Field(..., description="Risk score")

    # Status
    pruned: bool = Field(False, description="Was path pruned")
    pruning_reason: Optional[str] = Field(None, description="Why pruned")
    selected: bool = Field(False, description="Was this path selected")

    # Metadata
    explored_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

class PerformanceMetrics(BaseModel):
    """Performance metrics for monitoring"""
    metric_id: str = Field(..., description="Metric identifier")
    agent: str = Field(..., description="Agent being measured")

    # Timing metrics
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    # Quality metrics
    success_rate: float = Field(..., description="Success rate 0-1")
    error_rate: float = Field(..., description="Error rate 0-1")

    # Resource metrics
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    api_calls_made: Optional[int] = Field(None, description="Number of API calls")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = Field(None)


# ============================================================================
# EXPORTS
# ============================================================================

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
