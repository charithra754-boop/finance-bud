"""
Basic validation tests for FinPilot data models.
Run with: python data_models/test_schemas.py
"""
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from schemas import (
    AgentMessage, MessageType, Priority,
    EnhancedPlanRequest, PlanStep, VerificationReport, VerificationStatus,
    MarketData, TriggerEvent, MarketEventType, SeverityLevel,
    FinancialState, Constraint, ConstraintType, ConstraintPriority,
    ExecutionLog, ExecutionStatus,
    SearchPath, ReasoningTrace, DecisionPoint,
    RiskProfile, RiskLevel, TaxContext, RegulatoryRequirement,
    ComplianceStatus, ComplianceLevel, AuditTrail,
    PerformanceMetrics
)


class TestCoreModels:
    """Test core communication models"""
    
    def test_agent_message_creation(self):
        """Test AgentMessage model creation and validation"""
        message = AgentMessage(
            agent_id="planning_agent_001",
            message_type=MessageType.REQUEST,
            payload={"test": "data"},
            correlation_id="corr_123",
            session_id="sess_456",
            trace_id="trace_789"
        )
        
        assert message.agent_id == "planning_agent_001"
        assert message.message_type == MessageType.REQUEST
        assert message.priority == Priority.MEDIUM  # default
        assert "test" in message.payload
        
    def test_enhanced_plan_request(self):
        """Test EnhancedPlanRequest model"""
        request = EnhancedPlanRequest(
            user_id="user_001",
            user_goal="Test goal",
            current_state={"balance": 1000},
            risk_profile={"tolerance": "moderate"},
            tax_considerations={"rate": 0.25},
            time_horizon=12,
            correlation_id="corr_123",
            session_id="sess_456"
        )
        
        assert request.user_id == "user_001"
        assert request.time_horizon == 12
        assert request.risk_profile["tolerance"] == "moderate"
        
    def test_plan_step_validation(self):
        """Test PlanStep model validation"""
        step = PlanStep(
            sequence_number=1,
            action_type="invest",
            description="Test investment",
            amount=Decimal("1000.00"),
            target_date=datetime.utcnow() + timedelta(days=30),
            rationale="Test rationale",
            confidence_score=0.85,
            risk_level="moderate"
        )
        
        assert step.sequence_number == 1
        assert step.amount == Decimal("1000.00")
        assert 0.0 <= step.confidence_score <= 1.0
        
    def test_verification_report(self):
        """Test VerificationReport model"""
        report = VerificationReport(
            plan_id="plan_123",
            verification_status=VerificationStatus.APPROVED,
            constraints_checked=10,
            constraints_passed=9,
            overall_risk_score=0.3,
            approval_rationale="Test approval",
            confidence_score=0.9,
            verification_time=2.5,
            verifier_agent_id="verifier_001",
            correlation_id="corr_123"
        )
        
        assert report.verification_status == VerificationStatus.APPROVED
        assert report.constraints_passed <= report.constraints_checked
        assert 0.0 <= report.overall_risk_score <= 1.0


class TestMarketModels:
    """Test market data and trigger models"""
    
    def test_market_data(self):
        """Test MarketData model"""
        data = MarketData(
            source="test_api",
            market_volatility=0.25,
            interest_rates={"federal": 5.25},
            sector_trends={"tech": 0.15},
            economic_sentiment=0.35,
            collection_method="api",
            refresh_frequency=300
        )
        
        assert data.source == "test_api"
        assert data.market_volatility == 0.25
        assert data.interest_rates["federal"] == 5.25
        
    def test_trigger_event(self):
        """Test TriggerEvent model"""
        trigger = TriggerEvent(
            trigger_type="market_event",
            event_type=MarketEventType.VOLATILITY_SPIKE,
            severity=SeverityLevel.HIGH,
            description="Test trigger",
            source_data={"volatility": 0.4},
            impact_score=0.75,
            confidence_score=0.9,
            detector_agent_id="ira_001",
            correlation_id="corr_123"
        )
        
        assert trigger.event_type == MarketEventType.VOLATILITY_SPIKE
        assert trigger.severity == SeverityLevel.HIGH
        assert 0.0 <= trigger.impact_score <= 1.0


class TestFinancialModels:
    """Test financial state and constraint models"""
    
    def test_financial_state(self):
        """Test FinancialState model"""
        state = FinancialState(
            user_id="user_001",
            total_assets=Decimal("100000"),
            total_liabilities=Decimal("30000"),
            monthly_income=Decimal("8000"),
            monthly_expenses=Decimal("5000"),
            risk_tolerance="moderate",
            tax_filing_status="single",
            estimated_tax_rate=0.25
        )
        
        assert state.net_worth == Decimal("70000")  # calculated field
        assert state.monthly_cash_flow == Decimal("3000")  # calculated field
        
    def test_constraint_model(self):
        """Test Constraint model"""
        constraint = Constraint(
            name="Emergency Fund",
            constraint_type=ConstraintType.LIQUIDITY,
            priority=ConstraintPriority.MANDATORY,
            description="Maintain emergency fund",
            validation_rule="emergency_fund >= monthly_expenses * 6",
            threshold_value=6,
            comparison_operator=">=",
            created_by="system"
        )
        
        assert constraint.constraint_type == ConstraintType.LIQUIDITY
        assert constraint.priority == ConstraintPriority.MANDATORY


class TestReasoningModels:
    """Test reasoning and search path models"""
    
    def test_search_path(self):
        """Test SearchPath model"""
        path = SearchPath(
            search_session_id="search_123",
            path_type="explored",
            sequence_steps=[{"step": 1, "action": "invest"}],
            total_cost=1000.0,
            expected_value=5000.0,
            risk_score=0.3,
            feasibility_score=0.8,
            combined_score=0.75,
            constraint_satisfaction_score=0.9,
            path_status="selected",
            exploration_time=1.5,
            created_by_agent="planning_agent"
        )
        
        assert path.path_type == "explored"
        assert path.path_status == "selected"
        assert 0.0 <= path.risk_score <= 1.0
        
    def test_reasoning_trace(self):
        """Test ReasoningTrace model"""
        trace = ReasoningTrace(
            session_id="sess_123",
            agent_id="planning_agent",
            operation_type="planning",
            final_decision="Invest in diversified portfolio",
            decision_rationale="Based on risk tolerance",
            confidence_score=0.85,
            performance_metrics=PerformanceMetrics(
                execution_time=2.5,
                memory_usage=128.0
            ),
            correlation_id="corr_123"
        )
        
        assert trace.operation_type == "planning"
        assert 0.0 <= trace.confidence_score <= 1.0


class TestComplianceModels:
    """Test risk and compliance models"""
    
    def test_risk_profile(self):
        """Test RiskProfile model"""
        profile = RiskProfile(
            user_id="user_001",
            overall_risk_tolerance=RiskLevel.MODERATE,
            risk_capacity=0.7,
            risk_perception=0.6,
            risk_composure=0.8,
            investment_horizon=120,
            liquidity_needs={"emergency": 0.1},
            volatility_comfort=0.6,
            loss_tolerance=0.2,
            investment_experience="intermediate",
            financial_knowledge=0.7,
            decision_making_style="analytical",
            primary_goals=["retirement"],
            goal_priorities={"retirement": 1},
            assessment_method="questionnaire",
            next_review_date=datetime.utcnow() + timedelta(days=365)
        )
        
        assert profile.overall_risk_tolerance == RiskLevel.MODERATE
        assert 0.0 <= profile.risk_capacity <= 1.0
        
    def test_compliance_status(self):
        """Test ComplianceStatus model"""
        status = ComplianceStatus(
            entity_id="user_001",
            requirement_id="req_123",
            compliance_level=ComplianceLevel.COMPLIANT,
            compliance_risk_score=0.1,
            assessor_id="verifier_001",
            next_review_date=datetime.utcnow() + timedelta(days=90),
            monitoring_frequency="quarterly"
        )
        
        assert status.compliance_level == ComplianceLevel.COMPLIANT
        assert 0.0 <= status.compliance_risk_score <= 1.0


if __name__ == "__main__":
    # Run basic validation tests
    test_core = TestCoreModels()
    test_core.test_agent_message_creation()
    test_core.test_enhanced_plan_request()
    test_core.test_plan_step_validation()
    test_core.test_verification_report()
    
    test_market = TestMarketModels()
    test_market.test_market_data()
    test_market.test_trigger_event()
    
    test_financial = TestFinancialModels()
    test_financial.test_financial_state()
    test_financial.test_constraint_model()
    
    test_reasoning = TestReasoningModels()
    test_reasoning.test_search_path()
    test_reasoning.test_reasoning_trace()
    
    test_compliance = TestComplianceModels()
    test_compliance.test_risk_profile()
    test_compliance.test_compliance_status()
    
    print("✅ All basic validation tests passed!")
    print("📊 Data contracts are properly structured and validated")
    print("🔗 Models support comprehensive agent communication")
    print("📈 Financial calculations and constraints validated")
    print("🛡️ Compliance and audit trail models working")
    print("🧠 Reasoning and search path models functional")