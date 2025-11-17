"""
Agent Testing Framework

Tests for individual agent functionality, mock interfaces,
and agent communication protocols.

Requirements: 11.1, 11.2
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from agents.mock_interfaces import (
    MockOrchestrationAgent,
    MockPlanningAgent,
    MockInformationRetrievalAgent,
    MockVerificationAgent,
    MockExecutionAgent
)
from agents.communication import AgentCommunicationFramework
from data_models.schemas import AgentMessage, MessageType, Priority


class TestMockAgents:
    """Test suite for mock agent implementations"""
    
    @pytest.fixture
    def communication_framework(self):
        """Create communication framework for testing"""
        return AgentCommunicationFramework()
    
    @pytest.fixture
    def mock_orchestration_agent(self):
        """Create mock orchestration agent"""
        return MockOrchestrationAgent()
    
    @pytest.fixture
    def mock_planning_agent(self):
        """Create mock planning agent"""
        return MockPlanningAgent()
    
    @pytest.fixture
    def mock_ira_agent(self):
        """Create mock information retrieval agent"""
        return MockInformationRetrievalAgent()
    
    @pytest.fixture
    def mock_verification_agent(self):
        """Create mock verification agent"""
        return MockVerificationAgent()
    
    @pytest.fixture
    def mock_execution_agent(self):
        """Create mock execution agent"""
        return MockExecutionAgent()
    
    async def test_orchestration_agent_planning_request(self, mock_orchestration_agent):
        """Test orchestration agent handles planning requests"""
        message = AgentMessage(
            agent_id="test_client",
            target_agent_id=mock_orchestration_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Save for retirement"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_orchestration_agent.process_message(message)
        
        assert response is not None
        assert response.message_type == MessageType.RESPONSE
        assert "workflow_plan" in response.payload
        assert response.payload["status"] == "workflow_initiated"
    
    async def test_orchestration_agent_trigger_handling(self, mock_orchestration_agent):
        """Test orchestration agent handles trigger events"""
        message = AgentMessage(
            agent_id="test_ira",
            target_agent_id=mock_orchestration_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "trigger_event": {
                    "trigger_id": str(uuid4()),
                    "severity": "high",
                    "description": "Market volatility spike"
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_orchestration_agent.process_message(message)
        
        assert response is not None
        assert response.payload["cmvl_activated"] is True
        assert "actions_initiated" in response.payload
    
    async def test_planning_agent_plan_generation(self, mock_planning_agent):
        """Test planning agent generates comprehensive plans"""
        message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=mock_planning_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "planning_request": {
                    "user_goal": "Build emergency fund",
                    "time_horizon": 12
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_planning_agent.process_message(message)
        
        assert response is not None
        assert response.payload["plan_generated"] is True
        assert "plan_steps" in response.payload
        assert "search_paths" in response.payload
        assert "reasoning_trace" in response.payload
        assert len(response.payload["search_paths"]) >= 3  # Multiple strategies
    
    async def test_ira_market_data_fetching(self, mock_ira_agent):
        """Test IRA fetches realistic market data"""
        message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=mock_ira_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"market_data_request": {"symbols": ["SPY", "BND"]}},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_ira_agent.process_message(message)
        
        assert response is not None
        assert "market_data" in response.payload
        market_data = response.payload["market_data"]
        assert "market_volatility" in market_data
        assert "interest_rates" in market_data
        assert "sector_trends" in market_data
    
    async def test_ira_trigger_detection(self, mock_ira_agent):
        """Test IRA detects market triggers"""
        # Set volatile scenario
        await mock_ira_agent._simulate_scenario(AgentMessage(
            agent_id="test",
            target_agent_id=mock_ira_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"scenario": "volatile"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        ))
        
        message = AgentMessage(
            agent_id="test_orchestrator",
            target_agent_id=mock_ira_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={"trigger_detection": {}},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_ira_agent.process_message(message)
        
        assert response is not None
        assert "triggers_detected" in response.payload
        assert response.payload["monitoring_active"] is True
    
    async def test_verification_agent_plan_verification(self, mock_verification_agent):
        """Test verification agent verifies plans"""
        message = AgentMessage(
            agent_id="test_planning",
            target_agent_id=mock_verification_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "verification_request": {
                    "plan_id": str(uuid4()),
                    "plan_steps": [
                        {
                            "step_id": str(uuid4()),
                            "action_type": "invest",
                            "amount": 50000,
                            "risk_level": "medium"
                        }
                    ]
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_verification_agent.process_message(message)
        
        assert response is not None
        assert "verification_report" in response.payload
        report = response.payload["verification_report"]
        assert "verification_status" in report
        assert "constraints_checked" in report
        assert "overall_risk_score" in report
    
    async def test_execution_agent_plan_execution(self, mock_execution_agent):
        """Test execution agent executes plans"""
        message = AgentMessage(
            agent_id="test_verification",
            target_agent_id=mock_execution_agent.agent_id,
            message_type=MessageType.REQUEST,
            payload={
                "execution_request": {
                    "plan_steps": [
                        {
                            "step_id": str(uuid4()),
                            "action_type": "buy_stock",
                            "amount": 10000
                        }
                    ]
                }
            },
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        response = await mock_execution_agent.process_message(message)
        
        assert response is not None
        assert response.payload["execution_completed"] is True
        assert "execution_results" in response.payload
        assert response.payload["portfolio_updated"] is True
    
    async def test_agent_health_monitoring(self, mock_orchestration_agent):
        """Test agent health monitoring"""
        await mock_orchestration_agent.start()
        
        health_status = mock_orchestration_agent.get_health_status()
        
        assert health_status["agent_id"] == mock_orchestration_agent.agent_id
        assert health_status["status"] == "running"
        assert "uptime_seconds" in health_status
        assert "success_rate" in health_status
        
        await mock_orchestration_agent.stop()
    
    async def test_agent_performance_metrics(self, mock_planning_agent):
        """Test agent performance metrics collection"""
        metrics = mock_planning_agent.get_performance_metrics()
        
        assert hasattr(metrics, 'execution_time')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'success_rate')
        assert 0.0 <= metrics.success_rate <= 1.0


class TestAgentCommunication:
    """Test suite for agent communication protocols"""
    
    @pytest.fixture
    def communication_framework(self):
        """Create communication framework"""
        return AgentCommunicationFramework()
    
    @pytest.fixture
    def test_agents(self, communication_framework):
        """Create and register test agents"""
        agents = {
            'orchestrator': MockOrchestrationAgent("test_oa_001"),
            'planner': MockPlanningAgent("test_pa_001"),
            'ira': MockInformationRetrievalAgent("test_ira_001"),
            'verifier': MockVerificationAgent("test_va_001"),
            'executor': MockExecutionAgent("test_ea_001")
        }
        
        # Register agents with capabilities
        communication_framework.register_agent(agents['orchestrator'], ['workflow_management', 'trigger_handling'])
        communication_framework.register_agent(agents['planner'], ['financial_planning', 'strategy_generation'])
        communication_framework.register_agent(agents['ira'], ['market_data', 'trigger_detection'])
        communication_framework.register_agent(agents['verifier'], ['constraint_checking', 'compliance'])
        communication_framework.register_agent(agents['executor'], ['transaction_execution', 'portfolio_management'])
        
        return agents
    
    async def test_message_routing(self, communication_framework, test_agents):
        """Test message routing between agents"""
        message = communication_framework.create_message(
            sender_id="test_oa_001",
            target_id="test_pa_001",
            message_type=MessageType.REQUEST,
            payload={"test": "routing"}
        )
        
        success = await communication_framework.send_message(message)
        assert success is True
        
        # Check message was received
        planner = test_agents['planner']
        assert planner.message_queue.qsize() > 0
    
    async def test_broadcast_messaging(self, communication_framework, test_agents):
        """Test broadcast messaging to all agents"""
        message = communication_framework.create_message(
            sender_id="test_oa_001",
            target_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            payload={"broadcast": "system_update"}
        )
        
        success = await communication_framework.send_message(message)
        assert success is True
        
        # Check all agents except sender received message
        for agent_name, agent in test_agents.items():
            if agent.agent_id != "test_oa_001":
                assert agent.message_queue.qsize() > 0
    
    async def test_correlation_tracking(self, communication_framework, test_agents):
        """Test correlation ID tracking across messages"""
        correlation_id = str(uuid4())
        
        # Send multiple related messages
        for i in range(3):
            message = communication_framework.create_message(
                sender_id="test_oa_001",
                target_id="test_pa_001",
                message_type=MessageType.REQUEST,
                payload={"sequence": i},
                correlation_id=correlation_id
            )
            await communication_framework.send_message(message)
        
        # Check correlation tracking
        trace = communication_framework.get_correlation_trace(correlation_id)
        assert len(trace) == 3
        assert all(msg.correlation_id == correlation_id for msg in trace)
    
    async def test_circuit_breaker_functionality(self, communication_framework, test_agents):
        """Test circuit breaker prevents cascade failures"""
        # This would require simulating failures in a real implementation
        # For now, test that circuit breakers are created
        assert "test_oa_001" in communication_framework.circuit_breakers
        assert "test_pa_001" in communication_framework.circuit_breakers
    
    async def test_system_health_monitoring(self, communication_framework, test_agents):
        """Test system health monitoring"""
        health = communication_framework.get_system_health()
        
        assert "framework_uptime" in health
        assert "total_messages" in health
        assert "success_rate" in health
        assert "registered_agents" in health
        assert health["registered_agents"] == len(test_agents)
        assert "agent_health" in health
        assert "circuit_breakers" in health


# Async test runner for pytest
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])