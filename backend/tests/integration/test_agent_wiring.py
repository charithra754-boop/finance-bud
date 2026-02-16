"""
Agent Communication Wiring Integration Tests

Verifies that all API endpoints and CMVL workflow components
properly route messages through the AgentCommunicationFramework.

Requirements: 6.1, 6.2, 11.1, 11.2
"""

import asyncio
import pytest
import time
from uuid import uuid4
from typing import Dict, Any, Optional

from agents.communication import AgentCommunicationFramework, AgentRegistry
from tests.mocks.mock_interfaces import (
    MockOrchestrationAgent, MockPlanningAgent, MockInformationRetrievalAgent,
    MockVerificationAgent, MockExecutionAgent
)
from data_models.schemas import (
    AgentMessage, MessageType, Priority
)


class TestAPIEndpointFrameworkRouting:
    """Verify that API endpoint messages are tracked by the communication framework"""
    
    @pytest.fixture
    def framework(self):
        """Create fresh communication framework"""
        return AgentCommunicationFramework()
    
    @pytest.fixture
    async def registered_agents(self, framework):
        """Create and register mock agents with the framework"""
        agents = {
            'orchestration': MockOrchestrationAgent("orchestration_agent"),
            'planning': MockPlanningAgent("planning_agent"),
            'information_retrieval': MockInformationRetrievalAgent("information_retrieval_agent"),
            'verification': MockVerificationAgent("verification_agent"),
            'execution': MockExecutionAgent("execution_agent")
        }
        
        for name, agent in agents.items():
            await framework.register_agent(agent, [f"{name}_capability"])
            agent.set_communication_framework(framework)
        
        return agents

    @pytest.mark.asyncio
    async def test_orchestration_message_tracked_by_framework(self, framework, registered_agents):
        """Messages sent to orchestration agent should be tracked in framework metrics"""
        request_id = str(uuid4())
        
        message = framework.create_message(
            sender_id="api_gateway",
            target_id="orchestration_agent",
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Save $10,000 for emergency fund"},
            correlation_id=request_id
        )
        
        success = await framework.send_message(message)
        assert success is True
        
        # Verify the framework tracked the message
        health = framework.get_system_health()
        assert health["total_messages"] >= 1
        assert health["successful_messages"] >= 1
        
        # Verify correlation tracking
        trace = framework.get_correlation_trace(request_id)
        assert len(trace) >= 1
        assert trace[0].correlation_id == request_id

    @pytest.mark.asyncio
    async def test_planning_message_tracked_by_framework(self, framework, registered_agents):
        """Messages sent to planning agent should be tracked in framework metrics"""
        request_id = str(uuid4())
        
        message = framework.create_message(
            sender_id="api_gateway",
            target_id="planning_agent",
            message_type=MessageType.REQUEST,
            payload={"planning_request": {"goal": "retirement"}},
            correlation_id=request_id
        )
        
        success = await framework.send_message(message)
        assert success is True
        
        health = framework.get_system_health()
        assert health["total_messages"] >= 1

    @pytest.mark.asyncio
    async def test_market_message_tracked_by_framework(self, framework, registered_agents):
        """Messages sent to information retrieval agent should be tracked"""
        request_id = str(uuid4())
        
        message = framework.create_message(
            sender_id="api_gateway",
            target_id="information_retrieval_agent",
            message_type=MessageType.REQUEST,
            payload={"market_data_request": {"symbols": ["AAPL"]}},
            correlation_id=request_id
        )
        
        success = await framework.send_message(message)
        assert success is True
        
        trace = framework.get_correlation_trace(request_id)
        assert len(trace) >= 1

    @pytest.mark.asyncio
    async def test_verification_message_tracked_by_framework(self, framework, registered_agents):
        """Messages sent to verification agent should be tracked"""
        request_id = str(uuid4())
        
        message = framework.create_message(
            sender_id="api_gateway",
            target_id="verification_agent",
            message_type=MessageType.REQUEST,
            payload={"verification_request": {"plan_id": "test"}},
            correlation_id=request_id
        )
        
        success = await framework.send_message(message)
        assert success is True
        
        health = framework.get_system_health()
        assert health["successful_messages"] >= 1

    @pytest.mark.asyncio
    async def test_multiple_endpoints_increment_framework_metrics(self, framework, registered_agents):
        """Messages from multiple endpoints should all be tracked cumulatively"""
        targets = [
            ("orchestration_agent", {"user_goal": "test"}),
            ("planning_agent", {"planning_request": {}}),
            ("information_retrieval_agent", {"market_data_request": {}}),
            ("verification_agent", {"verification_request": {}}),
        ]
        
        for target_id, payload in targets:
            message = framework.create_message(
                sender_id="api_gateway",
                target_id=target_id,
                message_type=MessageType.REQUEST,
                payload=payload
            )
            await framework.send_message(message)
        
        health = framework.get_system_health()
        assert health["total_messages"] >= 4
        assert health["success_rate"] >= 0.9


class TestOrchestrationAgentUserGoalHandler:
    """Verify OrchestrationAgent properly handles user_goal messages"""
    
    @pytest.fixture
    def framework(self):
        return AgentCommunicationFramework()
    
    @pytest.fixture
    async def orchestration_agent(self, framework):
        agent = MockOrchestrationAgent("orchestration_agent")
        await framework.register_agent(agent, ["workflow_coordination"])
        agent.set_communication_framework(framework)
        return agent

    @pytest.mark.asyncio
    async def test_user_goal_message_processed(self, framework, orchestration_agent):
        """Agent should process messages with user_goal payload"""
        message = framework.create_message(
            sender_id="api_gateway",
            target_id="orchestration_agent",
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Build a 6-month emergency fund"}
        )
        
        response = await orchestration_agent.process_message(message)
        assert response is not None
        assert response.message_type in [MessageType.RESPONSE, MessageType.NOTIFICATION]

    @pytest.mark.asyncio
    async def test_user_goal_message_has_correlation_id(self, framework, orchestration_agent):
        """Response should preserve the correlation ID for tracking"""
        correlation_id = str(uuid4())
        
        message = framework.create_message(
            sender_id="api_gateway",
            target_id="orchestration_agent",
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Plan for retirement"},
            correlation_id=correlation_id
        )
        
        response = await orchestration_agent.process_message(message)
        if response:
            assert response.correlation_id == correlation_id


class TestAgentHealthEndpoint:
    """Verify agent health monitoring data"""
    
    @pytest.fixture
    async def framework_with_agents(self):
        framework = AgentCommunicationFramework()
        agents = {
            'orchestration': MockOrchestrationAgent("orchestration_agent"),
            'planning': MockPlanningAgent("planning_agent"),
            'information_retrieval': MockInformationRetrievalAgent("information_retrieval_agent"),
            'verification': MockVerificationAgent("verification_agent"),
            'execution': MockExecutionAgent("execution_agent")
        }
        
        for name, agent in agents.items():
            await framework.register_agent(agent, [f"{name}_capability"])
        
        return framework, agents

    @pytest.mark.asyncio
    async def test_system_health_contains_required_fields(self, framework_with_agents):
        """Health data should contain all required monitoring fields"""
        framework, _ = framework_with_agents
        
        health = framework.get_system_health()
        
        required_fields = [
            "total_messages", "successful_messages", "failed_messages",
            "success_rate", "registered_agents", "agent_health",
            "circuit_breakers"
        ]
        
        for field in required_fields:
            assert field in health, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_agent_health_per_agent(self, framework_with_agents):
        """Health data should include per-agent status"""
        framework, agents = framework_with_agents
        
        health = framework.get_system_health()
        assert health["registered_agents"] == 5
        
        agent_health = health["agent_health"]
        assert len(agent_health) == 5

    @pytest.mark.asyncio
    async def test_health_after_message_traffic(self, framework_with_agents):
        """Health metrics should reflect message traffic"""
        framework, _ = framework_with_agents
        
        # Send some messages
        for i in range(5):
            message = framework.create_message(
                sender_id="orchestration_agent",
                target_id="planning_agent",
                message_type=MessageType.REQUEST,
                payload={"test": i}
            )
            await framework.send_message(message)
        
        health = framework.get_system_health()
        assert health["total_messages"] >= 5
        assert health["success_rate"] > 0


class TestBaseAgentFrameworkEnforcement:
    """Verify BaseAgent raises error when framework is not set"""
    
    @pytest.mark.asyncio
    async def test_send_message_raises_without_framework(self):
        """Agent should raise RuntimeError if no framework set"""
        agent = MockPlanningAgent("test_no_framework_agent")
        # Don't set communication framework
        
        message = AgentMessage(
            agent_id="test_no_framework_agent",
            target_agent_id="some_target",
            message_type=MessageType.REQUEST,
            payload={"test": "data"},
            correlation_id=str(uuid4()),
            session_id=str(uuid4()),
            trace_id=str(uuid4())
        )
        
        with pytest.raises(RuntimeError, match="no communication framework set"):
            await agent.send_message(message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
