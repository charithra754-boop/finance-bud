"""
Simple test for ExecutionAgent implementation
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.execution_agent import ExecutionAgent, TransactionType, AccountType
from data_models.schemas import AgentMessage, MessageType
from uuid import uuid4
from decimal import Decimal


async def test_execution_agent():
    """Test basic ExecutionAgent functionality"""
    
    print("Testing ExecutionAgent implementation...")
    
    # Create execution agent
    agent = ExecutionAgent("test_execution_agent")
    await agent.start()
    
    print(f"✓ ExecutionAgent created: {agent.agent_id}")
    
    # Test health check
    health_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={"action": "health_check"},
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    response = await agent.process_message(health_message)
    assert response is not None
    assert response.payload["action"] == "health_check_response"
    print("✓ Health check passed")
    
    # Test portfolio status
    portfolio_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={"action": "get_portfolio_status"},
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    response = await agent.process_message(portfolio_message)
    assert response is not None
    assert "portfolio_status" in response.payload
    print("✓ Portfolio status request passed")
    
    # Test transaction execution
    transaction_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "execute_transaction",
            "transaction": {
                "transaction_type": "investment",
                "amount": 1000,
                "asset_symbol": "SPY",
                "description": "Test investment"
            }
        },
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    response = await agent.process_message(transaction_message)
    assert response is not None
    assert "transaction_executed" in response.payload["action"]
    print("✓ Transaction execution passed")
    
    # Test plan execution
    plan_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "execute_plan",
            "plan": {
                "plan_id": str(uuid4()),
                "steps": [
                    {
                        "step_id": str(uuid4()),
                        "action_type": "invest",
                        "amount": 5000,
                        "asset_symbol": "SPY",
                        "account_type": "investment"
                    },
                    {
                        "step_id": str(uuid4()),
                        "action_type": "save",
                        "amount": 2000,
                        "account_type": "savings"
                    }
                ]
            }
        },
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    response = await agent.process_message(plan_message)
    assert response is not None
    assert "execution_results" in response.payload
    print("✓ Plan execution passed")
    
    # Test forecast generation
    forecast_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "generate_forecast",
            "forecast_params": {
                "time_horizon_months": 24
            }
        },
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    response = await agent.process_message(forecast_message)
    assert response is not None
    assert "forecast" in response.payload
    print("✓ Forecast generation passed")
    
    # Test compliance check
    compliance_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={"action": "check_compliance"},
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    response = await agent.process_message(compliance_message)
    assert response is not None
    assert "compliance_report" in response.payload
    print("✓ Compliance check passed")
    
    # Test tax optimization
    tax_message = AgentMessage(
        agent_id="test_client",
        target_agent_id=agent.agent_id,
        message_type=MessageType.REQUEST,
        payload={"action": "optimize_taxes"},
        correlation_id=str(uuid4()),
        session_id=str(uuid4()),
        trace_id=str(uuid4())
    )
    
    response = await agent.process_message(tax_message)
    assert response is not None
    assert "tax_optimization_complete" in response.payload["action"]
    print("✓ Tax optimization passed")
    
    # Test ledger functionality
    print("\nTesting ledger functionality...")
    
    # Create accounts
    investment_account = agent.ledger.create_account(
        AccountType.INVESTMENT,
        "Test Investment Account",
        Decimal('10000')
    )
    print(f"✓ Created investment account: {investment_account}")
    
    savings_account = agent.ledger.create_account(
        AccountType.SAVINGS,
        "Test Savings Account",
        Decimal('5000')
    )
    print(f"✓ Created savings account: {savings_account}")
    
    # Check total balance
    total_balance = agent.ledger.get_total_balance()
    assert total_balance == Decimal('15000')
    print(f"✓ Total balance correct: ${total_balance}")
    
    # Test forecast engine
    print("\nTesting forecast engine...")
    
    forecast = agent.forecast_engine.forecast_portfolio_performance(
        {"total_value": 15000},
        12  # 12 months
    )
    
    assert "forecasts" in forecast
    assert "expected_value" in forecast["forecasts"]
    print(f"✓ Forecast generated: Expected value ${forecast['forecasts']['expected_value']:.2f}")
    
    # Test tax optimizer
    print("\nTesting tax optimizer...")
    
    tax_opportunities = agent.tax_optimizer.suggest_tax_loss_harvesting(
        agent.ledger.accounts,
        agent.market_prices
    )
    
    print(f"✓ Tax loss harvesting opportunities: {len(tax_opportunities)}")
    
    # Test compliance engine
    print("\nTesting compliance engine...")
    
    compliance_report = agent.compliance_engine.generate_compliance_report(
        list(agent.ledger.transactions.values()),
        agent.ledger.accounts
    )
    
    assert "compliance_summary" in compliance_report
    print(f"✓ Compliance report generated: {compliance_report['compliance_summary']}")
    
    # Get comprehensive status
    status = agent.get_comprehensive_status()
    print(f"\nAgent Status:")
    print(f"  - Status: {status['status']}")
    print(f"  - Total Accounts: {status['ledger_summary']['total_accounts']}")
    print(f"  - Portfolio Value: ${status['portfolio_value']:.2f}")
    print(f"  - Total Transactions: {status['execution_metrics']['total_transactions']}")
    
    await agent.stop()
    print("\n✓ All tests passed! ExecutionAgent implementation is working correctly.")


if __name__ == "__main__":
    asyncio.run(test_execution_agent())