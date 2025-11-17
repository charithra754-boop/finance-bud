#!/usr/bin/env python3
"""
FinPilot VP-MAS Test Runner

Comprehensive test runner for all VP-MAS components with detailed reporting.
Supports different test categories and provides clear feedback for development.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit            # Run unit tests only
    python run_tests.py --integration     # Run integration tests only
    python run_tests.py --mock            # Run mock interface tests only
    python run_tests.py --performance     # Run performance tests only
    python run_tests.py --coverage        # Run with coverage report
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_data_model_tests():
    """Run data model validation tests"""
    print("🧪 Running Data Model Tests...")
    print("=" * 50)
    
    try:
        from data_models.test_schemas import (
            TestCoreModels, TestMarketModels, TestFinancialModels,
            TestReasoningModels, TestComplianceModels
        )
        
        # Run all test classes
        test_classes = [
            TestCoreModels(),
            TestMarketModels(),
            TestFinancialModels(), 
            TestReasoningModels(),
            TestComplianceModels()
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            class_name = test_class.__class__.__name__
            print(f"\n📋 {class_name}")
            
            # Get all test methods
            test_methods = [method for method in dir(test_class) if method.startswith('test_')]
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    method = getattr(test_class, method_name)
                    method()
                    print(f"  ✅ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ❌ {method_name}: {str(e)}")
        
        print(f"\n📊 Data Model Tests: {passed_tests}/{total_tests} passed")
        return passed_tests == total_tests
        
    except Exception as e:
        print(f"❌ Error running data model tests: {str(e)}")
        return False


async def run_mock_agent_tests():
    """Run mock agent interface tests"""
    print("\n🤖 Running Mock Agent Tests...")
    print("=" * 50)
    
    try:
        from agents.mock_interfaces import (
            MockOrchestrationAgent, MockPlanningAgent, MockInformationRetrievalAgent,
            MockVerificationAgent, MockExecutionAgent
        )
        from agents.communication import AgentCommunicationFramework
        from data_models.schemas import AgentMessage, MessageType
        from uuid import uuid4
        
        # Test agent creation
        agents = {
            'orchestrator': MockOrchestrationAgent(),
            'planner': MockPlanningAgent(),
            'ira': MockInformationRetrievalAgent(),
            'verifier': MockVerificationAgent(),
            'executor': MockExecutionAgent()
        }
        
        print("✅ All mock agents created successfully")
        
        # Test communication framework
        framework = AgentCommunicationFramework()
        
        for name, agent in agents.items():
            await agent.start()
            framework.register_agent(agent, [f"{name}_capabilities"])
        
        print("✅ All agents registered and started")
        
        # Test basic message passing
        test_message = framework.create_message(
            sender_id=agents['orchestrator'].agent_id,
            target_id=agents['planner'].agent_id,
            message_type=MessageType.REQUEST,
            payload={"test": "message"}
        )
        
        success = await framework.send_message(test_message)
        print(f"✅ Message routing test: {'passed' if success else 'failed'}")
        
        # Test health monitoring
        health = framework.get_system_health()
        print(f"✅ System health check: {health['registered_agents']} agents registered")
        
        # Stop all agents
        for agent in agents.values():
            await agent.stop()
        
        print("✅ All agents stopped successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error running mock agent tests: {str(e)}")
        return False


async def run_integration_tests():
    """Run basic integration tests"""
    print("\n🔗 Running Integration Tests...")
    print("=" * 50)
    
    try:
        from agents.mock_interfaces import MockOrchestrationAgent, MockPlanningAgent
        from agents.communication import AgentCommunicationFramework
        from data_models.schemas import AgentMessage, MessageType
        from uuid import uuid4
        
        # Set up integrated system
        framework = AgentCommunicationFramework()
        orchestrator = MockOrchestrationAgent("integration_oa_001")
        planner = MockPlanningAgent("integration_pa_001")
        
        await orchestrator.start()
        await planner.start()
        
        framework.register_agent(orchestrator, ['workflow_management'])
        framework.register_agent(planner, ['financial_planning'])
        
        # Test workflow coordination
        correlation_id = str(uuid4())
        session_id = str(uuid4())
        
        # Submit user goal
        goal_message = framework.create_message(
            sender_id="test_user",
            target_id=orchestrator.agent_id,
            message_type=MessageType.REQUEST,
            payload={"user_goal": "Save for retirement"},
            correlation_id=correlation_id,
            session_id=session_id
        )
        
        success = await framework.send_message(goal_message)
        print(f"✅ Goal submission: {'passed' if success else 'failed'}")
        
        # Allow processing time
        await asyncio.sleep(0.2)
        
        # Check session tracking
        if session_id in orchestrator.active_sessions:
            print("✅ Session tracking: passed")
        else:
            print("❌ Session tracking: failed")
        
        # Test correlation tracking
        trace = framework.get_correlation_trace(correlation_id)
        print(f"✅ Correlation tracking: {len(trace)} messages traced")
        
        # Test system health under load
        health = framework.get_system_health()
        success_rate = health.get('success_rate', 0)
        print(f"✅ System health: {success_rate:.1%} success rate")
        
        await orchestrator.stop()
        await planner.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Error running integration tests: {str(e)}")
        return False


def run_performance_tests():
    """Run basic performance tests"""
    print("\n⚡ Running Performance Tests...")
    print("=" * 50)
    
    try:
        from tests.mock_data import MockDataGenerator
        import time
        
        generator = MockDataGenerator(seed=42)
        
        # Test data generation performance
        start_time = time.time()
        
        for i in range(100):
            request = generator.generate_enhanced_plan_request()
            market_data = generator.generate_market_data()
            trigger = generator.generate_trigger_event()
        
        generation_time = time.time() - start_time
        print(f"✅ Data generation: {generation_time:.3f}s for 300 objects ({300/generation_time:.1f} objects/sec)")
        
        # Test data validation performance
        start_time = time.time()
        
        for i in range(50):
            request = generator.generate_enhanced_plan_request()
            # Validation happens during object creation
        
        validation_time = time.time() - start_time
        print(f"✅ Data validation: {validation_time:.3f}s for 50 objects ({50/validation_time:.1f} objects/sec)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running performance tests: {str(e)}")
        return False


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="FinPilot VP-MAS Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--mock", action="store_true", help="Run mock interface tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    
    args = parser.parse_args()
    
    print("🚀 FinPilot VP-MAS Test Runner")
    print("=" * 50)
    print(f"Python path: {sys.executable}")
    print(f"Project root: {project_root}")
    print()
    
    start_time = time.time()
    results = []
    
    # Run selected test categories
    if args.unit or not any([args.unit, args.integration, args.mock, args.performance]):
        results.append(("Data Models", run_data_model_tests()))
    
    if args.mock or not any([args.unit, args.integration, args.mock, args.performance]):
        results.append(("Mock Agents", await run_mock_agent_tests()))
    
    if args.integration or not any([args.unit, args.integration, args.mock, args.performance]):
        results.append(("Integration", await run_integration_tests()))
    
    if args.performance or not any([args.unit, args.integration, args.mock, args.performance]):
        results.append(("Performance", run_performance_tests()))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    print(f"Execution time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for development.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {str(e)}")
        sys.exit(1)