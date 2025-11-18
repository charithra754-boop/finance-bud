"""
Agent Port Interfaces

Defines contracts that all agent implementations must follow.
Infrastructure layer implements these interfaces.

Following Interface Segregation Principle - specific interfaces
for each agent type rather than one large interface.

Created: Phase 1 - Foundation & Safety Net
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime


class IAgent(ABC):
    """
    Base interface for all agents in the VP-MAS system.

    All agents must implement this core contract.
    Specific agent types extend this with additional methods.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent"""
        pass

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Type of agent (orchestration, planning, verification, etc.)"""
        pass

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """Health check - is this agent functioning properly?"""
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        Initialize and start the agent.

        Should set up any necessary resources (connections, caches, etc.)
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Gracefully shut down the agent.

        Should clean up resources and complete in-flight work.
        """
        pass

    @abstractmethod
    async def process_message(self, message: Any) -> Any:
        """
        Process an incoming message.

        Args:
            message: AgentMessage from another agent

        Returns:
            Response message or None

        Raises:
            AgentProcessingError: If message processing fails
        """
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status for monitoring.

        Returns:
            Dictionary with status information:
            - status: "healthy" | "degraded" | "unhealthy"
            - uptime_seconds: float
            - messages_processed: int
            - error_count: int
            - last_activity: datetime
        """
        pass


class IPlanningAgent(IAgent):
    """
    Interface for Planning Agent.

    Responsible for generating financial plans using GSM and ToS algorithms.
    """

    @abstractmethod
    async def generate_plan(
        self,
        user_goal: str,
        financial_state: Dict[str, Any],
        constraints: List[Dict[str, Any]],
        risk_profile: Dict[str, Any],
        time_horizon_months: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a financial plan for the given goal and constraints.

        Args:
            user_goal: Natural language description of the goal
            financial_state: Current financial state
            constraints: List of constraints to satisfy
            risk_profile: User's risk tolerance and preferences
            time_horizon_months: Planning time horizon

        Returns:
            Dictionary containing:
            - plan_id: str
            - steps: List[PlanStep]
            - reasoning: ReasoningTrace
            - confidence_score: float
            - estimated_success_probability: float
        """
        pass

    @abstractmethod
    async def adjust_plan(
        self,
        plan_id: str,
        trigger_type: str,
        trigger_context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Adjust existing plan based on trigger (market changes, life events).

        Args:
            plan_id: ID of plan to adjust
            trigger_type: Type of trigger causing adjustment
            trigger_context: Context about the trigger

        Returns:
            Updated plan with adjustments applied
        """
        pass

    @abstractmethod
    async def evaluate_plan_feasibility(
        self,
        plan_steps: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate if a plan is feasible given constraints.

        Args:
            plan_steps: Steps in the plan
            constraints: Constraints to check against

        Returns:
            Dictionary containing:
            - is_feasible: bool
            - constraint_violations: List[str]
            - feasibility_score: float
        """
        pass


class IOrchestrationAgent(IAgent):
    """
    Interface for Orchestration Agent.

    Responsible for workflow coordination, session management,
    and inter-agent communication.
    """

    @abstractmethod
    async def submit_goal(
        self,
        user_goal: str,
        user_id: Optional[str] = None,
        priority: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Submit a new financial goal for processing.

        Args:
            user_goal: Natural language goal description
            user_id: Optional user identifier
            priority: Priority level (low|medium|high|critical)

        Returns:
            Dictionary containing:
            - session_id: str
            - workflow_id: str
            - status: str
            - estimated_completion_time: datetime
        """
        pass

    @abstractmethod
    async def get_workflow_status(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """
        Get status of an active workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Dictionary containing:
            - workflow_id: str
            - status: str (pending|in_progress|completed|failed)
            - current_step: str
            - progress_percentage: float
            - steps_completed: List[str]
        """
        pass

    @abstractmethod
    async def activate_cmvl(
        self,
        session_id: str,
        trigger_event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Activate Continuous Monitoring and Verification Loop.

        Args:
            session_id: Session to activate CMVL for
            trigger_event: Event that triggered CMVL activation

        Returns:
            CMVL activation result
        """
        pass


class IVerificationAgent(IAgent):
    """
    Interface for Verification Agent.

    Responsible for verifying plans against constraints
    and ensuring regulatory compliance.
    """

    @abstractmethod
    async def verify_plan(
        self,
        plan_id: str,
        plan_steps: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Verify a financial plan against constraints.

        Args:
            plan_id: Plan identifier
            plan_steps: Steps in the plan
            constraints: Constraints to verify against

        Returns:
            Dictionary containing:
            - verification_id: str
            - status: str (approved|rejected|conditional)
            - violations: List[Dict[str, Any]]
            - recommendations: List[str]
            - confidence_score: float
        """
        pass

    @abstractmethod
    async def check_compliance(
        self,
        plan_steps: List[Dict[str, Any]],
        regulatory_requirements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check plan compliance with regulatory requirements.

        Args:
            plan_steps: Steps to check
            regulatory_requirements: Applicable regulations

        Returns:
            Compliance check result
        """
        pass


class IExecutionAgent(IAgent):
    """
    Interface for Execution Agent.

    Responsible for executing approved plan steps.
    """

    @abstractmethod
    async def execute_step(
        self,
        plan_id: str,
        step_id: str,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single plan step.

        Args:
            plan_id: Plan identifier
            step_id: Step to execute
            execution_context: Context for execution

        Returns:
            Execution result
        """
        pass

    @abstractmethod
    async def get_execution_status(
        self,
        execution_id: str
    ) -> Dict[str, Any]:
        """
        Get status of an execution.

        Args:
            execution_id: Execution identifier

        Returns:
            Execution status and results
        """
        pass


class IInformationRetrievalAgent(IAgent):
    """
    Interface for Information Retrieval Agent.

    Responsible for retrieving market data and external information.
    """

    @abstractmethod
    async def get_market_data(
        self,
        symbols: List[str],
        data_type: str = "latest",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve market data for given symbols.

        Args:
            symbols: List of ticker symbols
            data_type: Type of data (latest|historical|realtime)

        Returns:
            Market data dictionary
        """
        pass

    @abstractmethod
    async def detect_triggers(
        self,
        monitored_assets: List[str],
        trigger_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect market triggers based on conditions.

        Args:
            monitored_assets: Assets to monitor
            trigger_conditions: Conditions that constitute a trigger

        Returns:
            List of detected triggers
        """
        pass


class IConversationalAgent(IAgent):
    """
    Interface for Conversational Agent.

    Responsible for natural language processing and goal parsing.
    """

    @abstractmethod
    async def parse_goal(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language goal into structured format.

        Args:
            user_input: Natural language goal description
            context: Optional conversation context

        Returns:
            Structured goal representation
        """
        pass

    @abstractmethod
    async def generate_narrative(
        self,
        plan: Dict[str, Any],
        reasoning: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable narrative from plan and reasoning.

        Args:
            plan: Financial plan
            reasoning: Reasoning trace

        Returns:
            Natural language narrative
        """
        pass
