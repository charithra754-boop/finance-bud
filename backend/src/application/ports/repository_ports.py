"""
Repository Port Interfaces

Defines contracts for data persistence.
Following Repository Pattern - abstracts data storage details.

Created: Phase 1 - Foundation & Safety Net
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime


class IRepository(ABC):
    """Base repository interface with common CRUD operations"""

    @abstractmethod
    async def save(self, entity: Any) -> str:
        """
        Save an entity.

        Args:
            entity: Entity to save

        Returns:
            Entity ID

        Raises:
            RepositoryError: If save fails
        """
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[Any]:
        """
        Retrieve entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def update(self, entity: Any) -> bool:
        """
        Update an existing entity.

        Args:
            entity: Entity with updated data

        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: Entity to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists.

        Args:
            entity_id: Entity ID to check

        Returns:
            True if exists, False otherwise
        """
        pass


class IPlanRepository(IRepository):
    """
    Repository for financial plans.

    Handles persistence of plans with versioning and history.
    """

    @abstractmethod
    async def save_plan(
        self,
        plan: Dict[str, Any],
        user_id: str,
        session_id: str
    ) -> str:
        """
        Save a financial plan.

        Args:
            plan: Plan data
            user_id: User who owns the plan
            session_id: Session that created the plan

        Returns:
            Plan ID
        """
        pass

    @abstractmethod
    async def get_plan(
        self,
        plan_id: str,
        version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a plan by ID.

        Args:
            plan_id: Plan identifier
            version: Optional specific version (defaults to latest)

        Returns:
            Plan data if found
        """
        pass

    @abstractmethod
    async def get_plans_for_user(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all plans for a user.

        Args:
            user_id: User identifier
            status: Optional status filter (active|completed|cancelled)
            limit: Maximum number of plans to return

        Returns:
            List of plans
        """
        pass

    @abstractmethod
    async def get_plan_history(
        self,
        plan_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a plan.

        Args:
            plan_id: Plan identifier

        Returns:
            List of plan versions, newest first
        """
        pass

    @abstractmethod
    async def update_plan_status(
        self,
        plan_id: str,
        new_status: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Update plan status.

        Args:
            plan_id: Plan to update
            new_status: New status value
            reason: Optional reason for status change

        Returns:
            True if updated
        """
        pass

    @abstractmethod
    async def create_plan_version(
        self,
        plan_id: str,
        updated_plan: Dict[str, Any],
        change_reason: str
    ) -> int:
        """
        Create a new version of an existing plan.

        Args:
            plan_id: Plan to version
            updated_plan: Updated plan data
            change_reason: Why the plan changed

        Returns:
            New version number
        """
        pass


class IFinancialStateRepository(IRepository):
    """
    Repository for user financial state.

    Handles persistence of current financial state with historical tracking.
    """

    @abstractmethod
    async def save_state(
        self,
        user_id: str,
        financial_state: Dict[str, Any]
    ) -> str:
        """
        Save current financial state for a user.

        Args:
            user_id: User identifier
            financial_state: Current financial state

        Returns:
            State snapshot ID
        """
        pass

    @abstractmethod
    async def get_current_state(
        self,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get current financial state for a user.

        Args:
            user_id: User identifier

        Returns:
            Current financial state if exists
        """
        pass

    @abstractmethod
    async def get_state_history(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical financial states for a user.

        Args:
            user_id: User identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum snapshots to return

        Returns:
            List of financial state snapshots
        """
        pass

    @abstractmethod
    async def get_state_at_time(
        self,
        user_id: str,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get financial state at a specific point in time.

        Args:
            user_id: User identifier
            timestamp: Point in time to query

        Returns:
            Financial state at that time (nearest snapshot before timestamp)
        """
        pass


class IExecutionLogRepository(IRepository):
    """
    Repository for execution logs and audit trail.
    """

    @abstractmethod
    async def log_execution(
        self,
        plan_id: str,
        step_id: str,
        execution_data: Dict[str, Any],
        status: str,
        result: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a plan step execution.

        Args:
            plan_id: Plan being executed
            step_id: Step being executed
            execution_data: Execution context
            status: Execution status
            result: Execution result if completed

        Returns:
            Log entry ID
        """
        pass

    @abstractmethod
    async def get_execution_logs(
        self,
        plan_id: str,
        step_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution logs for a plan.

        Args:
            plan_id: Plan identifier
            step_id: Optional specific step

        Returns:
            List of execution log entries
        """
        pass

    @abstractmethod
    async def get_audit_trail(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get complete audit trail for a user.

        Args:
            user_id: User identifier
            start_date: Optional date range start
            end_date: Optional date range end

        Returns:
            Chronological audit trail
        """
        pass


class IWorkflowRepository(IRepository):
    """
    Repository for workflow sessions and state.
    """

    @abstractmethod
    async def create_workflow(
        self,
        workflow_data: Dict[str, Any]
    ) -> str:
        """
        Create a new workflow session.

        Args:
            workflow_data: Initial workflow data

        Returns:
            Workflow ID
        """
        pass

    @abstractmethod
    async def get_workflow(
        self,
        workflow_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get workflow by ID.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow data if found
        """
        pass

    @abstractmethod
    async def update_workflow_state(
        self,
        workflow_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """
        Update workflow state.

        Args:
            workflow_id: Workflow to update
            state: New state data

        Returns:
            True if updated
        """
        pass

    @abstractmethod
    async def get_active_workflows(
        self,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all active workflows.

        Args:
            user_id: Optional filter by user

        Returns:
            List of active workflows
        """
        pass


class IUnitOfWork(ABC):
    """
    Unit of Work pattern for transactional operations.

    Ensures multiple repository operations succeed or fail together.
    """

    @abstractmethod
    async def __aenter__(self):
        """Start a transaction"""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Commit or rollback transaction"""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit the current transaction"""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the current transaction"""
        pass

    @property
    @abstractmethod
    def plans(self) -> IPlanRepository:
        """Access plan repository"""
        pass

    @property
    @abstractmethod
    def financial_states(self) -> IFinancialStateRepository:
        """Access financial state repository"""
        pass

    @property
    @abstractmethod
    def execution_logs(self) -> IExecutionLogRepository:
        """Access execution log repository"""
        pass

    @property
    @abstractmethod
    def workflows(self) -> IWorkflowRepository:
        """Access workflow repository"""
        pass
