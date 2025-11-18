"""
Message Bus Port Interface

Defines the contract for inter-agent communication.
Infrastructure layer provides the actual implementation.

Created: Phase 1 - Foundation & Safety Net
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime


class IMessageBus(ABC):
    """
    Interface for message bus handling inter-agent communication.

    The message bus decouples agents from each other - they communicate
    only through messages, never direct method calls.
    """

    @abstractmethod
    async def send_message(
        self,
        message: Any,  # AgentMessage type
        timeout_seconds: Optional[float] = None
    ) -> bool:
        """
        Send a message to another agent.

        Args:
            message: Message to send (AgentMessage instance)
            timeout_seconds: Optional timeout for delivery

        Returns:
            True if message was successfully queued, False otherwise

        Raises:
            MessageBusError: If message bus is unavailable
            ValidationError: If message is invalid
        """
        pass

    @abstractmethod
    async def register_agent(
        self,
        agent_id: str,
        message_handler: Callable[[Any], Any],
        capabilities: List[str]
    ) -> None:
        """
        Register an agent with the message bus.

        Args:
            agent_id: Unique agent identifier
            message_handler: Async function to handle incoming messages
            capabilities: List of capabilities this agent provides

        Raises:
            RegistrationError: If agent_id already registered
        """
        pass

    @abstractmethod
    async def unregister_agent(
        self,
        agent_id: str
    ) -> None:
        """
        Unregister an agent from the message bus.

        Args:
            agent_id: Agent to unregister

        Raises:
            NotFoundError: If agent not registered
        """
        pass

    @abstractmethod
    def get_correlation_trace(
        self,
        correlation_id: str
    ) -> List[Any]:
        """
        Get all messages in a correlation trace.

        Useful for debugging and understanding workflow execution.

        Args:
            correlation_id: Correlation ID to trace

        Returns:
            List of messages with this correlation_id, in chronological order
        """
        pass

    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get message bus health and statistics.

        Returns:
            Dictionary containing:
            - status: str (healthy|degraded|unhealthy)
            - registered_agents: int
            - messages_processed: int
            - messages_failed: int
            - success_rate: float
            - average_latency_ms: float
            - active_circuits: int (circuit breakers)
        """
        pass

    @abstractmethod
    async def broadcast(
        self,
        message: Any,
        capability: Optional[str] = None
    ) -> int:
        """
        Broadcast message to multiple agents.

        Args:
            message: Message to broadcast
            capability: Optional capability filter (only send to agents with this capability)

        Returns:
            Number of agents the message was sent to
        """
        pass

    @abstractmethod
    def subscribe(
        self,
        agent_id: str,
        message_type: str,
        handler: Callable[[Any], Any]
    ) -> str:
        """
        Subscribe to specific message types.

        Args:
            agent_id: Agent subscribing
            message_type: Type of message to subscribe to
            handler: Handler function for this message type

        Returns:
            Subscription ID (for unsubscribing)
        """
        pass

    @abstractmethod
    def unsubscribe(
        self,
        subscription_id: str
    ) -> None:
        """
        Unsubscribe from message type.

        Args:
            subscription_id: ID from subscribe()
        """
        pass

    @abstractmethod
    async def request_reply(
        self,
        message: Any,
        timeout_seconds: float = 30.0
    ) -> Any:
        """
        Send a message and wait for reply.

        Implements request-reply pattern with timeout.

        Args:
            message: Request message
            timeout_seconds: How long to wait for reply

        Returns:
            Reply message

        Raises:
            TimeoutError: If no reply within timeout
            MessageBusError: If delivery fails
        """
        pass


class IMessageSerializer(ABC):
    """
    Interface for message serialization/deserialization.

    Allows message bus to be transport-agnostic.
    """

    @abstractmethod
    def serialize(self, message: Any) -> bytes:
        """
        Serialize message for transport.

        Args:
            message: Message to serialize

        Returns:
            Serialized bytes
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize message from transport.

        Args:
            data: Serialized message bytes

        Returns:
            Deserialized message object
        """
        pass


class ICircuitBreaker(ABC):
    """
    Interface for circuit breaker pattern.

    Prevents cascading failures when an agent is unhealthy.
    """

    @abstractmethod
    def call(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation through circuit breaker.

        Args:
            operation: Function to call
            *args, **kwargs: Arguments to operation

        Returns:
            Operation result

        Raises:
            CircuitOpenError: If circuit is open (too many failures)
        """
        pass

    @abstractmethod
    def get_state(self) -> str:
        """
        Get current circuit breaker state.

        Returns:
            "closed" | "open" | "half_open"
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        pass
