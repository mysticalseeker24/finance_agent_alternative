"""Base agent class for the Finance Agent system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

from loguru import logger


class BaseAgent(ABC):
    """Abstract base class for all agents in the Finance Agent system."""

    def __init__(self, name: str):
        """Initialize the base agent.

        Args:
            name: The name of the agent.
        """
        self.name = name
        self.last_run_time = None
        logger.info(f"{name} agent initialized")

    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request.

        Args:
            request: The request to process.

        Returns:
            The processed result.
        """
        pass

    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent.

        This method handles common functionality for all agents, such as logging,
        error handling, and timing.

        Args:
            request: The request to process.

        Returns:
            The processed result.
        """
        start_time = datetime.now()
        self.last_run_time = start_time

        logger.info(f"{self.name} agent started processing request")

        try:
            result = await self.process(request)

            # Add metadata to the result
            result["agent"] = self.name
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
            result["timestamp"] = datetime.now().isoformat()

            logger.info(
                f"{self.name} agent completed processing in {result['processing_time']:.2f} seconds"
            )
            return result

        except Exception as e:
            logger.error(f"{self.name} agent encountered an error: {str(e)}")
            return {
                "agent": self.name,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
            }

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the agent.

        Returns:
            A dictionary containing status information.
        """
        return {
            "agent": self.name,
            "status": "active",
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "timestamp": datetime.now().isoformat(),
        }
