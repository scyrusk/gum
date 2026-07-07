from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import asyncio

class Observer(ABC):
    """Base class for all observers in the GUM system.

    This abstract base class defines the interface for all observers that monitor user behavior.
    Observers are responsible for collecting data about user interactions and sending updates
    through an asynchronous queue.

    Args:
        name (Optional[str]): A custom name for the observer. If not provided, the class name will be used.

    Attributes:
        update_queue (asyncio.Queue): Queue for sending updates to the main GUM system.
        _name (str): The name of the observer.
        _running (bool): Flag indicating if the observer is currently running.
        _task (Optional[asyncio.Task]): Background task handle for the observer's worker.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.update_queue = asyncio.Queue()
        self._name = name or self.__class__.__name__

        # running flag + background task handle
        self._running = True
        self._task: asyncio.Task | None = asyncio.create_task(self._worker_wrapper())

    # ─────────────────────────────── abstract worker
    @abstractmethod
    async def _worker(self) -> None:     # subclasses override
        """Main worker method that must be implemented by subclasses.
        
        This method should contain the main logic for the observer, such as monitoring
        user interactions or collecting data. It runs in a background task and should
        continue running until the observer is stopped.
        """
        pass

    # wrapper plugs running flag + exception handling
    async def _worker_wrapper(self) -> None:
        """Wrapper for the worker method that handles exceptions and cleanup.
        
        This method ensures proper cleanup of resources when the worker stops,
        whether due to normal termination or an exception.
        """
        try:
            await self._worker()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            raise
        finally:
            self._running = False

    # ─────────────────────────────── public API
    @property
    def name(self) -> str:
        """Get the name of the observer.
        
        Returns:
            str: The observer's name.
        """
        return self._name

    @property
    def warm_targets(self) -> list[tuple[str, str]]:
        """``(api_base, model)`` pairs this observer wants Ollama to keep resident.

        Observers that drive a local model return the endpoint and model name so
        the GUM's keep-alive pinger can prevent it from unloading during idle
        periods. Non-model observers keep the default (nothing to pin).
        """
        return []

    async def get_update(self):
        """Get the next update from the queue if available.
        
        Returns:
            Optional[Update]: The next update from the queue, or None if the queue is empty.
        """
        try:
            return self.update_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop(self) -> None:
        """Stop the observer and clean up resources.
        
        This method cancels the worker task and drains the update queue.
        """
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # unblock any awaiters
        while not self.update_queue.empty():
            self.update_queue.get_nowait()
