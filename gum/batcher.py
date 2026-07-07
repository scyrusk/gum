import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
from persistqueue import Queue

class ObservationBatcher:
    """A persistent queue for batching observations to reduce API calls."""

    def __init__(
        self,
        data_directory: str,
        min_batch_size: int = 5,
        max_batch_size: int = 50,
        batch_timeout: float | None = None,
    ):
        self.data_directory = Path(data_directory)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Idle flush: process a partial batch (below min_batch_size) once its
        # oldest observation has waited this many seconds. Without this a
        # partial batch sits in the queue indefinitely during low activity, so
        # propositions never appear until enough new interactions accumulate.
        # Override with GUM_BATCH_TIMEOUT (seconds); set to 0 to disable.
        if batch_timeout is None:
            batch_timeout = float(os.getenv("GUM_BATCH_TIMEOUT", "30"))
        self.batch_timeout = batch_timeout

        # Create persistent queue backed by SQLite
        queue_dir = self.data_directory / "batches"
        queue_dir.mkdir(parents=True, exist_ok=True)
        self._queue = Queue(path=str(queue_dir / "queue"))

        self._batch_ready_event = asyncio.Event()
        self.logger = logging.getLogger("gum.batcher")

        # Monotonic timestamp of the oldest un-popped observation, or None when
        # the queue is empty. Drives the idle-flush timer.
        self._oldest_pending: float | None = None
        self._timeout_task: asyncio.Task | None = None

    async def start(self):
        """Start the batching system."""
        self.logger.info(f"Started batcher with {self._queue.qsize()} items in queue")

        if self._queue.qsize() > 0 and self._oldest_pending is None:
            self._oldest_pending = time.monotonic()

        if self.should_process_batch():
            self._batch_ready_event.set()

        # Launch the idle-flush timer so partial batches don't stall forever.
        if self.batch_timeout > 0 and self._timeout_task is None:
            self._timeout_task = asyncio.create_task(self._timeout_loop())

    async def stop(self):
        """Stop the batching system."""
        if self._timeout_task is not None:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None
        self.logger.info("Stopped batcher")

    async def _timeout_loop(self):
        """Flush a partial batch once its oldest observation exceeds the timeout."""
        # Poll at a fraction of the timeout so the effective flush delay stays
        # close to batch_timeout without a busy loop.
        interval = max(1.0, self.batch_timeout / 4)
        while True:
            await asyncio.sleep(interval)
            if self._batch_ready_event.is_set():
                continue
            if self._queue.qsize() == 0 or self._oldest_pending is None:
                continue
            if (time.monotonic() - self._oldest_pending) >= self.batch_timeout:
                self.logger.info(
                    f"Idle-flushing partial batch of {self._queue.qsize()} "
                    f"observations after {self.batch_timeout}s"
                )
                self._batch_ready_event.set()
        
    def push(self, observer_name: str, content: str, content_type: str) -> str:
        """Push an observation onto the queue.
        
        Args:
            observer_name: Name of the observer
            content: Observation content
            content_type: Type of content
            
        Returns:
            str: Observation ID
        """
        observation_id = str(uuid.uuid4())
        observation_dict = {
            'id': observation_id,
            'observer_name': observer_name,
            'content': content,
            'content_type': content_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add to queue - automatically persisted by persist-queue
        self._queue.put(observation_dict)
        self.logger.debug(f"Pushed observation {observation_id} to queue (size: {self._queue.qsize()})")

        # Start the idle-flush clock when the queue transitions to non-empty.
        if self._oldest_pending is None:
            self._oldest_pending = time.monotonic()

        # Signal that a batch is ready if we've reached minimum size
        if self.should_process_batch():
            self._batch_ready_event.set()

        return observation_id
        
    def size(self) -> int:
        """Get the current size of the queue."""
        return self._queue.qsize()
        
    def should_process_batch(self) -> bool:
        """Check if the batch should be processed based on minimum batch size."""
        return self._queue.qsize() >= self.min_batch_size
        
    def pop_batch(self, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Pop a batch of observations from the front of the queue (FIFO).
        
        Args:
            batch_size: Number of items to pop. Defaults to max_batch_size
            
        Returns:
            List of observation dictionaries popped from queue
        """
        batch_size = batch_size or self.max_batch_size
        
        batch = []
        for _ in range(min(batch_size, self._queue.qsize())):
            batch.append(self._queue.get_nowait())
        
        if batch:
            self.logger.debug(f"Popped batch of {len(batch)} observations (queue size: {self._queue.qsize()})")

        # Reset the idle-flush clock: restart it for any leftover items, or
        # clear it when the queue is drained.
        self._oldest_pending = time.monotonic() if self._queue.qsize() > 0 else None

        if not self.should_process_batch():
            self._batch_ready_event.clear()

        return batch
    
    async def wait_for_batch_ready(self):
        """Wait for a batch to be ready for processing."""
        await self._batch_ready_event.wait()