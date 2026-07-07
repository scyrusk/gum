import asyncio
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
from persistqueue import SQLiteAckQueue, Empty

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

        self.logger = logging.getLogger("gum.batcher")

        # Persistent, *acknowledged* queue backed by SQLite. Items are only
        # removed once ack()'d after a batch is durably processed; a crash or
        # restart mid-batch leaves them in the 'unack' state, and auto_resume
        # returns them to 'ready' on reopen. This is what makes progress survive
        # `gum stop` / restarts — a plain persist-queue Queue only commits its
        # read position at chunk boundaries and never task_done()'d, so it would
        # rewind and re-deliver already-processed observations on every restart.
        queue_dir = self.data_directory / "batches"
        queue_dir.mkdir(parents=True, exist_ok=True)
        self._queue = SQLiteAckQueue(path=str(queue_dir / "ackqueue"), auto_commit=True)
        self._migrate_legacy_queue(queue_dir)

        self._batch_ready_event = asyncio.Event()

        # Monotonic timestamp of the oldest un-popped observation, or None when
        # the queue is empty. Drives the idle-flush timer.
        self._oldest_pending: float | None = None
        self._timeout_task: asyncio.Task | None = None

    def _migrate_legacy_queue(self, queue_dir: Path) -> None:
        """One-time migration of any backlog from the old plain persist-queue
        ``Queue`` (``batches/queue``) into the ack queue, then retire the legacy
        directory so this runs only once. Best-effort — never fails startup.

        The legacy queue never called ``task_done()`` and only committed its
        read position at chunk boundaries, so its backlog was replayed on every
        restart. We drain whatever it still holds into the ack queue so no
        genuinely pending observation is dropped, at the cost of a one-time
        re-processing pass (duplicate drafts are absorbed downstream by the
        identical/similar proposition filter).
        """
        legacy_path = queue_dir / "queue"
        if not (legacy_path / "info").exists():
            return
        try:
            from persistqueue import Queue as LegacyQueue

            legacy = LegacyQueue(path=str(legacy_path))
            moved = 0
            while legacy.qsize() > 0:
                try:
                    self._queue.put(legacy.get_nowait())
                    moved += 1
                except Empty:
                    break
            del legacy  # release file handles before retiring the directory
            if moved:
                self.logger.info(
                    f"Migrated {moved} pending observation(s) from the legacy queue "
                    "into the ack queue (one-time catch-up)"
                )
            retired = queue_dir / "queue.legacy"
            if retired.exists():
                shutil.rmtree(retired, ignore_errors=True)
            legacy_path.rename(retired)
        except Exception as exc:
            self.logger.warning(f"Legacy queue migration skipped: {exc}")

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

        The returned items move to the queue's 'unack' state; the caller MUST
        later ``ack_batch`` them on success or ``nack_batch`` them on failure.
        Until then they survive a crash/restart (auto-resumed to 'ready').

        Args:
            batch_size: Number of items to pop. Defaults to max_batch_size

        Returns:
            List of observation dictionaries popped from queue
        """
        batch_size = batch_size or self.max_batch_size

        batch = []
        for _ in range(min(batch_size, self._queue.qsize())):
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break

        if batch:
            self.logger.debug(f"Popped batch of {len(batch)} observations (queue size: {self._queue.qsize()})")

        # Reset the idle-flush clock: restart it for any leftover items, or
        # clear it when the queue is drained.
        self._oldest_pending = time.monotonic() if self._queue.qsize() > 0 else None

        if not self.should_process_batch():
            self._batch_ready_event.clear()

        return batch

    def ack_batch(self, items: List[Dict[str, Any]]) -> None:
        """Acknowledge a durably-processed batch so its items are removed for
        good (they will not be re-delivered after a restart)."""
        for item in items:
            self._queue.ack(item)

    def nack_batch(self, items: List[Dict[str, Any]]) -> None:
        """Return an unprocessed batch (failed or interrupted by shutdown) to the
        queue for a later retry, re-using the same items (no duplicate IDs)."""
        for item in items:
            self._queue.nack(item)
        # The nacked items are 'ready' again: restart the idle-flush clock and
        # wake the processor if we're back at/above the batch threshold.
        if self._oldest_pending is None and self._queue.qsize() > 0:
            self._oldest_pending = time.monotonic()
        if self.should_process_batch():
            self._batch_ready_event.set()

    async def wait_for_batch_ready(self):
        """Wait for a batch to be ready for processing."""
        await self._batch_ready_event.wait()