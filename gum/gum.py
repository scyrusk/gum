# gum.py

from __future__ import annotations

import asyncio
import logging
import os
import time
from uuid import uuid4
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Callable, List
from .models import observation_proposition
import traceback

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from .db_utils import (
    get_related_observations,
    search_propositions_bm25,
    get_recent_propositions,
    count_observations,
    get_recent_observations,
    get_next_unreviewed_proposition,
    add_proposition_feedback,
    get_recent_feedback,
    select_relevant_balanced_feedback,
    count_review_progress,
)
from .llm import (
    make_client,
    structured_completion,
    resolve_api_base,
    keep_models_warm,
    release_models,
    keep_warm_enabled,
    resolve_keep_alive,
    resolve_keep_warm_interval,
    resolve_text_idle_unload,
)
from .models import Observation, Proposition, init_db
from .observers import Observer
from .schemas import (
    PropositionItem,
    PropositionSchema,
    RelationSchema,
    Update,
    AuditSchema
)
from gum.prompts.gum import AUDIT_PROMPT, PROPOSE_PROMPT, REVISE_PROMPT, SIMILAR_PROMPT
from .batcher import ObservationBatcher

class gum:
    """A class for managing general user models.

    This class provides functionality for observing user behavior, generating and managing
    propositions about user behavior, and maintaining relationships between observations
    and propositions.

    Args:
        user_name (str): The name of the user being modeled.
        *observers (Observer): Variable number of observer instances to track user behavior.
        propose_prompt (str, optional): Custom prompt for proposition generation.
        similar_prompt (str, optional): Custom prompt for similarity analysis.
        revise_prompt (str, optional): Custom prompt for proposition revision.
        audit_prompt (str, optional): Custom prompt for auditing.
        data_directory (str, optional): Directory for storing data. Defaults to "~/.cache/gum".
        db_name (str, optional): Name of the database file. Defaults to "gum.db".

        verbosity (int, optional): Logging verbosity level. Defaults to logging.INFO.
        audit_enabled (bool, optional): Whether to enable auditing. Defaults to False.
    """

    def __init__(
        self,
        user_name: str,
        model: str,
        *observers: Observer,
        propose_prompt: str | None = None,
        similar_prompt: str | None = None,
        revise_prompt: str | None = None,
        audit_prompt: str | None = None,
        data_directory: str = "~/.cache/gum",
        db_name: str = "gum.db",
        verbosity: int = logging.INFO,
        audit_enabled: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        min_batch_size: int = 5,
        max_batch_size: int = 50,
        batch_timeout: float | None = None,
    ):
        # basic paths
        data_directory = os.path.expanduser(data_directory)
        os.makedirs(data_directory, exist_ok=True)

        # runtime
        self.user_name = user_name
        self.observers: list[Observer] = list(observers)
        self.model = model
        self.audit_enabled = audit_enabled

        # batching configuration
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # When a batch's drafts match no existing persisted propositions, the
        # relation ("SIMILAR") 32B call can only dedup drafts against each other,
        # a low-value use of a full text-model call on the local GPU. Skip it in
        # that case and treat every draft as new. Override with
        # GUM_SKIP_SIMILAR_WHEN_NEW=0 to always run the relation call.
        self.skip_similar_when_new = os.getenv("GUM_SKIP_SIMILAR_WHEN_NEW", "1") != "0"

        # logging
        self.logger = logging.getLogger("gum")
        self.logger.setLevel(verbosity)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)

        # prompts
        self.propose_prompt = propose_prompt or PROPOSE_PROMPT
        self.similar_prompt = similar_prompt or SIMILAR_PROMPT
        self.revise_prompt = revise_prompt or REVISE_PROMPT
        self.audit_prompt = audit_prompt or AUDIT_PROMPT

        # Local-first inference client (defaults to Ollama; refuses non-local
        # endpoints unless GUM_ALLOW_REMOTE=1). See gum/llm.py.
        self._api_base = resolve_api_base("gum", api_base)
        self.client = make_client("gum", api_base=api_base, api_key=api_key)

        self.engine = None
        self.Session = None
        self._db_name        = db_name
        self._data_directory = data_directory

        # Initialize batcher if enabled
        self.batcher = ObservationBatcher(
            data_directory=data_directory,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
        )

        self._loop_task: asyncio.Task | None = None
        self._batch_task: asyncio.Task | None = None
        self._warm_task: asyncio.Task | None = None
        self._batch_processing_lock = asyncio.Lock()
        # Monotonic timestamp of the last observation, used to decide when the
        # text model may be released from memory during quiet periods.
        self._last_activity = time.monotonic()
        self._text_idle_unload = resolve_text_idle_unload()
        self.update_handlers: list[Callable[[Observer, Update], None]] = [self._default_handler]

    def _vision_warm_targets(self) -> list[tuple[str, str]]:
        """The observers' resident models (e.g. the screen vision model)."""
        targets: list[tuple[str, str]] = []
        for obs in self.observers:
            targets.extend(obs.warm_targets)
        return targets

    def _all_warm_targets(self) -> list[tuple[str, str]]:
        """Every local model the GUM keeps warm: observer vision + text model."""
        return self._vision_warm_targets() + [(self._api_base, self.model)]

    def start_update_loop(self):
        """Start the asynchronous update loop for processing observer updates."""
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._update_loop())
            
        # Start batch processing if enabled
        if self._batch_task is None:
            self._batch_task = asyncio.create_task(self._batch_processing_loop())

        # Keep the local models resident so an idle pause doesn't unload them
        # and stall the next observation on a cold reload (see keep_models_warm).
        # The observers' models (e.g. the screen vision model) are pinned
        # unconditionally — they run on nearly every interaction and are cheap to
        # keep warm. The text model is activity-gated: it's the biggest resident
        # cost and only runs on batches, so it's released after a quiet spell
        # (GUM_TEXT_IDLE_UNLOAD) and reloaded on the next batch.
        if self._warm_task is None and keep_warm_enabled():
            vision_targets = self._vision_warm_targets()
            text_targets = [(self._api_base, self.model)]

            gate = self._text_idle_unload
            is_active = None if gate <= 0 else (
                lambda: (time.monotonic() - self._last_activity) < gate
            )

            self._warm_task = asyncio.create_task(
                keep_models_warm(
                    vision_targets,
                    gated_targets=text_targets,
                    is_active=is_active,
                    keep_alive=resolve_keep_alive(),
                    interval=resolve_keep_warm_interval(),
                    logger=self.logger,
                )
            )

    async def stop_update_loop(self):
        """Stop the asynchronous update loop and clean up resources."""
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
            
        # Stop batch processing if enabled
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
            self._batch_task = None

        # Stop the keep-alive pinger. Cancelling it alone would leave the models
        # pinned until their finite keep_alive window lapses (up to ~15m), so
        # eagerly release them here — a keep_alive=0 ping unloads them from VRAM
        # right away. This runs even when the pinger was disabled: models loaded
        # by normal inference otherwise linger on Ollama's default timer.
        if self._warm_task:
            self._warm_task.cancel()
            try:
                await self._warm_task
            except asyncio.CancelledError:
                pass
            self._warm_task = None

        try:
            await release_models(self._all_warm_targets(), logger=self.logger)
        except Exception:  # pragma: no cover - shutdown must not raise
            self.logger.debug("keep-warm: model release on shutdown failed", exc_info=True)

        if self.batcher:
            await self.batcher.stop()

    async def connect_db(self):
        """Initialize the database connection if not already connected."""
        if self.engine is None:
            self.engine, self.Session = await init_db(
                self._db_name, self._data_directory
            )

    async def __aenter__(self):
        """Async context manager entry point.
        
        Returns:
            gum: The instance of the gum class.
        """
        await self.connect_db()
        self.start_update_loop()
        
        # Start batcher if enabled
        if self.batcher:
            await self.batcher.start()
            
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit point.
        
        Args:
            exc_type: The type of exception if any.
            exc: The exception instance if any.
            tb: The traceback if any.
        """
        await self.stop_update_loop()

        # stop observers
        for obs in self.observers:
            await obs.stop()

    async def _update_loop(self):
        """Efficiently wait for any observer to produce an Update and dispatch it.

        This method continuously monitors all observers for updates and processes them
        through the semaphore-guarded handler.
        """
        # Keep exactly one outstanding ``update_queue.get()`` per observer,
        # persisted across loop iterations. The previous implementation rebuilt a
        # get() task for every observer on each pass and discarded the
        # un-completed ones WITHOUT cancelling them; with more than one observer
        # those orphaned getters stayed registered on their queues and would
        # eventually consume a later Update whose result nobody awaited —
        # silently dropping observations. A long-lived getter per observer fixes
        # that loss and also removes the per-iteration task churn.
        pending: dict[asyncio.Task, Observer] = {}
        try:
            while True:
                # (Re)arm a getter for any observer that doesn't currently have
                # one outstanding — new observers, or ones whose getter just
                # completed and was popped below.
                armed = set(pending.values())
                for obs in self.observers:
                    if obs not in armed:
                        pending[asyncio.create_task(obs.update_queue.get())] = obs

                if not pending:
                    # No observers registered yet; yield briefly and re-check.
                    await asyncio.sleep(0.05)
                    continue

                done, _ = await asyncio.wait(
                    pending.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for fut in done:
                    obs = pending.pop(fut)
                    upd: Update = fut.result()

                    for handler in self.update_handlers:
                        asyncio.create_task(handler(obs, upd))
        finally:
            # Cancel any still-outstanding getters so they don't leak when the
            # loop is cancelled on shutdown.
            for fut in pending:
                fut.cancel()

    async def _batch_processing_loop(self):
        """Process batched observations when minimum batch size is reached."""
        while True:
            # Wait for batch to be ready (event-driven, no polling!)
            await self.batcher.wait_for_batch_ready()
            
            # Use lock to ensure batch processing runs synchronously
            async with self._batch_processing_lock:
                batch = self.batcher.pop_batch()
                # A batch needs the text model; count it as activity so the
                # pinger won't release the model out from under a running batch.
                self._last_activity = time.monotonic()
                self.logger.info(f"Processing batch of {len(batch)} observations")
                await self._process_batch(batch)

    async def _process_batch(self, batched_observations):
        """Process a batch of observations together to reduce API calls."""
        
        # Combine all observations into a single content for analysis
        combined_content = []
        observation_ids = []
        
        for obs in batched_observations:
            combined_content.append(f"[{obs['observer_name']}] {obs['content']}")
            observation_ids.append(obs['id'])
            
        combined_text = "\n\n".join(combined_content)
        
        # Create a combined update
        combined_update = Update(
            content=combined_text,
            content_type="input_text"
        )
        
        try:
            async with self._session() as session:
                # Create observations in database
                observations = []
                for obs in batched_observations:
                    observation = Observation(
                        observer_name=obs['observer_name'],
                        content=obs['content'],
                        content_type=obs['content_type'],
                    )
                    session.add(observation)
                    observations.append(observation)
                
                await session.flush()
                
                # Process the combined content
                drafts, existing = await self._generate_and_search(session, combined_update)
                if self.skip_similar_when_new and not existing:
                    # Nothing persisted matched these drafts, so the relation
                    # call would only compare drafts against each other. Skip
                    # that 32B call and treat each draft as a new proposition.
                    identical, similar, different = [], [], drafts
                else:
                    identical, similar, different = await self._filter_propositions(
                        existing + drafts
                    )

                self.logger.info("Applying proposition updates for batch...")
                await self._handle_identical(session, identical, observations)
                await self._handle_similar(session, similar, observations)
                await self._handle_different(session, different, observations)

                self.logger.info(f"Completed processing batch of {len(batched_observations)} observations")

            # The session above committed on clean exit, so the observations and
            # propositions are durably persisted — only now is it safe to ack and
            # permanently remove these items from the queue.
            self.batcher.ack_batch(batched_observations)

        except asyncio.CancelledError:
            # Shutdown interrupted the batch: return the items to the queue so
            # they're re-processed next run instead of being stranded/dropped.
            self.batcher.nack_batch(batched_observations)
            raise
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.logger.error(f"Batch size: {len(batched_observations)}")
            if batched_observations:
                self.logger.error(f"First observation type: {type(batched_observations[0])}")
                self.logger.error(f"First observation: {batched_observations[0]}")
            # Bounded retry (same items, no dup IDs): retries with idle-flush
            # backoff, then sets a persistently-failing poison batch aside as
            # 'failed' so it can't hot-loop and starve newer observations.
            self.batcher.fail_batch(batched_observations)

    async def _construct_propositions(self, update: Update) -> list[PropositionItem]:
        """Generate propositions from an update.
        
        Args:
            update (Update): The update to generate propositions from.
            
        Returns:
            list[PropositionItem]: List of generated propositions.
        """
        prompt = (
            self.propose_prompt.replace("{user_name}", self.user_name)
            .replace("{feedback_examples}", await self._build_feedback_examples(update.content))
            .replace("{inputs}", update.content)
        )

        result = await structured_completion(
            self.client,
            self.model,
            [{"role": "user", "content": prompt}],
            PropositionSchema,
            logger=self.logger,
        )
        return result.propositions

    async def _build_feedback_examples(self, query_text: str = "") -> str:
        """Format the user's judgments as a calibration block for the propose
        prompt, choosing examples that are relevant to the current activity
        (*query_text*) and balanced across ratings.

        Empty string when there's no feedback yet or GUM_FEWSHOT_LIMIT=0, so the
        prompt is unchanged until the user reviews.
        """
        try:
            limit = int(os.getenv("GUM_FEWSHOT_LIMIT", "8"))
        except ValueError:
            limit = 8
        if limit <= 0:
            return ""

        try:
            pool = int(os.getenv("GUM_FEWSHOT_POOL", "200"))
        except ValueError:
            pool = 200

        # Pull a recent candidate pool, then pick the relevant + balanced subset.
        candidates = await self.recent_feedback(limit=max(pool, limit))
        feedback = select_relevant_balanced_feedback(candidates, query_text, limit)
        if not feedback:
            return ""

        label = {
            "accurate": "ACCURATE",
            "partial": "PARTIALLY ACCURATE",
            "inaccurate": "INACCURATE",
        }
        lines = []
        for fb in feedback:
            line = f"- [{label.get(fb.rating, fb.rating.upper())}] {fb.proposition_text}"
            if fb.note:
                line += f"\n    (context from {self.user_name}: {fb.note})"
            lines.append(line)
        examples = "\n".join(lines)
        return (
            f"# Calibration from {self.user_name}'s feedback\n\n"
            f"{self.user_name} has personally reviewed earlier propositions about themselves and rated each "
            f"as ACCURATE, PARTIALLY ACCURATE, or INACCURATE — sometimes adding context. Learn from these: "
            f"generate propositions in the spirit of the ACCURATE examples, refine toward the context given on "
            f"PARTIALLY ACCURATE ones, and avoid the kinds of unsupported or incorrect inferences shown in the "
            f"INACCURATE examples.\n\n"
            f"{examples}\n"
        )

    async def _build_relation_prompt(self, all_props) -> str:
        """Build a prompt for analyzing relationships between propositions.
        
        Args:
            all_props: List of propositions to analyze.
            
        Returns:
            str: The formatted prompt for relationship analysis.
        """
        blocks = [
            f"[id={p['id']}] {p['proposition']}\n    Reasoning: {p['reasoning']}"
            for p in all_props
        ]
        body = "\n\n".join(blocks)
        return self.similar_prompt.replace("{body}", body)

    async def _filter_propositions(
        self, rel_props: list[Proposition]
    ) -> tuple[list[Proposition], list[Proposition], list[Proposition]]:
        """Filter propositions into identical, similar, and unrelated groups.
        
        Args:
            rel_props (list[Proposition]): List of propositions to filter.
            
        Returns:
            tuple[list[Proposition], list[Proposition], list[Proposition]]: Three lists containing
                identical, similar, and unrelated propositions respectively.
        """
        if not rel_props:
            return [], [], []

        payload = [
            {"id": p.id, "proposition": p.text, "reasoning": p.reasoning or ""}
            for p in rel_props
        ]
        prompt_text = await self._build_relation_prompt(payload)

        data = await structured_completion(
            self.client,
            self.model,
            [{"role": "user", "content": prompt_text}],
            RelationSchema,
            logger=self.logger,
        )

        id_to_prop = {p.id: p for p in rel_props}
        ident, sim, unrel = set(), set(), set()

        for r in data.relations:
            if r.label == "IDENTICAL":
                ident.add(r.source)
                ident.update(r.target or [])
            elif r.label == "SIMILAR":
                sim.add(r.source)
                sim.update(r.target or [])
            else:
                unrel.add(r.source)

        # only keep IDs we actually know about
        valid_ids = set(id_to_prop.keys())
        ident &= valid_ids
        sim &= valid_ids
        unrel &= valid_ids

        return (
            [id_to_prop[i] for i in ident],
            [id_to_prop[i] for i in sim - ident],
            [id_to_prop[i] for i in unrel - ident - sim],
        )

    async def _build_revision_body(
        self, similar: List[Proposition], related_obs: List[Observation]
    ) -> str:
        """Build the body text for proposition revision.
        
        Args:
            similar (List[Proposition]): List of similar propositions.
            related_obs (List[Observation]): List of related observations.
            
        Returns:
            str: The formatted body text for revision.
        """
        blocks = [
            f"Proposition {idx}: {p.text}\nReasoning: {p.reasoning}"
            for idx, p in enumerate(similar, 1)
        ]
        if related_obs:
            blocks.append("\nSupporting observations:")
            blocks.extend(f"- {o.content}" for o in related_obs[:10])
        return "\n".join(blocks)

    async def _revise_propositions(
        self,
        related_obs: list[Observation],
        similar_cluster: list[Proposition],
    ) -> list[PropositionItem]:
        """Revise propositions based on related observations and similar propositions.
        
        Args:
            related_obs (list[Observation]): List of related observations.
            similar_cluster (list[Proposition]): List of similar propositions.
            
        Returns:
            list[dict]: List of revised propositions.
        """
        body = await self._build_revision_body(similar_cluster, related_obs)
        prompt = self.revise_prompt.replace("{body}", body)
        result = await structured_completion(
            self.client,
            self.model,
            [{"role": "user", "content": prompt}],
            PropositionSchema,
            logger=self.logger,
        )
        return result.propositions

    async def _generate_and_search(
        self, session: AsyncSession, update: Update
    ) -> tuple[list[Proposition], list[Proposition]]:
        """Generate draft propositions and find related persisted ones.

        Returns:
            tuple[list[Proposition], list[Proposition]]: the freshly created
            draft propositions, and the distinct already-persisted propositions
            matched by BM25 search against those drafts.
        """

        drafts_raw = await self._construct_propositions(update)
        drafts: list[Proposition] = []
        existing: dict[int, Proposition] = {}

        for itm in drafts_raw:
            draft = Proposition(
                text=itm.proposition,
                reasoning=itm.reasoning,
                confidence=itm.confidence,
                decay=itm.decay,
                revision_group=str(uuid4()),
                version=1,
            )
            drafts.append(draft)

            # search existing persisted props
            with session.no_autoflush:
                hits = await search_propositions_bm25(
                    session, f"{draft.text}\n{draft.reasoning}", mode="OR",
                    include_observations=False,
                    enable_mmr=False,
                    enable_decay=True
                )

            for prop, _score in hits:
                existing[prop.id] = prop

        session.add_all(drafts)
        await session.flush()

        return drafts, list(existing.values())

    async def _handle_identical(
        self, session, identical: list[Proposition], observations: list[Observation]
    ) -> None:
        for p in identical:
            for obs in observations:
                await self._attach_obs_if_missing(p, obs, session)

    async def _handle_similar(
        self,
        session: AsyncSession,
        similar: list[Proposition],
        observations: list[Observation],
    ) -> None:

        if not similar:
            return

        # Collect all observations from similar propositions
        rel_obs = {
            o
            for p in similar
            for o in await get_related_observations(session, p.id)
        }
        # Add all the batched observations
        rel_obs.update(observations)

        # Generate revised propositions
        revised_items = await self._revise_propositions(list(rel_obs), similar)
        
        # Delete all old similar propositions
        for prop in similar:
            await session.delete(prop)
        
        # Create new propositions to replace them
        revision_group = str(uuid4())
        for item in revised_items:
            new_prop = Proposition(
                text=item.proposition,
                reasoning=item.reasoning,
                confidence=item.confidence,
                decay=item.decay,
                version=1,  # Start fresh with version 1
                revision_group=revision_group,
                observations=rel_obs,
            )
            session.add(new_prop)

        await session.flush()

    async def _handle_different(
        self, session, different: list[Proposition], observations: list[Observation]
    ) -> None:
        for p in different:
            for obs in observations:
                await self._attach_obs_if_missing(p, obs, session)

    async def _handle_audit(self, obs: Observation) -> bool:
        if not self.audit_enabled:
            return False

        hits = await self.query(obs.content, limit=10, mode="OR")

        if not hits:
            past_interaction = "*None*"
        else:
            ctx_chunks: list[str] = []
            async with self._session() as session:
                for prop, score in hits:
                    chunk = [f"• {prop.text}"]
                    if prop.reasoning:
                        chunk.append(f"  Reasoning: {prop.reasoning}")
                    if prop.confidence is not None:
                        chunk.append(f"  Confidence: {prop.confidence}")
                    chunk.append(f"  Relevance Score: {score:.2f}")

                    obs_list = await get_related_observations(session, prop.id)
                    if obs_list:
                        chunk.append("  Supporting Observations:")
                        for rel_obs in obs_list:
                            preview = rel_obs.content.replace("\n", " ")[:120]
                            chunk.append(f"    - [{rel_obs.observer_name}] {preview}")

                    ctx_chunks.append("\n".join(chunk))

            past_interaction = "\n\n".join(ctx_chunks)

        prompt = (
            self.audit_prompt
            .replace("{past_interaction}", past_interaction)
            .replace("{user_input}", obs.content)
            .replace("{user_name}", self.user_name)
        )

        decision = await structured_completion(
            self.client,
            self.model,
            [{"role": "user", "content": prompt}],
            AuditSchema,
            temperature=0.0,
            logger=self.logger,
        )

        if not decision.transmit_data:
            self.logger.warning(
                "Audit blocked transmission (data_type=%s, subject=%s)",
                decision.data_type,
                decision.subject,
            )
            return True

        return False

    async def _default_handler(self, observer: Observer, update: Update) -> None:
        self.logger.info(f"Processing update from {observer.name}")

        # mark activity so the keep-warm pinger holds the text model resident
        self._last_activity = time.monotonic()

        # add to batch
        observation_id = self.batcher.push(
            observer_name=observer.name,
            content=update.content,
            content_type=update.content_type
        )
        self.logger.info(f"Added observation {observation_id} to queue (size: {self.batcher.size()})")

    @asynccontextmanager
    async def _session(self):
        async with self.Session() as s:
            async with s.begin():
                yield s

    @staticmethod
    async def _attach_obs_if_missing(prop: Proposition, obs: Observation, session):
        await session.execute(
            insert(observation_proposition)
            .prefix_with("OR IGNORE")
            .values(observation_id=obs.id, proposition_id=prop.id)
        )
        prop.updated_at = datetime.now(timezone.utc)

    def add_observer(self, observer: Observer):
        """Add an observer to track user behavior.
        
        Args:
            observer (Observer): The observer to add.
        """
        self.observers.append(observer)

    def remove_observer(self, observer: Observer):
        """Remove an observer from tracking.
        
        Args:
            observer (Observer): The observer to remove.
        """
        if observer in self.observers:
            self.observers.remove(observer)

    def register_update_handler(self, fn: Callable[[Observer, Update], None]):
        """Register a custom update handler function.
        
        Args:
            fn (Callable[[Observer, Update], None]): The handler function to register.
        """
        self.update_handlers.append(fn)

    async def query(
        self,
        user_query: str,
        *,
        limit: int = 3,
        mode: str = "OR",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[tuple[Proposition, float]]:
        """Query the database for propositions matching the user query.
        
        Args:
            user_query (str): The query string to search for.
            limit (int, optional): Maximum number of results to return. Defaults to 3.
            mode (str, optional): Search mode ("OR" or "AND"). Defaults to "OR".
            start_time (datetime, optional): Start time for filtering results. Defaults to None.
            end_time (datetime, optional): End time for filtering results. Defaults to None.
            
        Returns:
            list[tuple[Proposition, float]]: List of tuples containing propositions and their relevance scores.
        """
        async with self._session() as session:
            return await search_propositions_bm25(
                session,
                user_query,
                limit=limit,
                mode=mode,
                start_time=start_time,
                end_time=end_time,
            )

    async def recent(
        self,
        *,
        limit: int = 10,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_observations: bool = False,
    ) -> list[Proposition]:
        """Return the most recent propositions ordered by created_at descending."""
        async with self._session() as session:
            return await get_recent_propositions(
                session,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
                include_observations=include_observations,
            )

    async def recent_observations(
        self,
        *,
        limit: int = 10,
        offset: int = 0,
        ascending: bool = False,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Observation]:
        """Return observations in the window, ordered by created_at.

        Defaults to newest-first. ``ascending`` plus ``offset`` lets callers page
        chronologically through a large time window instead of loading it in one shot.
        """
        async with self._session() as session:
            return await get_recent_observations(
                session,
                limit=limit,
                offset=offset,
                ascending=ascending,
                start_time=start_time,
                end_time=end_time,
            )

    async def count_observations(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> tuple[int, int]:
        """Return ``(row_count, total_content_chars)`` for observations in a window
        without loading any rows — lets a caller size a window before streaming it."""
        async with self._session() as session:
            return await count_observations(
                session, start_time=start_time, end_time=end_time
            )

    # ── proposition review / feedback ─────────────────────────────────────────
    async def next_for_review(
        self, exclude_ids: set[int] | None = None
    ) -> tuple[Proposition, list[Observation]] | None:
        """Return the next unreviewed proposition and its observations, or None.

        ``exclude_ids`` skips propositions the caller deferred this session.
        """
        async with self._session() as session:
            prop = await get_next_unreviewed_proposition(session, exclude_ids=exclude_ids)
            if prop is None:
                return None
            obs = sorted(prop.observations, key=lambda o: o.created_at)
            return prop, obs

    async def add_review(
        self, proposition_id: int, rating: str, note: str | None = None
    ) -> bool:
        """Record a rating (accurate/partial/inaccurate) plus optional context
        note for a proposition. Returns False if the proposition no longer
        exists."""
        async with self._session() as session:
            prop = await session.get(Proposition, proposition_id)
            if prop is None:
                return False
            await add_proposition_feedback(session, prop, rating, note)
            return True

    async def review_progress(self) -> tuple[int, int]:
        """Return (total_propositions, reviewed_count)."""
        async with self._session() as session:
            return await count_review_progress(session)

    async def recent_feedback(self, *, limit: int = 8):
        """Return the most recent true/false judgments (newest first)."""
        async with self._session() as session:
            return await get_recent_feedback(session, limit=limit)

    async def add_suggestion_feedback(
        self,
        *,
        title: str,
        vote: str,
        description: str | None = None,
        focus: str | None = None,
    ) -> bool:
        """Record the user's thumbs up/down on a GUMBO suggestion as an observation.

        Feeding reactions back in as observations closes the mixed-initiative loop
        (paper §4.3): a thumbs-down on a proactive suggestion is itself evidence
        about the user, so GUMBO submits it through the same batching pipeline as
        any other observation. Future propositions — and thus future suggestions —
        then reflect what the user actually found useful. Returns False for an
        unrecognized vote so callers can validate input.
        """
        vote = (vote or "").strip().lower()
        if vote not in ("up", "down"):
            return False
        reaction = "found helpful" if vote == "up" else "did not find helpful"
        title = (title or "").strip() or "(untitled)"
        parts = [f'{self.user_name} {reaction} a proactive GUMBO suggestion titled "{title}".']
        if description and description.strip():
            parts.append(f"The suggestion was: {description.strip()}")
        if focus and focus.strip():
            parts.append(f"(project focus: {focus.strip()})")
        self.batcher.push(
            observer_name="gumbo_feedback",
            content=" ".join(parts),
            content_type="input_text",
        )
        return True
