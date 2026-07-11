# executor.py
#
# The GUMBO execution bridge (spec #4). The paper's loudest negative finding
# (§4.3.3, §8.4) is that GUMBO produces good ideas but cannot ACT on them —
# "Ideas are cheap. Execution is everything." This module closes that loop: it
# takes a scored `Suggestion` (gum.gumbo) and, when — and only when — the
# suggestion is high-confidence AND the action it implies is low-risk and
# reversible, dispatches it to a sandboxed agent that already receives grounded
# GUM context, capturing the agent's output as a *reviewable artifact* held for
# the user's approval. Nothing irreversible is ever done automatically.
#
# This file is built up across iterations. This iteration lands the safety gate:
# the risk/reversibility assessment (local text model) and the decision rule that
# separates "safe to auto-dispatch" from "keep proposal-only." Dispatch to an
# agent backend and the approval surface are layered on top of it next, behind an
# explicit, default-OFF opt-in on the Gumbo engine.

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import signal
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

from .context import gather_context, render_context
from .gumbo import Suggestion, _env_float, _env_int
from .llm import structured_completion
from .prompts.gumbo import EXECUTION_AGENT_PROMPT, RISK_ASSESSMENT_PROMPT
from .schemas import RiskAssessmentSchema

# The suggestion itself must clear a high-confidence bar before its action is even
# considered for auto-dispatch: GUMBO already gates *surfacing* on the
# mixed-initiative decision, and acting is a strictly higher-stakes commitment
# than surfacing, so we additionally require the model's own P(useful) to be high.
DEFAULT_MIN_PROBABILITY = 8
# The action's assessed risk (1–10) must be at or below this to auto-dispatch.
# Kept deliberately low: the whole point of the bridge is that only near-harmless,
# reversible actions ever run without the user first saying yes.
DEFAULT_MAX_RISK = 3
# How many GUM propositions to ground a dispatched task on. The backend agent
# gets these as context (the same assembly the MCP hands local agents); a handful
# of the most relevant, high-confidence facts is enough to ground the work
# without bloating the agent's prompt.
DEFAULT_CONTEXT_LIMIT = 10
# Hard wall-clock cap on a dispatched agent run. The backend is killed if it
# overruns; a dispatch that can't finish in a couple of minutes is not the kind
# of small, reversible task the bridge is meant to auto-run, so failing closed
# (no result) is the right outcome.
DEFAULT_TIMEOUT = 120.0
# The permission mode the shipped backend runs the `claude` CLI in by default.
# "plan" restricts the agent to reading/researching and returning a proposal —
# the CLI itself refuses file edits, Bash, and outward-facing tools — which
# enforces the executor's "reviewable artifact, never act" contract at the tool
# layer rather than trusting the prompt alone. See ClaudeCLIBackend.
DEFAULT_PERMISSION_MODE = "plan"

# Classifications the gate is willing to run automatically. "irreversible" is
# never in this set by construction — those actions are always proposal-only.
_AUTO_DISPATCH_REVERSIBILITY = frozenset({"read_only", "reversible"})

# The three terminal states a dispatch can land in. Every state that involves an
# agent run holds its output for the user — nothing here commits anything.
STATUS_PROPOSAL_ONLY = "proposal_only"   # gate rejected it; agent never ran
STATUS_PENDING_APPROVAL = "pending_approval"  # agent produced a reviewable draft
STATUS_FAILED = "failed"  # agent ran but errored/timed out; nothing to approve


@dataclass
class RiskAssessment:
    """The execution bridge's safety read on a single suggestion's action.

    ``reversibility`` and ``risk`` come straight from the risk-assessment LLM call
    (see :meth:`Executor.assess_risk`); the derived helpers express the gate's view
    of them. High-level policy — whether *this* suggestion may auto-dispatch —
    lives in :meth:`Executor.is_auto_dispatchable`, which also weighs the
    suggestion's own confidence; this object only describes the action's danger.
    """

    reversibility: str
    risk: int
    rationale: str

    @property
    def is_reversible(self) -> bool:
        """True when the action only reads or can be trivially undone."""
        return self.reversibility in _AUTO_DISPATCH_REVERSIBILITY


@dataclass
class AgentResult:
    """The reviewable artifact a backend produces from a dispatched task.

    The executor never lets a backend commit an irreversible side effect; a
    backend's job is to produce *output for the user to approve*. ``ok`` is False
    when the run failed or timed out, in which case ``error`` explains why and
    ``output`` may be partial or empty.
    """

    ok: bool
    output: str
    error: str | None = None


@runtime_checkable
class AgentBackend(Protocol):
    """A thin, swappable interface to a sandboxed agent that carries out a task.

    Kept minimal on purpose so backends stay interchangeable: the shipped backend
    shells out to the local ``claude`` CLI in a restricted working directory, but a
    test double or an alternative agent runtime can satisfy the same contract. The
    backend receives the task text and the GUM-grounded ``context`` string the
    executor assembled (the same grounding the MCP server hands local agents) and
    must confine its work to ``cwd`` and honour ``timeout`` seconds.
    """

    async def run(
        self, task: str, context: str, *, cwd: str, timeout: float
    ) -> AgentResult:
        ...


@dataclass
class ExecutionOutcome:
    """The reviewable artifact :meth:`Executor.dispatch` produces for a suggestion.

    This is the object a UI (the web suggestion card or ``gum execute --review``)
    renders so the user can accept or reject it. It never represents a committed
    action: ``status`` is one of :data:`STATUS_PROPOSAL_ONLY` (the gate declined to
    run it), :data:`STATUS_PENDING_APPROVAL` (an agent produced a draft awaiting
    approval), or :data:`STATUS_FAILED` (the agent ran but errored/timed out).
    ``reason`` explains a proposal-only/failed outcome; ``context`` is the
    (pseudonymized) GUM grounding the agent received.
    """

    suggestion: Suggestion
    status: str
    assessment: RiskAssessment | None = None
    context: str | None = None
    result: AgentResult | None = None
    reason: str | None = None

    @property
    def is_pending_approval(self) -> bool:
        """True when an agent produced a draft the user still needs to approve."""
        return self.status == STATUS_PENDING_APPROVAL

    @property
    def dispatched(self) -> bool:
        """True when the gate cleared the suggestion and an agent actually ran."""
        return self.status in (STATUS_PENDING_APPROVAL, STATUS_FAILED)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the REST API / CLI review surface (JSON-friendly)."""
        return {
            "suggestion": self.suggestion.to_dict(),
            "status": self.status,
            "assessment": asdict(self.assessment) if self.assessment else None,
            "context": self.context,
            "result": asdict(self.result) if self.result else None,
            "reason": self.reason,
        }


class ClaudeCLIBackend:
    """An :class:`AgentBackend` that shells out to the local ``claude`` CLI.

    Runs the agent non-interactively (``claude -p``) confined to a restricted
    working directory, with the GUM-grounded task fed on **stdin** (so no prompt
    text ever touches a shell command line) and a hard wall-clock timeout that
    kills the whole process tree on overrun (the CLI is launched as its own
    session leader, so its bash-tool/MCP-server children die with it rather than
    orphaning past the timeout). The CLI's stdout is captured verbatim as the
    reviewable draft; the backend commits nothing itself.

    Sandboxing here is defence-in-depth on top of the executor's risk gate:

    - the agent only ever sees a *restricted* ``cwd`` (the executor hands it a
      scratch workspace, not the user's real project);
    - the run is time-boxed and its whole process tree torn down on overrun;
    - the CLI runs in a read/research-only **permission mode** (``plan`` by
      default), so the tool layer itself refuses file edits, Bash, and
      outward-facing actions — the executor's "produce a reviewable draft, never
      act" contract is enforced by the CLI, not merely requested in the prompt.

    ``command`` and ``extra_args`` are injectable so a deployment can pass its own
    binary path or tighten the CLI's tool-permission flags without editing this
    class; the ``GUM_EXECUTOR_CLAUDE_ARGS`` env var appends further args
    (shell-split). ``permission_mode`` is kept a first-class, separate property
    (env override ``GUM_EXECUTOR_PERMISSION_MODE``; set it empty to disable) so
    customizing ``extra_args`` can never silently drop the safety posture.
    """

    def __init__(
        self,
        *,
        command: str = "claude",
        extra_args: list[str] | None = None,
        permission_mode: str | None = DEFAULT_PERMISSION_MODE,
        logger: logging.Logger | None = None,
    ) -> None:
        self.command = command
        # `-p` runs the CLI in non-interactive "print" mode: it reads the prompt,
        # emits the result to stdout, and exits — no REPL, no user prompts.
        self.extra_args = list(extra_args) if extra_args is not None else ["-p"]
        env_args = os.getenv("GUM_EXECUTOR_CLAUDE_ARGS", "").strip()
        if env_args:
            self.extra_args = self.extra_args + shlex.split(env_args)
        # Read/research-only posture, enforced at the CLI's tool layer rather than
        # trusted to the prompt. Kept out of `extra_args` so a deployment that
        # customizes args can't accidentally strip it; an explicit empty override
        # (``GUM_EXECUTOR_PERMISSION_MODE=``) opts a fully-trusted backend out.
        env_mode = os.getenv("GUM_EXECUTOR_PERMISSION_MODE")
        if env_mode is not None:
            permission_mode = env_mode.strip() or None
        self.permission_mode = permission_mode
        self.logger = logger or logging.getLogger("gum.executor.backend")

    def _build_argv(self) -> list[str]:
        """Assemble the CLI argv, always folding in the safety permission mode.

        Split out from :meth:`run` so the invocation — in particular that the
        read/research-only ``--permission-mode`` is present by default — is unit
        testable without launching a subprocess.
        """
        argv = [self.command, *self.extra_args]
        if self.permission_mode:
            argv += ["--permission-mode", self.permission_mode]
        return argv

    def _build_prompt(self, task: str, context: str) -> str:
        # Fold the pseudonymized GUM grounding and the task into the single agent
        # instruction. `task` already carries the safety framing (see
        # Executor._build_task); nothing here re-derives context, keeping the one
        # grounding path the spec requires.
        return f"{context}\n\n{task}\n" if context else f"{task}\n"

    def _kill_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """SIGKILL the agent *and every process it spawned*.

        The ``claude`` CLI spawns children of its own (bash-tool commands, MCP
        servers); a plain ``proc.kill()`` reaches only the direct child and would
        orphan those grandchildren past the timeout, defeating the sandbox's
        wall-clock guarantee. Because the process was launched into its own session
        (``start_new_session=True``) it leads a process group whose id equals its
        pid, so one ``killpg`` tears the whole tree down. Falls back to a direct
        kill where process groups are unavailable (e.g. Windows).
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # already gone
        except (AttributeError, OSError):  # pragma: no cover - platform-dependent
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def run(
        self, task: str, context: str, *, cwd: str, timeout: float
    ) -> AgentResult:
        prompt = self._build_prompt(task, context)
        argv = self._build_argv()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Run the agent as its own session/process-group leader so a timeout
                # can tear down the whole tree (see _kill_process_tree), not just the
                # CLI's direct child.
                start_new_session=True,
            )
        except FileNotFoundError:
            return AgentResult(
                ok=False,
                output="",
                error=f"agent CLI {self.command!r} not found on PATH",
            )
        except OSError as exc:  # pragma: no cover - platform-dependent
            return AgentResult(ok=False, output="", error=f"failed to launch agent: {exc}")

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=timeout
            )
        except asyncio.TimeoutError:
            self._kill_process_tree(proc)
            await proc.wait()
            return AgentResult(
                ok=False, output="", error=f"agent timed out after {timeout:g}s"
            )

        out = stdout.decode(errors="replace").strip()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip() or f"exit code {proc.returncode}"
            return AgentResult(ok=False, output=out, error=err)
        return AgentResult(ok=True, output=out)


class Executor:
    """Decides whether a GUMBO suggestion may act, and (later) dispatches it.

    Cheap to construct; does no I/O until a method is called. This iteration
    implements the safety gate only: :meth:`assess_risk` asks the local text model
    to classify the action's reversibility and risk, and :meth:`is_auto_dispatchable`
    applies the policy that combines that assessment with the suggestion's own
    confidence. A suggestion that fails the gate stays proposal-only.
    """

    def __init__(
        self,
        gum_instance,
        *,
        backend: AgentBackend | None = None,
        min_probability: int | None = None,
        max_risk: int | None = None,
        context_limit: int | None = None,
        timeout: float | None = None,
        workspace_dir: str | None = None,
        sanitize: bool = True,
        sanitizer: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.gum = gum_instance
        self.backend = backend
        self.min_probability = (
            min_probability if min_probability is not None
            else _env_int("GUM_EXECUTOR_MIN_PROBABILITY", DEFAULT_MIN_PROBABILITY)
        )
        self.max_risk = (
            max_risk if max_risk is not None
            else _env_int("GUM_EXECUTOR_MAX_RISK", DEFAULT_MAX_RISK)
        )
        self.context_limit = (
            context_limit if context_limit is not None
            else _env_int("GUM_EXECUTOR_CONTEXT_LIMIT", DEFAULT_CONTEXT_LIMIT)
        )
        self.timeout = (
            timeout if timeout is not None
            else _env_float("GUM_EXECUTOR_TIMEOUT", DEFAULT_TIMEOUT)
        )
        # The sandboxed cwd handed to the backend. Defaults to a scratch directory
        # under the GUM data dir — deliberately NOT the user's real project — so a
        # dispatched agent is confined to a throwaway workspace. Created lazily on
        # first dispatch so construction stays I/O-free.
        self._workspace_dir = (
            workspace_dir if workspace_dir is not None
            else os.getenv("GUM_EXECUTOR_WORKSPACE")
        )
        # The shipped backend shells the task out to the local `claude` CLI, i.e.
        # a frontier model off the device, so the GUM context that grounds it must
        # be pseudonymized on the way out — same fail-closed default as the MCP
        # server. A fully-local, trusted backend can opt out with sanitize=False.
        # An explicit *sanitizer* (a test double, or a pre-loaded instance) skips
        # lazy loading; otherwise it is loaded the first time context is assembled
        # so constructing an Executor stays cheap and import-light.
        self.sanitize = sanitize
        self._sanitizer = sanitizer
        self.logger = logger or logging.getLogger("gum.executor")

    async def assess_risk(self, suggestion: Suggestion) -> RiskAssessment:
        """Classify the reversibility and risk of *suggestion*'s implied action.

        Uses the same local text model the rest of the pipeline uses (nothing
        leaves the machine). The prompt biases toward the less-safe classification
        under uncertainty, so a genuinely ambiguous action lands proposal-only.
        """
        prompt = RISK_ASSESSMENT_PROMPT.format(
            user_name=self.gum.user_name,
            title=suggestion.title,
            description=suggestion.description,
        )
        result = await structured_completion(
            self.gum.client,
            self.gum.model,
            [{"role": "user", "content": prompt}],
            RiskAssessmentSchema,
            logger=self.logger,
        )
        return RiskAssessment(
            reversibility=result.reversibility,
            risk=result.risk,
            rationale=result.rationale,
        )

    def _get_sanitizer(self) -> Any:
        """Return the egress sanitizer for grounding context, or None if disabled.

        Loaded lazily and fail-closed: the first call constructs and loads the
        PII model, so if its dependencies are missing this raises rather than
        silently handing raw identities to an off-device agent. Constructing an
        Executor stays cheap because this is deferred until context is assembled.
        """
        if not self.sanitize:
            return None
        if self._sanitizer is None:
            from .sanitize import get_sanitizer

            self._sanitizer = get_sanitizer()
            self._sanitizer.load()
        return self._sanitizer

    async def assemble_context(self, suggestion: Suggestion) -> str:
        """Build the GUM-grounded prompt block for *suggestion*'s dispatched task.

        Reuses the exact same assembly the MCP server's ``gather_context`` tool
        uses (:func:`gum.context.gather_context`) — retrieval on the substantive
        terms plus fail-closed egress pseudonymization — so the execution bridge
        does not fork a second grounding path (spec #4). Returns a text block a
        backend can embed in the agent's instructions; the suggestion's own
        title/description seed the retrieval topic.
        """
        topic = f"{suggestion.title}. {suggestion.description}".strip()
        result = await gather_context(
            self.gum,
            topic,
            sanitizer=self._get_sanitizer(),
            limit=self.context_limit,
        )
        return render_context(result)

    def _ensure_workspace_root(self) -> str:
        """Return the base scratch directory dispatched agents are sandboxed under.

        The default is a directory under the GUM data dir — never the user's real
        working tree. Individual dispatches get their *own* ephemeral subdirectory
        beneath this root (see :meth:`dispatch`); this only resolves and creates
        the shared parent. Done lazily so constructing an Executor does no I/O.
        """
        path = self._workspace_dir or os.path.join(
            self.gum._data_directory, "executor_workspace"
        )
        os.makedirs(path, exist_ok=True)
        self._workspace_dir = path
        return path

    async def _build_task(self, suggestion: Suggestion, context: str) -> str:
        """Compose the safety-framed, GUM-grounded instruction for the backend.

        Wraps the suggestion in :data:`EXECUTION_AGENT_PROMPT`, which embeds the
        (already pseudonymized) ``context`` and instructs the agent to produce a
        reviewable draft rather than take any irreversible action.

        The shipped backend ships this whole instruction to an **off-device**
        model, so every identity that reaches it must be pseudonymized on egress —
        not just the ``context`` block (which :meth:`assemble_context` already
        pseudonymized), but the two identities this prompt reintroduces itself:

        - the **suggestion text** (title/description), which is GUM-generated and
          may embed real names/projects drawn from the user's propositions; and
        - the user's own **name**, which :data:`EXECUTION_AGENT_PROMPT` stamps in
          verbatim — leaving it raw both leaks it and defeats the context's
          pseudonymization (the agent could tie "[PERSON_1]" back to the user).

        Both go through the SAME sanitizer as the grounding context, whose entity
        map is stable, so a name here maps to the exact pseudo-ID it already
        carries in the context block. With sanitization disabled (a fully-local,
        trusted backend) the raw text is used unchanged.
        """
        parts = [suggestion.title.strip()]
        if suggestion.description and suggestion.description.strip():
            parts.append(suggestion.description.strip())
        task_body = "\n\n".join(parts)
        user_name = self.gum.user_name
        sanitizer = self._get_sanitizer()
        if sanitizer is not None:
            # Blocking model inference; offload like gather_context does. The
            # sanitizer's entity map is loaded/warmed by assemble_context above.
            task_body = await asyncio.to_thread(sanitizer.sanitize, task_body)
            user_name = await asyncio.to_thread(sanitizer.sanitize, user_name)
        return EXECUTION_AGENT_PROMPT.format(
            user_name=user_name,
            context=context,
            task=task_body,
        )

    async def dispatch(self, suggestion: Suggestion) -> ExecutionOutcome:
        """Run the full bridge for *suggestion*: gate → ground → dispatch → capture.

        Assesses the action's risk and applies the auto-dispatch gate; a
        suggestion that fails stays :data:`STATUS_PROPOSAL_ONLY` and no agent runs.
        A suggestion that clears it is grounded on the shared GUM context assembly
        and handed to the configured :class:`AgentBackend`, whose output is
        captured into a :class:`ExecutionOutcome` held for the user's approval
        (:data:`STATUS_PENDING_APPROVAL`, or :data:`STATUS_FAILED` if the agent
        errored or timed out). This method never commits an irreversible effect —
        it only ever produces a reviewable artifact.
        """
        try:
            assessment = await self.assess_risk(suggestion)
        except Exception as exc:  # fail closed on ANY assessment error
            # If the safety classifier itself can't complete (a flaky/failed local
            # model call, a malformed structured response), we must NOT dispatch:
            # an un-assessable action is exactly the case the gate exists to hold.
            # Fail closed to proposal-only for THIS suggestion rather than raising,
            # so one bad assessment can't abort a whole execute() batch of others.
            self.logger.warning(
                "executor: risk assessment failed for suggestion %r; holding "
                "proposal-only (%s)", suggestion.title, exc,
            )
            return ExecutionOutcome(
                suggestion=suggestion,
                status=STATUS_PROPOSAL_ONLY,
                assessment=None,
                reason=f"held for review: risk assessment failed ({exc})",
            )
        if not self.is_auto_dispatchable(suggestion, assessment):
            return ExecutionOutcome(
                suggestion=suggestion,
                status=STATUS_PROPOSAL_ONLY,
                assessment=assessment,
                reason="held for review: did not clear the auto-dispatch gate",
            )
        if self.backend is None:
            # Gate cleared but nothing to run it — stay proposal-only rather than
            # pretending a dispatch happened.
            return ExecutionOutcome(
                suggestion=suggestion,
                status=STATUS_PROPOSAL_ONLY,
                assessment=assessment,
                reason="no agent backend configured",
            )

        try:
            context = await self.assemble_context(suggestion)
            task = await self._build_task(suggestion, context)
        except Exception as exc:  # fail closed if grounding/sanitization can't build
            # The gate cleared, but before we can dispatch we must assemble the
            # GUM grounding and pseudonymize the whole prompt for the off-device
            # backend. If that setup fails — a transient retrieval/DB error, or the
            # fail-closed egress sanitizer refusing to load its PII model — we must
            # NOT hand the backend an un-grounded or un-sanitized prompt. Hold THIS
            # suggestion proposal-only (assessment preserved; agent never touched)
            # rather than raising, so one setup failure can't abort a whole
            # execute() batch of other suggestions.
            self.logger.warning(
                "executor: could not build grounded prompt for suggestion %r; "
                "holding proposal-only (%s)", suggestion.title, exc,
            )
            return ExecutionOutcome(
                suggestion=suggestion,
                status=STATUS_PROPOSAL_ONLY,
                assessment=assessment,
                reason=f"held for review: could not build grounded prompt ({exc})",
            )
        # Each dispatch gets its OWN ephemeral subdirectory under the workspace root,
        # torn down when the run finishes. A single shared cwd would leak one agent
        # run's scratch files and state into the next dispatch's sandbox and let
        # them accumulate unbounded; a fresh, isolated, self-cleaning directory per
        # run keeps the "restricted cwd" sandbox guarantee honest across dispatches.
        cwd = tempfile.mkdtemp(prefix="dispatch-", dir=self._ensure_workspace_root())
        self.logger.info(
            "executor: dispatching suggestion %r to agent (cwd=%s, timeout=%gs)",
            suggestion.title, cwd, self.timeout,
        )
        try:
            # The backend already received the grounding folded into `task`; passing
            # an empty context here avoids duplicating it into the prompt twice.
            result = await self.backend.run(task, "", cwd=cwd, timeout=self.timeout)
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        status = STATUS_PENDING_APPROVAL if result.ok else STATUS_FAILED
        return ExecutionOutcome(
            suggestion=suggestion,
            status=status,
            assessment=assessment,
            context=context,
            result=result,
            reason=None if result.ok else (result.error or "agent run failed"),
        )

    def is_auto_dispatchable(
        self, suggestion: Suggestion, assessment: RiskAssessment
    ) -> bool:
        """Whether *suggestion* may run automatically given its *assessment*.

        All four conditions must hold: the mixed-initiative decision already found
        the suggestion worth surfacing, its P(useful) clears the higher execution
        bar, and the action is both reversible and low-risk. Any miss keeps the
        suggestion proposal-only — the safe default the whole bridge is built on.
        """
        reasons: list[str] = []
        if not suggestion.should_surface:
            reasons.append("suggestion not worth surfacing")
        if suggestion.probability_useful < self.min_probability:
            reasons.append(
                f"P(useful) {suggestion.probability_useful} < {self.min_probability}"
            )
        if not assessment.is_reversible:
            reasons.append(f"action is {assessment.reversibility}")
        if assessment.risk > self.max_risk:
            reasons.append(f"risk {assessment.risk} > {self.max_risk}")

        if reasons:
            self.logger.debug(
                "executor: holding suggestion %r proposal-only (%s)",
                suggestion.title,
                "; ".join(reasons),
            )
            return False
        return True
