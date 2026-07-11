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

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .gumbo import Suggestion, _env_int
from .llm import structured_completion
from .prompts.gumbo import RISK_ASSESSMENT_PROMPT
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

# Classifications the gate is willing to run automatically. "irreversible" is
# never in this set by construction — those actions are always proposal-only.
_AUTO_DISPATCH_REVERSIBILITY = frozenset({"read_only", "reversible"})


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
