# gumbo.py
#
# GUMBO — a proactive assistant built on the GUM (paper §4.3). This module is the
# engine: given a live `gum` instance, it retrieves the user's most relevant,
# high-confidence propositions and asks the *local* text model (the same
# Ollama-backed client the GUM already uses — nothing leaves the machine) for
# concrete suggestions. Each suggestion is scored with the mixed-initiative
# expected-utility calculation (Horvitz [36], paper §4.3.2) so callers — the tray
# and, later, the desktop app — can decide which are worth surfacing.
#
# The engine is deliberately UI-agnostic: it returns plain `Suggestion` objects
# (and `to_dict()` for serving over the REST API). Discovery of *when* to poll,
# rate-limiting, and rendering all live in the front-ends that build on this.

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

from .gum import gum
from .llm import structured_completion
from .models import Proposition
from .prompts.gumbo import SUGGESTIONS_PROMPT
from .schemas import SuggestionSchema

# Only propositions the GUM is fairly sure about should seed suggestions — a
# suggestion built on a shaky inference is worse than no suggestion. 7/10 keeps
# the bar high without being so strict that a fresh GUM surfaces nothing.
DEFAULT_MIN_CONFIDENCE = 7
# How many propositions to feed the model. Capped so the prompt stays well inside
# the text model's context window (see DEFAULT_TEXT_NUM_CTX in llm.py).
DEFAULT_MAX_PROPOSITIONS = 20
DEFAULT_NUM_SUGGESTIONS = 5


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Suggestion:
    """A scored, proactive suggestion ready to show (or withhold from) the user."""

    title: str
    description: str
    rationale: str
    probability_useful: int
    benefit: int
    cost_if_wrong: int
    cost_if_missed: int
    # Derived (see expected_utility): net gain of interrupting over staying quiet,
    # and whether that gain is positive (i.e. worth surfacing).
    expected_utility: float
    should_surface: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_utility(
    probability_useful: int,
    benefit: int,
    cost_if_wrong: int,
    cost_if_missed: int,
) -> tuple[float, bool]:
    """Mixed-initiative decision from the paper (§4.3.2, Eqs. 1–4).

    With p = P(useful), benefit B, false-positive cost C_FP and false-negative
    cost C_FN:

        E[U_interrupt]    = p·B − (1 − p)·C_FP
        E[U_no-interrupt] = −p·C_FN

    We return (E[U_interrupt] − E[U_no-interrupt], E[U_interrupt] > E[U_no-interrupt]).
    The scalar is a natural ranking key; the boolean is the surface/withhold
    decision. Scores arrive on a 1–10 scale; probability is normalised to [0, 1].
    """
    p = max(0.0, min(1.0, probability_useful / 10.0))
    e_interrupt = p * benefit - (1.0 - p) * cost_if_wrong
    e_quiet = -p * cost_if_missed
    return e_interrupt - e_quiet, e_interrupt > e_quiet


class Gumbo:
    """Suggestion engine over a live GUM. Cheap to construct; does no I/O until
    :meth:`generate` is called."""

    def __init__(
        self,
        gum_instance: gum,
        *,
        min_confidence: int | None = None,
        max_propositions: int | None = None,
        num_suggestions: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.gum = gum_instance
        self.min_confidence = (
            min_confidence if min_confidence is not None
            else _env_int("GUMBO_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
        )
        self.max_propositions = (
            max_propositions if max_propositions is not None
            else _env_int("GUMBO_MAX_PROPOSITIONS", DEFAULT_MAX_PROPOSITIONS)
        )
        self.num_suggestions = (
            num_suggestions if num_suggestions is not None
            else _env_int("GUMBO_NUM_SUGGESTIONS", DEFAULT_NUM_SUGGESTIONS)
        )
        self.logger = logger or logging.getLogger("gum.gumbo")

    # ── proposition selection ────────────────────────────────────────────────
    async def select_propositions(self, focus: str | None = None) -> list[Proposition]:
        """Pick the high-confidence propositions that seed suggestions.

        With a *focus* (e.g. a project tab's topic), retrieve by semantic/keyword
        relevance so suggestions stay on-topic (paper §4.3.1 pulls related
        propositions with the GUM query function). Without one, fall back to the
        most recent propositions. Either way, keep only those at or above the
        confidence bar, newest/most-relevant first, capped at ``max_propositions``.
        """
        if focus and focus.strip():
            # Over-fetch, then filter by confidence, since BM25 order is by
            # relevance not confidence.
            results = await self.gum.query(focus.strip(), limit=self.max_propositions * 3)
            candidates = [p for p, _ in results]
        else:
            candidates = await self.gum.recent(limit=self.max_propositions * 3)

        high_conf = [p for p in candidates if (p.confidence or 0) >= self.min_confidence]
        return high_conf[: self.max_propositions]

    def _format_propositions(self, props: list[Proposition]) -> str:
        lines: list[str] = []
        for i, p in enumerate(props, 1):
            conf = p.confidence if p.confidence is not None else "?"
            lines.append(f"{i}. (confidence {conf}/10) {p.text.strip()}")
            if p.reasoning:
                lines.append(f"   reasoning: {p.reasoning.strip()}")
        return "\n".join(lines)

    # ── generation ───────────────────────────────────────────────────────────
    async def generate(self, focus: str | None = None) -> list[Suggestion]:
        """Return scored suggestions for the user, ranked by expected utility.

        Returns an empty list when there aren't enough confident propositions to
        ground a suggestion — GUMBO stays quiet rather than guessing.
        """
        props = await self.select_propositions(focus)
        if not props:
            self.logger.debug(
                "gumbo: no propositions at/above confidence %d%s; no suggestions",
                self.min_confidence,
                f" for focus '{focus}'" if focus else "",
            )
            return []

        prompt = SUGGESTIONS_PROMPT.format(
            user_name=self.gum.user_name,
            propositions=self._format_propositions(props),
            num_suggestions=self.num_suggestions,
        )

        result = await structured_completion(
            self.gum.client,
            self.gum.model,
            [{"role": "user", "content": prompt}],
            SuggestionSchema,
            logger=self.logger,
        )

        suggestions = [self._score(item) for item in result.suggestions]
        # Rank by expected utility so the most worth-surfacing float to the top.
        suggestions.sort(key=lambda s: s.expected_utility, reverse=True)
        return suggestions

    def _score(self, item) -> Suggestion:
        eu, surface = expected_utility(
            item.probability_useful,
            item.benefit,
            item.cost_if_wrong,
            item.cost_if_missed,
        )
        return Suggestion(
            title=item.title,
            description=item.description,
            rationale=item.rationale,
            probability_useful=item.probability_useful,
            benefit=item.benefit,
            cost_if_wrong=item.cost_if_wrong,
            cost_if_missed=item.cost_if_missed,
            expected_utility=eu,
            should_surface=surface,
        )
