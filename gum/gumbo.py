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
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from .gum import gum
from .llm import structured_completion, text_completion
from .models import Proposition
from .prompts.gumbo import CHAT_SYSTEM_PROMPT, SUGGESTIONS_PROMPT
from .schemas import SuggestionSchema

# Only propositions the GUM is fairly sure about should seed suggestions — a
# suggestion built on a shaky inference is worse than no suggestion. 7/10 keeps
# the bar high without being so strict that a fresh GUM surfaces nothing.
DEFAULT_MIN_CONFIDENCE = 7
# How many propositions to feed the model. Capped so the prompt stays well inside
# the text model's context window (see DEFAULT_TEXT_NUM_CTX in llm.py).
DEFAULT_MAX_PROPOSITIONS = 20
DEFAULT_NUM_SUGGESTIONS = 5
# Two suggestions whose title+description share at least this fraction of their
# words (Jaccard token overlap) are treated as repeats. The paper (§4.3.2) filters
# repeats "using lexical overlap heuristics"; a slew of related propositions often
# makes the model emit near-duplicate suggestions, and surfacing the same idea
# twice is exactly the kind of noise mixed-initiative interaction tries to avoid.
DEFAULT_DEDUP_THRESHOLD = 0.6
# Even after the mixed-initiative filter, the paper (§4.3.2) found suggestions
# would "pour through the decision boundary" because Horvitz's framework treats
# each interruption as independent, whereas the cost of *another* notification
# depends on how many the user already got this minute. So GUMBO adds a
# token-bucketing rate limit on top, capped at ~1 surfaced suggestion per minute.
DEFAULT_SURFACE_INTERVAL = 60.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings, in [0, 1].

    1.0 means the two share every word; 0.0 means no word in common. Used to spot
    near-duplicate suggestions (paper §4.3.2's "lexical overlap heuristics").
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class TokenBucket:
    """A classic token bucket (paper §4.3.2, [17]) for rate-limiting surfacing.

    Starts full with *capacity* tokens and regains one token every
    *refill_seconds*, never exceeding *capacity*. :meth:`take` withdraws whole
    tokens (one per suggestion we want to surface) and returns how many were
    granted — so with capacity 1 and refill 60s, at most one suggestion surfaces
    per minute, with no burst beyond a single held token.

    The clock is injectable (defaults to a monotonic wall clock) so callers can
    drive it deterministically in tests and so time only ever moves forward.
    """

    def __init__(
        self,
        capacity: float,
        refill_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = float(capacity)
        self.refill_seconds = float(refill_seconds)
        self._clock = clock
        self._tokens = self.capacity
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        if self.refill_seconds > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed / self.refill_seconds)

    @property
    def available(self) -> float:
        """Tokens currently available (after accounting for elapsed refill)."""
        self._refill()
        return self._tokens

    def take(self, max_n: int) -> int:
        """Withdraw up to *max_n* whole tokens; return the number granted (>= 0)."""
        self._refill()
        n = min(int(max_n), int(self._tokens))
        if n > 0:
            self._tokens -= n
        return n


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
        dedup_threshold: float | None = None,
        surface_interval: float | None = None,
        clock: Callable[[], float] = time.monotonic,
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
        self.dedup_threshold = (
            dedup_threshold if dedup_threshold is not None
            else _env_float("GUMBO_DEDUP_THRESHOLD", DEFAULT_DEDUP_THRESHOLD)
        )
        self.surface_interval = (
            surface_interval if surface_interval is not None
            else _env_float("GUMBO_SURFACE_INTERVAL", DEFAULT_SURFACE_INTERVAL)
        )
        # One token, refilled once per interval → at most one surfaced suggestion
        # per interval (paper's ~1/min). A non-positive interval disables the
        # limit entirely (bucket is None → surface() returns every worthy item).
        self._bucket = (
            TokenBucket(1, self.surface_interval, clock=clock)
            if self.surface_interval > 0 else None
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

    # ── de-duplication ───────────────────────────────────────────────────────
    def _dedupe(
        self,
        suggestions: list[Suggestion],
        seen: Iterable[str] | None = None,
    ) -> list[Suggestion]:
        """Drop near-duplicate suggestions via lexical overlap (paper §4.3.2).

        *suggestions* is assumed already ranked best-first, so when two overlap we
        keep the earlier (higher-utility) one. *seen* is an optional set of already
        surfaced suggestion texts (title/description) — used to suppress repeats
        across successive polls, not just within one batch. A threshold of 0 turns
        de-duplication off.
        """
        if self.dedup_threshold <= 0:
            return suggestions

        kept: list[Suggestion] = []
        kept_texts: list[str] = [t for t in (seen or []) if t and t.strip()]
        for s in suggestions:
            text = f"{s.title} {s.description}"
            if any(lexical_overlap(text, prev) >= self.dedup_threshold for prev in kept_texts):
                self.logger.debug("gumbo: dropping near-duplicate suggestion %r", s.title)
                continue
            kept.append(s)
            kept_texts.append(text)
        return kept

    # ── generation ───────────────────────────────────────────────────────────
    async def generate(
        self,
        focus: str | None = None,
        *,
        seen: Iterable[str] | None = None,
    ) -> list[Suggestion]:
        """Return scored suggestions for the user, ranked by expected utility.

        Near-duplicate suggestions are filtered out by lexical overlap (paper
        §4.3.2); pass *seen* (already surfaced title/description strings) to also
        suppress repeats carried over from an earlier poll. Returns an empty list
        when there aren't enough confident propositions to ground a suggestion —
        GUMBO stays quiet rather than guessing.
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
        # Filter repeats (paper §4.3.2) *after* ranking, so each de-duplicated
        # cluster is represented by its highest-utility member.
        return self._dedupe(suggestions, seen=seen)

    # ── surfacing (rate-limited notifications) ───────────────────────────────
    async def surface(
        self,
        focus: str | None = None,
        *,
        seen: Iterable[str] | None = None,
    ) -> list[Suggestion]:
        """The suggestions worth *interrupting* the user with, right now.

        Runs the full pipeline (:meth:`generate` → rank → de-dup), keeps only the
        mixed-initiative-surfaced ones, then applies the paper's token-bucket rate
        limit (§4.3.2) so at most ~1 surfaces per minute even when several clear
        the expected-utility bar in the same poll. Because the bucket lives on the
        (shared) engine, that cap holds *across* successive calls, not just within
        one. With the limit disabled it returns every worthy suggestion.
        """
        ranked = await self.generate(focus, seen=seen)
        worthy = [s for s in ranked if s.should_surface]
        if self._bucket is None or not worthy:
            return worthy
        granted = self._bucket.take(len(worthy))
        if granted < len(worthy):
            self.logger.debug(
                "gumbo: rate-limited %d surfaced suggestion(s) to %d this interval",
                len(worthy), granted,
            )
        return worthy[:granted]

    # ── conversation ─────────────────────────────────────────────────────────
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        suggestion: dict[str, str] | None = None,
        focus: str | None = None,
    ) -> str:
        """Continue a "Start Chat" conversation about a suggestion (paper §4.3.3).

        *messages* is the running user/assistant turn list (each ``{"role", "content"}``).
        The reply is grounded in the user's high-confidence propositions — retrieved
        by *focus* (the active project tab) if given, otherwise the suggestion's own
        title/description — so GUMBO answers from what it actually knows. Returns the
        assistant's plain-text reply.
        """
        seed = focus
        if not (seed and seed.strip()) and suggestion:
            seed = " ".join(
                v for v in (suggestion.get("title"), suggestion.get("description")) if v
            ).strip() or None
        props = await self.select_propositions(seed)

        suggestion_context = ""
        if suggestion and (suggestion.get("title") or suggestion.get("description")):
            parts = [f"\n## The suggestion {self.gum.user_name} is asking about\n"]
            if suggestion.get("title"):
                parts.append(f"Title: {suggestion['title'].strip()}")
            if suggestion.get("description"):
                parts.append(f"Details: {suggestion['description'].strip()}")
            if suggestion.get("rationale"):
                parts.append(f"Why GUMBO raised it: {suggestion['rationale'].strip()}")
            suggestion_context = "\n".join(parts) + "\n"

        system = CHAT_SYSTEM_PROMPT.format(
            user_name=self.gum.user_name,
            propositions=self._format_propositions(props) or "(nothing confidently known yet)",
            suggestion_context=suggestion_context,
        )
        convo = [{"role": "system", "content": system}]
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                convo.append({"role": role, "content": content})

        return await text_completion(
            self.gum.client, self.gum.model, convo, logger=self.logger
        )

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
