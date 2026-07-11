# agenda.py
#
# Commitment & Deadline Radar (spec #1, `gum agenda`). The GUM already infers
# time-critical signals — "has a major impending deadline", grants/papers/reviews
# in flight — but that intelligence stays latent inside free-text propositions.
# This module turns it into a standing artifact: a ranked, dated list of the
# user's open commitments, the "surface the one time-critical item, suppress the
# rest" idea from the paper's §4.2 OS vision.
#
# The engine is deliberately UI-agnostic (mirroring gum.gumbo): given a live
# `gum` instance it pulls candidate propositions through the existing
# query/recent APIs, asks the *local* text model to pick out the ones that imply
# an open commitment and extract a structured `{title, due_date, source,
# status_guess}`, then ranks them by urgency. It returns plain `Commitment`
# dataclasses (with `to_dict()` for the JSON API / MCP), so the CLI, REST, and
# MCP surfaces can each render or sanitize them however they need.

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any

from .gum import gum
from .llm import structured_completion
from .models import Proposition
from .prompts.gum import AGENDA_PROMPT
from .schemas import CommitmentItem, CommitmentSchema

# Commitments can matter even when the GUM is only moderately sure, so the
# candidate bar sits lower than GUMBO's suggestion bar (7): the urgency ranking
# already down-weights low-confidence items rather than hiding them outright.
DEFAULT_MIN_CONFIDENCE = 3
# How many candidate propositions to feed the extractor. Wider than GUMBO's net
# (20) because a commitment can hide in any recent proposition, but still capped
# well inside the text model's context window (DEFAULT_TEXT_NUM_CTX in llm.py).
DEFAULT_MAX_PROPOSITIONS = 40
# Default number of commitments the radar returns (the CLI's --limit default).
DEFAULT_LIMIT = 10

# BM25 signal terms used to fish commitment-bearing propositions out of the
# index before falling back to plain recency. OR-mode search, so any hit
# surfaces the proposition; the model does the real judging afterwards.
_COMMITMENT_QUERY = (
    "deadline due submit send respond reply review meeting appointment call "
    "owe promised commit task assignment paper grant proposal application "
    "invoice bill payment renew register rsvp schedule prepare finish"
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _created_dt(prop: Proposition) -> datetime:
    """The proposition's creation time as a tz-aware UTC datetime.

    Rows are written with SQLite's ``func.now()`` (UTC) and read back naive, so
    attach UTC when tzinfo is missing (mirroring cmd_observations). Falls back to
    "now" only for the theoretical case of a missing timestamp (created_at is
    NOT NULL), so downstream age math never sees ``None``.
    """
    dt = prop.created_at
    if dt is None:
        return datetime.now(timezone.utc)
    if isinstance(dt, str):
        # Defensive: some drivers hand back the timestamp as an ISO string.
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_due(value: str | None) -> date | None:
    """Parse a model-supplied due date into a ``date``, or None if unusable.

    The extractor is asked for ISO ``YYYY-MM-DD``, but local models drift, so a
    couple of common alternatives are accepted. Anything unparseable degrades to
    "undated" rather than raising — an undated commitment still belongs on the
    radar, just ranked by confidence and recency instead of proximity.
    """
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def urgency_score(
    days_until_due: int | None,
    confidence: int | None,
    decay: int | None,
    age_days: float,
) -> float:
    """Rank key for a commitment: higher = more worth surfacing now.

    Two regimes, kept on comparable scales so one ranked list mixes them sensibly:

    * **Dated** commitments score in ``[1, 2]`` — always above undated ones,
      since a concrete deadline is more actionable. Proximity dominates: due
      today or overdue scores 1.0, receding with a ~1-week scale as the deadline
      moves into the future, then modulated by confidence so a shaky "deadline
      Friday" doesn't outrank a certain one.
    * **Undated** commitments score in ``[0, 1]`` by ``confidence × recency``,
      with the proposition's own ``decay`` (the GUM's 1 short-lived – 10
      long-lasting "how long will this matter" score) setting how fast relevance
      fades: a low-decay note goes cold in a day, a high-decay one stays warm for
      a week-plus.
    """
    conf = _clamp01((confidence or 0) / 10.0)
    if days_until_due is not None:
        proximity = 1.0 if days_until_due <= 0 else 1.0 / (1.0 + days_until_due / 7.0)
        return 1.0 + proximity * (0.5 + 0.5 * conf)
    half_life = max(1.0, float(decay or 1))
    recency = 1.0 / (1.0 + max(0.0, age_days) / half_life)
    return conf * recency


@dataclass
class Commitment:
    """A single open commitment/deadline, ranked for the radar.

    Text fields (``title``, ``source``, ``proposition_text``) are model-written
    from raw propositions and therefore carry PII — surfaces that leave the
    device (REST/MCP) must pass them through the same sanitizer the other
    proposition surfaces use. The numeric/date fields carry no PII.
    """

    title: str
    due_date: str | None            # ISO 'YYYY-MM-DD', or None if undated
    source: str
    status_guess: str
    confidence: int | None
    decay: int | None
    days_until_due: int | None      # negative = overdue; None if undated
    urgency: float
    proposition_id: int | None
    proposition_text: str
    created_at: str | None          # ISO timestamp of the source proposition

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CommitmentRadar:
    """Extract and rank the user's open commitments from a live GUM.

    Cheap to construct; does no I/O until :meth:`build` is called (mirroring
    gum.gumbo.Gumbo).
    """

    def __init__(
        self,
        gum_instance: gum,
        *,
        min_confidence: int | None = None,
        max_propositions: int | None = None,
        limit: int | None = None,
        window_days: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.gum = gum_instance
        self.min_confidence = (
            min_confidence if min_confidence is not None
            else _env_int("GUM_AGENDA_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
        )
        self.max_propositions = (
            max_propositions if max_propositions is not None
            else _env_int("GUM_AGENDA_MAX_PROPOSITIONS", DEFAULT_MAX_PROPOSITIONS)
        )
        self.limit = limit if limit is not None else DEFAULT_LIMIT
        # None = no horizon; else only keep commitments due within this many days
        # (overdue and undated commitments are always kept — see :meth:`build`).
        self.window_days = window_days
        self.logger = logger or logging.getLogger("gum.agenda")

    # ── candidate selection ───────────────────────────────────────────────────
    async def select_candidates(self) -> list[Proposition]:
        """Pick propositions likely to contain a commitment.

        First a BM25 search on commitment/deadline signal terms (so a stale-but-
        relevant "grant deadline" proposition surfaces even if the user hasn't
        touched it lately), then a recency backfill (so a sparse index or a fresh
        commitment still gets seen). De-duplicated by id, confidence-gated, and
        capped at ``max_propositions``.
        """
        seen: set[int] = set()
        candidates: list[Proposition] = []

        results = await self.gum.query(
            _COMMITMENT_QUERY, limit=self.max_propositions * 2
        )
        for prop, _score in results:
            self._maybe_add(prop, seen, candidates)

        recent = await self.gum.recent(limit=self.max_propositions * 2)
        for prop in recent:
            self._maybe_add(prop, seen, candidates)

        return candidates[: self.max_propositions]

    def _maybe_add(
        self, prop: Proposition, seen: set[int], out: list[Proposition]
    ) -> None:
        if prop.id in seen:
            return
        if (prop.confidence or 0) < self.min_confidence:
            return
        seen.add(prop.id)
        out.append(prop)

    def _format_propositions(self, props: list[Proposition]) -> str:
        lines: list[str] = []
        for i, p in enumerate(props, 1):
            conf = p.confidence if p.confidence is not None else "?"
            recorded = _created_dt(p).date().isoformat()
            lines.append(
                f"{i}. (confidence {conf}/10, recorded {recorded}) {p.text.strip()}"
            )
        return "\n".join(lines)

    # ── extraction + ranking ──────────────────────────────────────────────────
    async def build(self, *, now: datetime | None = None) -> list[Commitment]:
        """Return the ranked list of open commitments (most urgent first).

        Returns an empty list when there are no candidate propositions or the
        model finds nothing that qualifies — the radar stays silent rather than
        inventing deadlines. *now* is injectable so ranking is deterministic in
        tests; it defaults to the current UTC time.
        """
        now = now or datetime.now(timezone.utc)
        props = await self.select_candidates()
        if not props:
            self.logger.debug("agenda: no candidate propositions; empty radar")
            return []

        by_index = {i: p for i, p in enumerate(props, 1)}
        prompt = (
            AGENDA_PROMPT
            .replace("{user_name}", self.gum.user_name)
            .replace("{today}", now.date().isoformat())
            .replace("{propositions}", self._format_propositions(props))
        )

        result = await structured_completion(
            self.gum.client,
            self.gum.model,
            [{"role": "user", "content": prompt}],
            CommitmentSchema,
            logger=self.logger,
        )

        commitments: list[Commitment] = []
        for item in result.commitments:
            prop = by_index.get(item.source_index)
            if prop is None:
                # The model referenced a proposition number that wasn't offered —
                # drop it rather than fabricate provenance.
                self.logger.debug(
                    "agenda: dropping commitment with out-of-range source_index %s",
                    item.source_index,
                )
                continue
            commitment = self._build_commitment(item, prop, now)
            if self._outside_window(commitment):
                continue
            commitments.append(commitment)

        commitments.sort(key=lambda c: c.urgency, reverse=True)
        return commitments[: self.limit] if self.limit is not None else commitments

    def _build_commitment(
        self, item: CommitmentItem, prop: Proposition, now: datetime
    ) -> Commitment:
        due = _parse_due(item.due_date)
        days_until = (due - now.date()).days if due is not None else None
        created = _created_dt(prop)
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        score = urgency_score(days_until, prop.confidence, prop.decay, age_days)
        return Commitment(
            title=(item.title or "").strip(),
            due_date=due.isoformat() if due is not None else None,
            source=((item.source or "").strip() or "unknown"),
            status_guess=((item.status_guess or "").strip() or "unknown"),
            confidence=prop.confidence,
            decay=prop.decay,
            days_until_due=days_until,
            urgency=round(score, 4),
            proposition_id=prop.id,
            proposition_text=prop.text.strip(),
            created_at=created.isoformat(),
        )

    def _outside_window(self, commitment: Commitment) -> bool:
        """Whether a horizon (``window_days``) excludes this commitment.

        A horizon filters *future-dated* commitments beyond it. Overdue
        commitments (negative days) and undated ones are always kept: an overdue
        deadline is the most urgent thing on the radar, and an undated commitment
        has no date to compare against the horizon.
        """
        if self.window_days is None:
            return False
        days = commitment.days_until_due
        if days is None or days < 0:
            return False
        return days > self.window_days


async def build_agenda(
    gum_instance: gum,
    *,
    limit: int | None = DEFAULT_LIMIT,
    window_days: int | None = None,
    min_confidence: int | None = None,
    now: datetime | None = None,
) -> list[Commitment]:
    """Convenience one-shot: build a ranked commitment radar over *gum_instance*.

    Thin wrapper the CLI / REST / MCP surfaces call so they don't each repeat the
    engine construction. See :class:`CommitmentRadar` for the knobs.
    """
    radar = CommitmentRadar(
        gum_instance,
        min_confidence=min_confidence,
        limit=limit,
        window_days=window_days,
    )
    return await radar.build(now=now)
