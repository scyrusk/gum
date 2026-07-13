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

import asyncio
import logging
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from typing import Any

from .gum import gum
from .llm import structured_completion
from .models import Proposition
from .prompts.gum import AGENDA_PROMPT, AGENDA_VERIFY_PROMPT
from .schemas import CommitmentItem, CommitmentSchema, CommitmentVerdictSchema

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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


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


def _dedupe_key(title: str) -> str:
    """Normalize a commitment title for near-duplicate detection.

    The GUM re-infers overlapping propositions about the same underlying task
    (several notes each implying "submit the NSF grant"), so the extractor can
    surface one real-world commitment more than once with slightly reworded
    titles. Collapsing to lowercase alphanumeric tokens with single spaces folds
    that cosmetic drift together ("Submit the NSF grant!" == "submit  the  NSF
    grant") while keeping genuinely different commitments apart. Returns "" when the
    title has no comparable content, which callers treat as "never a duplicate".
    """
    stripped = re.sub(r"[^a-z0-9 ]+", "", title.lower())
    return re.sub(r"\s+", " ", stripped).strip()


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


# Propositions state deadlines as absolute ``YYYY-MM-DD`` calendar dates (see
# ``PROPOSE_PROMPT``'s temporal-grounding rule), so a dated commitment can be
# recovered from the proposition text with a plain regex — no model call. Shared
# by the deterministic MCP ``upcoming_deadlines`` scan and by the agenda-edit
# due-date rewrite (:func:`rewrite_due_date`), which both need to find the one
# canonical date a proposition carries.
_ABS_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _extract_dates(text: str) -> list[date]:
    """Parse every valid absolute ``YYYY-MM-DD`` date out of *text*.

    Malformed matches (e.g. ``2026-13-40``) are skipped rather than raising, so a
    stray number that merely looks like a date can never break the scan.
    """
    out: list[date] = []
    for m in _ABS_DATE_RE.finditer(text or ""):
        try:
            out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    return out


def rewrite_due_date(text: str, new_iso: str) -> str | None:
    """Rewrite the one absolute date in *text* to *new_iso*, or None if ambiguous.

    The hybrid agenda edit tries to push a corrected due date all the way down
    into the source proposition so the fix is durable (and visible to the MCP
    ``upcoming_deadlines`` scan), not just an overlay. That is only safe when the
    proposition carries **exactly one** absolute ``YYYY-MM-DD`` date: with zero
    dates there is nothing to replace, and with several the mapping from "the
    commitment's deadline" to "which date in the text" is ambiguous — in both
    cases we return None and let the caller fall back to the override + correction
    observation alone. Also None when the single date already equals *new_iso*
    (nothing to change).
    """
    matches = list(_ABS_DATE_RE.finditer(text or ""))
    if len(matches) != 1:
        return None
    m = matches[0]
    if m.group(0) == new_iso:
        return None
    return text[: m.start()] + new_iso + text[m.end() :]


def _today_anchor() -> dict[str, str]:
    """Server-local calendar date, for grounding a client's temporal reasoning.

    A capable frontier agent (or the GUMBO Agenda page) reasoning about the
    absolute ``YYYY-MM-DD`` deadlines the GUM's propositions now carry needs to
    know *what today is* — but an off-device model has a stale knowledge cutoff
    and no reliable sense of the user's local date or timezone, and a browser's
    clock can drift from the machine the GUM ran on. We compute it here, on the
    user's machine, in local time (the frame the screen/calendar observers
    recorded the deadlines in) and hand it back alongside the radar. This leaks
    nothing — a calendar date is not PII.
    """
    now = datetime.now().astimezone()
    return {"date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A")}


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
    # Set when this row is an explicitly-added AgendaItem rather than a
    # model-extracted commitment; the two are mutually exclusive. Lets the UI
    # route edits/dismissals to the /agenda/item/{id} endpoints.
    item_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sort_key_commitment(c: Commitment) -> tuple[float, int, int, int]:
    """Ranking key: most-urgent first, with deterministic tie-breaks.

    Shared by :meth:`CommitmentRadar.build` and :func:`apply_overrides` so a
    radar with user overlays sorts identically to a fresh one. Ties break by
    soonest deadline, then highest confidence, then oldest proposition.
    """
    days = c.days_until_due if c.days_until_due is not None else 10**9
    return (-c.urgency, days, -(c.confidence or 0), c.proposition_id or 0)


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
        verify: bool | None = None,
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
        # Second-pass isolation verification (see :meth:`_verify`). On by default;
        # GUM_AGENDA_VERIFY=0 disables it (used by offline tests that stub only the
        # extraction call).
        self.verify = (
            verify if verify is not None
            else _env_bool("GUM_AGENDA_VERIFY", True)
        )
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

        # Greedy decoding (temperature 0): commitment extraction is a
        # classification task with one right answer, not a creative one. The
        # model's default temperature makes the same GUM state yield a different
        # radar on each refresh — reshuffling a "standing artifact" is confusing —
        # and the sampling noise is exactly what lets an ongoing-activity
        # proposition slip through as a false-positive commitment on some runs.
        # Pinning temperature=0 makes the radar deterministic and picks the
        # model's most-probable classification, matching the other decision calls
        # in gum.gum (blacklist compliance, audit).
        result = await structured_completion(
            self.gum.client,
            self.gum.model,
            [{"role": "user", "content": prompt}],
            CommitmentSchema,
            temperature=0,
            logger=self.logger,
        )

        pairs: list[tuple[CommitmentItem, Proposition]] = []
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
            pairs.append((item, prop))

        # Second-pass verification: the extraction call judges commitments while
        # looking at the whole candidate pool, which pressures the model to
        # promote borderline ongoing activities to fill the list. Re-judging each
        # survivor in isolation (a cleaner binary decision) reliably drops those
        # false positives before ranking.
        if self.verify and pairs:
            pairs = await self._verify(pairs)

        commitments: list[Commitment] = []
        for item, prop in pairs:
            commitment = self._build_commitment(item, prop, now)
            if self._outside_window(commitment):
                continue
            commitments.append(commitment)

        # Rank most-urgent first, with deterministic tie-breaks so equal-urgency
        # items order sensibly (soonest deadline, then highest confidence, then
        # oldest proposition) instead of by arbitrary extraction order.
        commitments.sort(key=_sort_key_commitment)
        commitments = self._dedupe(commitments)
        return commitments[: self.limit] if self.limit is not None else commitments

    async def _verify(
        self, pairs: list[tuple[CommitmentItem, Proposition]]
    ) -> list[tuple[CommitmentItem, Proposition]]:
        """Re-judge each extracted commitment in isolation, dropping false positives.

        Each candidate is verified independently (concurrently) with a focused
        binary "is this a genuine discrete commitment, or an ongoing activity?"
        call. This is where the ongoing-habit false positives — the ones the
        pooled extraction over-promotes — get filtered out. Fails *open* per item:
        a verification error keeps the commitment, so a transient model hiccup
        never silently empties the radar.
        """

        async def judge(item: CommitmentItem, prop: Proposition) -> bool:
            prompt = (
                AGENDA_VERIFY_PROMPT
                .replace("{user_name}", self.gum.user_name)
                .replace("{proposition}", prop.text.strip())
                .replace("{title}", (item.title or "").strip())
            )
            try:
                verdict = await structured_completion(
                    self.gum.client,
                    self.gum.model,
                    [{"role": "user", "content": prompt}],
                    CommitmentVerdictSchema,
                    temperature=0,  # a classification, not a creative task
                    logger=self.logger,
                )
                keep = bool(getattr(verdict, "is_commitment", True))
            except Exception as exc:  # noqa: BLE001 — fail open, never crash the radar
                self.logger.debug(
                    "agenda: verification error for %r, keeping it: %s",
                    item.title, exc,
                )
                return True
            if not keep:
                self.logger.debug(
                    "agenda: verification dropped ongoing-activity %r", item.title
                )
            return keep

        verdicts = await asyncio.gather(*(judge(i, p) for i, p in pairs))
        return [pair for pair, keep in zip(pairs, verdicts) if keep]

    def _dedupe(self, commitments: list[Commitment]) -> list[Commitment]:
        """Drop near-duplicate commitments, keeping the most-urgent instance.

        Assumes *commitments* is already ranked, so the first occurrence of each
        normalized title is the one worth surfacing. Titles that normalize to
        empty are never treated as duplicates of one another.
        """
        seen: set[str] = set()
        deduped: list[Commitment] = []
        for c in commitments:
            key = _dedupe_key(c.title)
            if key and key in seen:
                self.logger.debug("agenda: dropping duplicate commitment %r", c.title)
                continue
            if key:
                seen.add(key)
            deduped.append(c)
        return deduped

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
    verify: bool | None = None,
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
        verify=verify,
    )
    return await radar.build(now=now)


# ── user overrides (GUMBO Agenda page) ─────────────────────────────────────────
#
# The agenda has no persistent row to edit — it is re-extracted by the local
# model on every request — so the user's direct edits live in the
# `agenda_overrides` table (see gum.models.AgendaOverride) and are overlaid here,
# on top of each freshly-built radar, before it reaches the UI. This keeps
# `build_agenda` (and thus the CLI + MCP surfaces) pure; overrides are a
# REST/desktop concern only. Propagation of the edit *into* the model (proposition
# rewrite + correction observation) is handled separately in gum.gum.


def _truncate_title(text: str, limit: int = 80) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _age_days(created_at_iso: str | None, now: datetime) -> float:
    """Age of a proposition (in days) from its ISO ``created_at``, clamped ≥ 0."""
    if not created_at_iso:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at_iso)
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (now - created).total_seconds() / 86400.0)


def _apply_override_fields(c: Commitment, ov: dict[str, Any], now: datetime) -> Commitment:
    """Return *c* with the user's overridden title/status/due-date applied.

    Only fields the user actually set replace the model's values. When the due
    date changes (including being cleared to "no fixed date"), ``days_until_due``
    and ``urgency`` are recomputed against *now* so the overlaid item ranks and
    buckets correctly.
    """
    title = ov["title"] if ov.get("title") else c.title
    status = ov["status"] if ov.get("status") else c.status_guess
    due_date = c.due_date
    days = c.days_until_due
    urgency = c.urgency
    date_changed = False

    if ov.get("due_date_cleared"):
        if c.due_date is not None:
            due_date, days, date_changed = None, None, True
    elif ov.get("due_date") and ov["due_date"] != c.due_date:
        due_date, date_changed = ov["due_date"], True

    if date_changed:
        parsed = _parse_due(due_date)
        due_date = parsed.isoformat() if parsed is not None else None
        days = (parsed - now.date()).days if parsed is not None else None
        urgency = round(
            urgency_score(days, c.confidence, c.decay, _age_days(c.created_at, now)), 4
        )

    return replace(
        c,
        title=title,
        status_guess=status,
        due_date=due_date,
        days_until_due=days,
        urgency=urgency,
    )


def _synthesize_commitment(ov: dict[str, Any], snap: dict[str, Any], now: datetime) -> Commitment:
    """Reconstruct a Commitment for an override the model didn't surface this run.

    The local model re-extracts the radar every load and may simply not emit a
    commitment for a proposition the user has already edited. Without this the
    user's edit would appear to vanish on refresh; instead we rebuild the item
    from the live proposition snapshot (*snap*) plus the overridden fields, so the
    edit persists visually until re-inference catches up.
    """
    text = (snap.get("text") or "").strip()
    title = ov["title"] if ov.get("title") else _truncate_title(text)
    status = ov.get("status") or "unknown"
    due_raw = None if ov.get("due_date_cleared") else ov.get("due_date")
    parsed = _parse_due(due_raw)
    days = (parsed - now.date()).days if parsed is not None else None
    created_at = snap.get("created_at")
    conf = snap.get("confidence")
    decay = snap.get("decay")
    urgency = round(urgency_score(days, conf, decay, _age_days(created_at, now)), 4)
    return Commitment(
        title=title,
        due_date=parsed.isoformat() if parsed is not None else None,
        source="you",
        status_guess=status,
        confidence=conf,
        decay=decay,
        days_until_due=days,
        urgency=urgency,
        proposition_id=ov["proposition_id"],
        proposition_text=text,
        created_at=created_at,
    )


def apply_overrides(
    commitments: list[Commitment],
    overrides: list[dict[str, Any]],
    *,
    now: datetime,
    limit: int | None = None,
) -> list[Commitment]:
    """Overlay the user's persisted agenda edits on a freshly-extracted radar.

    Pure function (no DB access): *overrides* are plain detached dicts as produced
    by ``gum.list_agenda_overrides`` — each carries the override fields plus a
    ``prop`` snapshot (``id``/``text``/``confidence``/``decay``/``created_at``, or
    None if the proposition was since deleted).

    Each surfaced commitment is matched to an override by ``proposition_id``. A
    matched override either drops the item (``dismissed``) or overlays its fields.
    Overrides the model didn't surface this run are reconstructed from their
    proposition snapshot so the edit still shows. Items with no ``proposition_id``
    can't be keyed and pass through untouched. The result is re-sorted and
    re-limited so overlays and reconstructions land in the right place.
    """
    by_id: dict[int, dict[str, Any]] = {}
    for ov in overrides:
        pid = ov.get("proposition_id")
        if pid is not None:
            by_id[pid] = ov

    used: set[int] = set()
    out: list[Commitment] = []
    for c in commitments:
        ov = None
        if c.proposition_id is not None:
            ov = by_id.get(c.proposition_id)
        if ov is None:
            out.append(c)
            continue
        used.add(ov["proposition_id"])
        if ov.get("dismissed"):
            continue  # user removed this item from the radar
        out.append(_apply_override_fields(c, ov, now))

    # Overrides the model didn't surface this run: reconstruct so the edit sticks.
    for ov in overrides:
        if ov["proposition_id"] in used or ov.get("dismissed"):
            continue
        snap = ov.get("prop")
        if not snap:
            continue  # proposition gone; nothing to rebuild from
        out.append(_synthesize_commitment(ov, snap, now))

    out.sort(key=_sort_key_commitment)
    return out[:limit] if limit is not None else out


# High salience for explicitly-added items: someone deliberately put them on the
# agenda, so they should rank among the model's own commitments rather than being
# buried. Standing in for the confidence/decay a proposition would carry.
_ADDED_ITEM_SALIENCE = 10


def agenda_item_to_commitment(item: dict[str, Any], now: datetime) -> Commitment:
    """Turn a stored :class:`gum.models.AgendaItem` (as a dict) into a Commitment.

    Added items are already in the user's real terms (rehydrated on ingest), carry
    an ``item_id`` rather than a ``proposition_id``, and rank as high-salience
    since they were put on the agenda on purpose. ``now`` is the same local anchor
    the rest of the radar uses so day math stays consistent.
    """
    due = _parse_due(item.get("due_date"))
    days = (due - now.date()).days if due is not None else None
    urgency = round(
        urgency_score(days, _ADDED_ITEM_SALIENCE, _ADDED_ITEM_SALIENCE, 0.0), 4
    )
    title = (item.get("title") or "").strip()
    return Commitment(
        title=title,
        due_date=due.isoformat() if due is not None else None,
        source=(item.get("source") or "added").strip() or "added",
        status_guess=(item.get("status") or "unknown"),
        confidence=None,
        decay=None,
        days_until_due=days,
        urgency=urgency,
        proposition_id=None,
        proposition_text=title,
        created_at=item.get("created_at"),
        item_id=item["id"],
    )
