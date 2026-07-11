# mcp_server.py
#
# A localhost Model Context Protocol (MCP) server that lets a *local executing
# agent* — Claude Desktop, Codex, or any other MCP client running on the user's
# machine — pull context out of the user's General User Model on demand.
#
# The motivating flow: the user asks their agent to "draft a grant proposal for
# Schmidt". The agent, before writing, calls the `gather_context` tool exposed
# here; the GUM returns the propositions it has learned about the user that are
# relevant to that topic (their research area, prior funders, writing style,
# deadlines, …) so the agent can act with real personal context instead of
# guessing — all without the user pasting anything by hand.
#
# Because the consuming agent may relay this context to a frontier model off the
# device, sanitization is ON by default and fail-closed: PII is replaced with
# consistent pseudo-IDs on the way out (see sanitize.py), and if the sanitizer
# cannot load the server refuses to start rather than leak raw identities. A
# fully-local, trusted agent can opt out with `gum mcp --no-sanitize`.
#
# The server is read-only and speaks MCP over stdio, the transport every local
# MCP client already knows how to launch and talk to.

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

# Reuse the exact serialization + egress-sanitization the REST API uses so the
# two machine-facing surfaces expose propositions identically.
from .api import (
    _serialize_commitment,
    _serialize_observation,
    _serialize_proposition,
)
from .gum import gum

# An agent calls gather_context with a *task instruction* ("draft a grant
# proposal for the Schmidt Foundation"), not a search query. The BM25 retrieval
# runs in OR mode, so every token in that sentence becomes a candidate filter —
# and the imperative verbs ("draft") and function words ("a", "for", "the")
# match propositions that have nothing to do with the task (e.g. "the user
# frequently drafts emails"). Because the returned score is min-max normalised
# *within* the result set, this noise can't be filtered after the fact; it has
# to be kept out of the query. We drop instruction verbs and stopwords so the
# search runs on the substantive terms ("grant", "Schmidt", "Foundation").
#
# Kept deliberately small and conservative: these are words that carry no signal
# about *which user context is relevant*, never domain nouns. If stripping would
# empty the query we fall back to the raw topic, so a terse topic still works.
_INSTRUCTION_STOPWORDS = frozenset(
    {
        # imperative/task verbs an agent uses to frame what it's about to do
        "draft", "write", "compose", "create", "make", "build", "prepare",
        "generate", "produce", "help", "assist", "summarize", "summarise",
        "outline", "review", "edit", "revise", "rewrite", "update", "send",
        "reply", "respond", "schedule", "plan", "find", "search", "look",
        "get", "fetch", "pull", "give", "tell", "show", "list", "draw",
        "please", "need", "want", "would", "like", "let", "using", "use",
        "based", "regarding", "context",
        # function words (articles, prepositions, conjunctions, pronouns, aux)
        "a", "an", "the", "for", "of", "to", "on", "in", "at", "by", "with",
        "from", "about", "into", "as", "and", "or", "but", "if", "then",
        "this", "that", "these", "those", "it", "its", "my", "me", "i",
        "we", "our", "us", "you", "your", "he", "she", "they", "their",
        "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
        "can", "could", "should", "will", "shall", "may", "might", "must",
        "have", "has", "had", "not", "so", "some", "any", "up",
    }
)


def _today_anchor() -> dict[str, str]:
    """Server-local calendar date, for grounding an off-device agent's temporal
    reasoning.

    A capable frontier agent building a daily agenda from the GUM's propositions
    needs to know *what today is* to reason about the absolute ``YYYY-MM-DD``
    deadlines those propositions now carry (see ``PROPOSE_PROMPT``'s temporal
    grounding) — but an off-device model has a stale knowledge cutoff and no
    reliable sense of the user's local date or timezone. We compute it here, on
    the user's machine, in local time (the frame the screen/calendar observers
    recorded the deadlines in) and hand it to the agent alongside the radar so it
    never has to guess. This leaks nothing — a calendar date is not PII.
    """
    now = datetime.now().astimezone()
    return {"date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A")}


# Propositions now state deadlines as absolute ``YYYY-MM-DD`` calendar dates (see
# ``PROPOSE_PROMPT``'s temporal-grounding rule), so an off-device agent's dated
# commitments can be recovered from the proposition text with a plain regex — no
# model call, and complete coverage of whatever the GUM actually wrote down.
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


def _focus_terms(topic: str) -> str:
    """Reduce a task instruction to its substantive search terms.

    Removes instruction verbs and stopwords (see ``_INSTRUCTION_STOPWORDS``) that
    would otherwise let an OR-mode BM25 search match propositions unrelated to
    the task. Falls back to the original ``topic`` when filtering leaves nothing,
    so a one-word or all-stopword topic still returns something.
    """
    tokens = re.findall(r"\w+", topic.lower())
    kept = [t for t in tokens if t not in _INSTRUCTION_STOPWORDS]
    return " ".join(kept) if kept else topic

_INSTRUCTIONS = (
    "Access to the user's General User Model (GUM): a private, continuously "
    "updated model of what this user does, needs, and prefers, expressed as "
    "natural-language propositions with confidence scores.\n\n"
    "Before carrying out a task that depends on knowing something about the "
    "user — drafting a document in their voice, referencing their projects, "
    "collaborators, deadlines, or preferences — call `gather_context` with a "
    "short description of the task to retrieve the relevant propositions, then "
    "ground your work in them. Use `recent_context` to see what the user has "
    "been doing lately. Use `agenda` to see the user's ranked open commitments "
    "and deadlines before scheduling, drafting, or planning on their behalf. "
    "Use `upcoming_deadlines` to get the raw, complete set of propositions that "
    "carry an absolute calendar deadline (a deterministic date scan that "
    "complements `agenda`'s local-model extraction). "
    "Higher `confidence` (1-10) means the model is more "
    "certain. Each proposition carries an `id`; call `inspect_proposition` with "
    "it to see the raw observations the model inferred it from when you need the "
    "underlying evidence to ground your work. Content may be pseudonymized "
    "(e.g. [PERSON_1]); treat each pseudo-ID as a stable stand-in for one real "
    "entity. When the pseudonymized context refers to an entity you named in the "
    "task, `gather_context` returns a `query_aliases` map (real name -> pseudo-ID) "
    "so you can tell which propositions concern it.\n\n"
    "Keep pseudo-IDs verbatim in whatever you produce and never guess the real "
    "value behind one. When a response is `sanitized`, the artifact you build from "
    "it still carries those placeholders and is not usable until the user restores "
    "the real names on-device with `gum rehydrate <file>`; save your output to a "
    "file and point the user at that command.\n\n"
    "The `with_user_context` prompt packages this whole workflow — gather "
    "context, optionally inspect evidence, then execute — for a given task; "
    "clients can offer it to the user as a one-shot action. The `daily_agenda` "
    "prompt packages a related workflow — pull the deadline radar and recent "
    "context, then synthesize the user's prioritized agenda for the day — for a "
    "morning briefing or 'what should I focus on today?'.\n\n"
    "The `agenda` tool's response includes a `today` field (the user's real "
    "local date); when reasoning about any commitment's `due_date` or "
    "`days_until_due`, anchor on that rather than your own sense of the date."
)


def build_mcp(gum_instance: gum, *, sanitize: bool = True) -> FastMCP:
    """Build the GUM MCP server over a live ``gum`` instance.

    When *sanitize* is True (the default) the sanitizer is loaded eagerly
    (fail-closed): if the model or its dependencies are missing this raises so
    the server never starts up handing raw PII to an external agent.
    """
    sanitizer = None
    if sanitize:
        from .sanitize import get_sanitizer

        sanitizer = get_sanitizer()
        sanitizer.load()

    @asynccontextmanager
    async def lifespan(_server: FastMCP):
        # Connect the database inside the server's own event loop. connect_db is
        # idempotent and lazy, so this is safe whether or not the caller already
        # connected. No observers/batcher run here — the MCP surface is read-only.
        await gum_instance.connect_db()
        yield {}

    mcp = FastMCP("gum-context", instructions=_INSTRUCTIONS, lifespan=lifespan)

    @mcp.tool(
        description=(
            "Retrieve propositions from the user's General User Model that are "
            "relevant to a task or topic, so you can act with the user's real "
            "context. Pass a short natural-language description of what you are "
            "about to do (e.g. 'draft a grant proposal for the Schmidt "
            "Foundation'). Returns the most relevant propositions ranked by a "
            "text-relevance score, each with the model's confidence (1-10)."
        )
    )
    async def gather_context(topic: str, limit: int = 10) -> dict[str, Any]:
        topic = (topic or "").strip()
        limit = max(1, min(int(limit), 50))
        search_terms = ""
        query_aliases: dict[str, str] = {}
        if not topic:
            # An empty topic degrades to "what is the user up to lately", which
            # is a reasonable default rather than an error for an agent probe.
            props = await gum_instance.recent(limit=limit)
            items = [await _serialize_proposition(p, sanitizer) for p in props]
        else:
            # Retrieve on the substantive terms only: an agent passes a whole
            # task instruction, and its verbs/stopwords would otherwise match
            # unrelated propositions under OR-mode BM25.
            search_terms = _focus_terms(topic)
            results = await gum_instance.query(search_terms, limit=limit)
            items = [
                await _serialize_proposition(p, sanitizer, score)
                for p, score in results
            ]
            # Bridge the task->context gap: the returned propositions are
            # pseudonymized, so an entity the agent named in `topic` (e.g.
            # "Schmidt") shows up in the context as a pseudo-ID (e.g. "[ORG_1]").
            # Expose how the topic's own entities map to those pseudo-IDs so the
            # agent can tell which pseudonymized propositions actually concern the
            # thing it was asked about. This leaks nothing new — the values come
            # from the caller's own topic, so it already knows them.
            if sanitizer is not None:
                _, query_aliases = await asyncio.to_thread(
                    sanitizer.sanitize_map, topic
                )
        return {
            "topic": topic,
            "search_terms": search_terms,
            "query_aliases": query_aliases,
            "count": len(items),
            "propositions": items,
            "sanitized": sanitizer is not None,
        }

    @mcp.tool(
        description=(
            "List the user's most recent propositions — a snapshot of what the "
            "General User Model has learned lately. Useful for orienting before "
            "a task when you have no specific topic to search for."
        )
    )
    async def recent_context(limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(int(limit), 50))
        props = await gum_instance.recent(limit=limit)
        return {
            "count": len(props),
            "propositions": [
                await _serialize_proposition(p, sanitizer) for p in props
            ],
            "sanitized": sanitizer is not None,
        }

    @mcp.prompt(
        name="with_user_context",
        title="Do a task with the user's GUM context",
        description=(
            "Wrap a task so it is carried out with context drawn from the user's "
            "General User Model. Use this when the task depends on knowing "
            "something about the user (their projects, collaborators, funders, "
            "deadlines, writing voice, preferences) — e.g. 'draft a grant "
            "proposal for the Schmidt Foundation'. Expands into an instruction to "
            "gather the relevant GUM propositions first, then act on them."
        ),
    )
    def with_user_context(task: str) -> str:
        # A discoverable entry point for the motivating workflow. Server
        # `instructions` are advisory and not surfaced by every MCP client, so
        # this prompt makes "gather context, then execute" a first-class,
        # user-invocable action that reliably triggers the tool calls.
        return (
            f"Task: {task}\n\n"
            "Before carrying this out, ground yourself in what the user's "
            "General User Model (GUM) knows about them:\n"
            f"1. Call `gather_context` with topic \"{task}\" to retrieve the "
            "relevant propositions (each has a `confidence` from 1-10 — weight "
            "higher-confidence facts more). If the context is pseudonymized, use "
            "the returned `query_aliases` map to see how the entities you named "
            "(e.g. the funder) appear as pseudo-IDs in it.\n"
            "2. For any proposition you intend to rely on but want evidence for, "
            "call `inspect_proposition` with its `id` to read the underlying "
            "observations.\n"
            "3. Then carry out the task, grounded in that context and written in "
            "the user's voice. Do not invent facts the GUM does not support; if "
            "the context is thin, say what you would still need.\n"
            "4. If the context you received was pseudonymized (the tool response's "
            "`sanitized` field is true), the artifact you just produced still "
            "carries pseudo-IDs like [PERSON_1] / [ORG_1] and is NOT yet usable by "
            "the user. Do not guess or fill in the real names yourself. Instead, "
            "save the finished artifact to a file and tell the user to restore the "
            "real values on-device by running `gum rehydrate <file>` — that step "
            "looks the names up in the local, private entity map and never sends "
            "them back to a model.\n\n"
            "Note: GUM content may be pseudonymized (e.g. [PERSON_1], [ORG_1]); "
            "treat each pseudo-ID as a stable stand-in for one real entity and "
            "keep it as-is in anything you produce."
        )

    @mcp.tool(
        description=(
            "Fetch the raw observations that back a single proposition, so you "
            "can ground your work in the underlying evidence rather than the "
            "one-line summary. Pass the `id` of a proposition returned by "
            "`gather_context` or `recent_context`. Returns the proposition plus "
            "its supporting observations (what the user actually did or wrote), "
            "newest first. `found` is false if that proposition no longer exists."
        )
    )
    async def inspect_proposition(
        proposition_id: int, limit: int = 5
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 20))
        result = await gum_instance.proposition_with_observations(
            proposition_id, limit=limit
        )
        if result is None:
            return {
                "found": False,
                "proposition_id": proposition_id,
                "evidence": [],
                "sanitized": sanitizer is not None,
            }
        prop, obs = result
        return {
            "found": True,
            "proposition": await _serialize_proposition(prop, sanitizer),
            "evidence": [
                await _serialize_observation(o, sanitizer) for o in obs
            ],
            "sanitized": sanitizer is not None,
        }

    @mcp.tool(
        description=(
            "Return the user's commitment & deadline radar: a ranked list of the "
            "open commitments and deadlines the General User Model has inferred "
            "(papers, grants, reviews, meetings, payments, promises), most urgent "
            "first. Use this to see what is time-critical for the user before "
            "scheduling, drafting, or planning on their behalf. Each item carries "
            "a `due_date` (or null if undated), `days_until_due` (negative = "
            "overdue), a `status_guess`, the model's `confidence` (1-10), and an "
            "`urgency` rank. Pass `window_days` to only include commitments due "
            "within that horizon (overdue and undated items are always kept)."
        )
    )
    async def agenda(
        limit: int = 10, window_days: int | None = None
    ) -> dict[str, Any]:
        # Import lazily so the (heavy) agenda engine and its LLM/schema deps only
        # load when this tool is actually called, mirroring cmd_agenda.
        from .agenda import build_agenda

        limit = max(1, min(int(limit), 50))
        window = None if window_days is None else max(0, int(window_days))
        commitments = await build_agenda(
            gum_instance, limit=limit, window_days=window
        )
        items = [
            await _serialize_commitment(c, sanitizer) for c in commitments
        ]
        return {
            # The temporal anchor an off-device agent needs to turn each
            # commitment's absolute `due_date` / `days_until_due` into "what is
            # due today / this week" — it cannot reliably know the user's local
            # date itself. See `_today_anchor`.
            "today": _today_anchor(),
            "count": len(items),
            "window_days": window,
            "commitments": items,
            "sanitized": sanitizer is not None,
        }

    @mcp.tool(
        description=(
            "Return the propositions whose text carries an absolute calendar "
            "deadline (a `YYYY-MM-DD` date), scanned deterministically from the "
            "user's General User Model — no model extraction, so it surfaces the "
            "complete, unfiltered set of dated commitments the GUM has recorded. "
            "Each item is a proposition (with its `id`, `text`, `confidence`, and "
            "`created_at`) plus the parsed `deadline` and `days_until_due` "
            "(negative = overdue). Overdue items are always included; upcoming "
            "ones are kept only if within `window_days` (default 30). Sorted "
            "soonest-first. Use this ALONGSIDE `agenda` when building a daily "
            "plan: `agenda` gives the local model's ranked, deduplicated "
            "interpretation (and includes undated commitments), while this gives "
            "you the raw dated signal to reason over yourself and to catch dated "
            "items the extractor may have dropped. Read the returned `today` as "
            "\"now\"."
        )
    )
    async def upcoming_deadlines(
        window_days: int = 30, limit: int = 20
    ) -> dict[str, Any]:
        window = max(0, int(window_days))
        limit = max(1, min(int(limit), 50))
        today = datetime.now().astimezone().date()
        # Scan a generous pool of recent propositions (not just the newest few)
        # so a deadline recorded a while back but due soon isn't missed for
        # falling out of a short recency window. Bounded so a huge GUM can't make
        # this unboundedly expensive.
        scan = min(500, max(200, limit * 20))
        props = await gum_instance.recent(limit=scan)
        matches: list[tuple[int, int, Any, date]] = []
        for p in props:
            dates = _extract_dates(p.text or "")
            if not dates:
                continue
            # Represent each proposition by its most actionable date: the soonest
            # one that is today-or-later, else (all past) the most recent overdue
            # date. A proposition can name several dates ("meet 2026-07-15,
            # deliverable due 2026-07-20"); the nearest upcoming one is what a
            # daily plan turns on.
            future = sorted(d for d in dates if d >= today)
            chosen = future[0] if future else max(dates)
            days = (chosen - today).days
            # Overdue (days < 0) is always kept; upcoming only within the window.
            if days > window:
                continue
            matches.append((days, p.id, p, chosen))
        # Soonest-first (most-overdue first), deterministic id tiebreak. Serialize
        # only the top `limit` so the (per-text) sanitizer runs a bounded number
        # of times regardless of how many propositions carry dates.
        matches.sort(key=lambda t: (t[0], t[1]))
        items: list[dict[str, Any]] = []
        for days, _pid, p, chosen in matches[:limit]:
            item = await _serialize_proposition(p, sanitizer)
            item["deadline"] = chosen.isoformat()
            item["days_until_due"] = days
            items.append(item)
        return {
            "today": _today_anchor(),
            "window_days": window,
            "count": len(items),
            "deadlines": items,
            "sanitized": sanitizer is not None,
        }

    @mcp.prompt(
        name="daily_agenda",
        title="Build the user's daily agenda from their GUM",
        description=(
            "Have a capable agent assemble the user's daily agenda — what is due, "
            "overdue, or worth doing today — grounded in the commitments and "
            "deadlines the user's General User Model has inferred. Use this for "
            "'what should I focus on today?', a morning briefing, or planning the "
            "day. Expands into an instruction to pull the deadline radar and "
            "recent context, then synthesize a prioritized, date-grounded agenda."
        ),
    )
    def daily_agenda(horizon: str = "today") -> str:
        # A discoverable entry point that lets a frontier agent (which extracts
        # and prioritizes far better than the local model) build the agenda
        # itself from the pseudonymized radar + raw dated propositions, rather
        # than just reformatting the local model's `agenda` output. It leans on
        # the `today` anchor the agenda tool returns so temporal reasoning is
        # grounded on-device, not on the agent's stale cutoff.
        return (
            f"Build the user's agenda for: {horizon}.\n\n"
            "Ground it entirely in what the user's General User Model (GUM) "
            "knows — do not invent commitments:\n"
            "1. Call `agenda` to get the ranked commitment & deadline radar. Read "
            "its `today` field (the user's real local date) and use THAT as "
            "\"now\" — not your own sense of the date — to judge each item's "
            "`due_date` / `days_until_due` (negative = overdue).\n"
            "2. Call `upcoming_deadlines` to get the raw, complete set of "
            "propositions carrying an absolute YYYY-MM-DD deadline (the radar in "
            "step 1 runs a lossy local extractor, so this catches dated "
            "commitments it dropped or merged). Reconcile the two by proposition "
            "`id`; each item carries a `deadline`, `days_until_due`, and a "
            "`confidence` (1-10) — weight higher-confidence items more. Optionally "
            "also call `recent_context` for undated but active work worth "
            "advancing.\n"
            "3. For any item you are unsure about, call `inspect_proposition` "
            "with its `id` to read the underlying observations before including "
            "it.\n"
            "4. Synthesize a prioritized agenda for the requested horizon: lead "
            "with overdue and due-today items, then upcoming deadlines, then "
            "high-confidence undated commitments worth advancing. For each, show "
            "the deadline (or 'no date') relative to today and keep it concise.\n\n"
            "GUM content may be pseudonymized (e.g. [PERSON_1], [ORG_1]); treat "
            "each pseudo-ID as a stable stand-in for one real entity and keep it "
            "verbatim — never guess the real value. If the radar was `sanitized` "
            "and you save the agenda to a file, tell the user to restore the real "
            "names on-device with `gum rehydrate <file>`."
        )

    return mcp


def run_stdio(gum_instance: gum, *, sanitize: bool = True) -> None:
    """Run the GUM MCP server over stdio (blocks; owns the event loop)."""
    build_mcp(gum_instance, sanitize=sanitize).run(transport="stdio")
