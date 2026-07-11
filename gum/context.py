# context.py
#
# The single GUM context-assembly path shared by every machine-facing surface
# that grounds an agent in what the GUM knows about the user. The MCP server
# (`gum/mcp_server.py`) exposes it as the `gather_context` tool; the execution
# bridge (`gum/executor.py`) reuses the *same* assembly to ground a dispatched
# task before handing it to a sandboxed agent. Spec #4 is explicit that these two
# must not fork a second grounding path — so the retrieval, egress
# pseudonymization, and topic->pseudo-ID bridging all live here, once.
#
# This module is deliberately FastMCP-free: it takes a live `gum` instance and an
# optional sanitizer and returns plain dicts/strings, so it can be driven from an
# MCP tool, the executor, a test, or anything else without pulling in a server.

from __future__ import annotations

import asyncio
import re
from typing import Any

# Reuse the exact serialization + egress-sanitization the REST API and MCP use so
# every machine-facing surface exposes propositions identically.
from .api import _serialize_proposition
from .gum import gum

# An agent calls context assembly with a *task instruction* ("draft a grant
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


async def gather_context(
    gum_instance: gum,
    topic: str,
    *,
    sanitizer: Any = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Assemble the GUM propositions relevant to *topic* for an agent to act on.

    Pass a short natural-language description of the task ("draft a grant proposal
    for the Schmidt Foundation"). Retrieval runs on the substantive terms only
    (see :func:`_focus_terms`); an empty topic degrades to the user's most recent
    propositions. When *sanitizer* is provided, proposition text is pseudonymized
    on egress and a ``query_aliases`` map (real entity in the topic -> pseudo-ID)
    is returned so the caller can tell which pseudonymized propositions concern
    the entities it named. This is the one assembly the MCP tool and the executor
    both use; it never mutates the GUM.
    """
    topic = (topic or "").strip()
    limit = max(1, min(int(limit), 50))
    search_terms = ""
    query_aliases: dict[str, str] = {}
    if not topic:
        # An empty topic degrades to "what is the user up to lately", which is a
        # reasonable default rather than an error for an agent probe.
        props = await gum_instance.recent(limit=limit)
        items = [await _serialize_proposition(p, sanitizer) for p in props]
    else:
        # Retrieve on the substantive terms only: an agent passes a whole task
        # instruction, and its verbs/stopwords would otherwise match unrelated
        # propositions under OR-mode BM25.
        search_terms = _focus_terms(topic)
        results = await gum_instance.query(search_terms, limit=limit)
        items = [
            await _serialize_proposition(p, sanitizer, score)
            for p, score in results
        ]
        # Bridge the task->context gap: the returned propositions are
        # pseudonymized, so an entity the agent named in `topic` (e.g. "Schmidt")
        # shows up in the context as a pseudo-ID (e.g. "[ORG_1]"). Expose how the
        # topic's own entities map to those pseudo-IDs so the caller can tell
        # which pseudonymized propositions actually concern the thing it asked
        # about. This leaks nothing new — the values come from the caller's own
        # topic, so it already knows them.
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


def render_context(result: dict[str, Any]) -> str:
    """Render a :func:`gather_context` result into a grounding text block.

    The block is meant to be embedded in the instruction handed to a dispatched
    agent (the executor's backend), which may be off-device — so it renders only
    the (already pseudonymized) proposition text plus the pseudonymization
    contract, and deliberately omits the ``query_aliases`` real->pseudo map: that
    map contains raw entity names and belongs in structured output for a caller
    that already knows them, not in a prompt bound for a frontier model. Returns
    a short "nothing known" line when no propositions matched, so a thin GUM
    produces an honest prompt rather than an empty one.
    """
    props = result.get("propositions") or []
    if not props:
        return "(The user's GUM has no confident context relevant to this task.)"

    lines = ["## What the user's GUM knows that is relevant to this task"]
    if result.get("sanitized"):
        lines.append(
            "(Content is pseudonymized: placeholders like [PERSON_1] / [ORG_1] "
            "each stand for one real entity. Keep them verbatim; never guess the "
            "real value behind one.)"
        )
    for i, p in enumerate(props, 1):
        conf = p.get("confidence")
        conf_str = conf if conf is not None else "?"
        lines.append(f"{i}. (confidence {conf_str}/10) {(p.get('text') or '').strip()}")
    return "\n".join(lines)
