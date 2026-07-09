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

from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

# Reuse the exact serialization + egress-sanitization the REST API uses so the
# two machine-facing surfaces expose propositions identically.
from .api import _serialize_proposition
from .gum import gum

_INSTRUCTIONS = (
    "Access to the user's General User Model (GUM): a private, continuously "
    "updated model of what this user does, needs, and prefers, expressed as "
    "natural-language propositions with confidence scores.\n\n"
    "Before carrying out a task that depends on knowing something about the "
    "user — drafting a document in their voice, referencing their projects, "
    "collaborators, deadlines, or preferences — call `gather_context` with a "
    "short description of the task to retrieve the relevant propositions, then "
    "ground your work in them. Use `recent_context` to see what the user has "
    "been doing lately. Higher `confidence` (1-10) means the model is more "
    "certain. Content may be pseudonymized (e.g. [PERSON_1]); treat each "
    "pseudo-ID as a stable stand-in for one real entity."
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
        if not topic:
            # An empty topic degrades to "what is the user up to lately", which
            # is a reasonable default rather than an error for an agent probe.
            props = await gum_instance.recent(limit=limit)
            items = [await _serialize_proposition(p, sanitizer) for p in props]
        else:
            results = await gum_instance.query(topic, limit=limit)
            items = [
                await _serialize_proposition(p, sanitizer, score)
                for p, score in results
            ]
        return {
            "topic": topic,
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

    return mcp


def run_stdio(gum_instance: gum, *, sanitize: bool = True) -> None:
    """Run the GUM MCP server over stdio (blocks; owns the event loop)."""
    build_mcp(gum_instance, sanitize=sanitize).run(transport="stdio")
