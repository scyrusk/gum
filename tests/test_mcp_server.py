# test_mcp_server.py
#
# Stdlib-only (unittest) tests for the MCP server that exposes the GUM to a local
# executing agent (paper: the "gumcp"). Runnable without pytest or a live model:
#     python -m unittest tests.test_mcp_server
#
# These drive the FastMCP tools end-to-end against a real temp database and assert
# both what an agent would receive (the structured result) and the egress
# sanitization contract that keeps raw PII off-device.

from __future__ import annotations

import tempfile
import unittest
import uuid

from gum import gum as Gum
from gum.models import Proposition
from gum.mcp_server import build_mcp


def _prop(text: str, confidence: int) -> Proposition:
    return Proposition(
        text=text,
        reasoning=f"because of {text}",
        confidence=confidence,
        decay=5,
        revision_group=uuid.uuid4().hex,
        version=1,
    )


class _FakeSanitizer:
    """Deterministic stand-in for the PII model, so tests need no torch/transformers."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def sanitize(self, text: str) -> str:
        for raw, pseudo in self._mapping.items():
            text = text.replace(raw, pseudo)
        return text


async def _call(mcp, name: str, args: dict) -> dict:
    """Invoke a FastMCP tool and return its structured (dict) result."""
    _blocks, structured = await mcp.call_tool(name, args)
    return structured


class _Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db")
        await self.gum.connect_db()

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    async def _seed(self, *props: Proposition) -> None:
        async with self.gum._session() as s:
            s.add_all(list(props))


class GatherContextTests(_Base):
    async def test_gather_returns_only_relevant_propositions(self):
        await self._seed(
            _prop("Omar is applying for a Schmidt Foundation research grant on privacy", 9),
            _prop("Omar prefers dark roast coffee in the morning", 6),
        )
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "gather_context", {"topic": "Schmidt grant proposal"})

        self.assertEqual(result["count"], 1)
        self.assertFalse(result["sanitized"])
        texts = [p["text"] for p in result["propositions"]]
        self.assertIn("Schmidt Foundation research grant", texts[0])
        # BM25 relevance ranking exposes a score for the agent to weigh.
        self.assertIn("score", result["propositions"][0])

    async def test_empty_topic_falls_back_to_recent(self):
        await self._seed(_prop("Omar drafted an outline today", 7))
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "gather_context", {"topic": "   "})

        self.assertEqual(result["topic"], "")
        self.assertEqual(result["count"], 1)

    async def test_limit_is_clamped(self):
        await self._seed(*[_prop(f"Omar did task {i}", 5) for i in range(5)])
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "gather_context", {"topic": "task", "limit": 999})

        self.assertLessEqual(result["count"], 50)


class RecentContextTests(_Base):
    async def test_recent_returns_latest_propositions(self):
        await self._seed(
            _prop("Omar reviewed a paper", 6),
            _prop("Omar answered email", 5),
        )
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "recent_context", {"limit": 10})

        self.assertEqual(result["count"], 2)
        self.assertFalse(result["sanitized"])


class SanitizationTests(_Base):
    async def test_pii_is_pseudonymized_on_egress(self):
        # The whole point of the gumcp: an external agent must never see raw
        # identities. build_mcp(sanitize=True) loads the sanitizer fail-closed;
        # here we substitute a deterministic fake for it.
        import gum.sanitize as sanitize_mod

        fake = _FakeSanitizer({"Schmidt": "[ORG_1]", "Omar": "[PERSON_1]"})
        original = sanitize_mod.get_sanitizer
        sanitize_mod.get_sanitizer = lambda: fake
        try:
            await self._seed(
                _prop("Omar is applying for a Schmidt Foundation grant", 9),
            )
            mcp = build_mcp(self.gum, sanitize=True)
            result = await _call(mcp, "gather_context", {"topic": "Schmidt grant"})
        finally:
            sanitize_mod.get_sanitizer = original

        self.assertTrue(fake.loaded)  # loaded eagerly (fail-closed) at build time
        self.assertTrue(result["sanitized"])
        text = result["propositions"][0]["text"]
        self.assertIn("[ORG_1]", text)
        self.assertIn("[PERSON_1]", text)
        self.assertNotIn("Schmidt", text)
        self.assertNotIn("Omar", text)


class ToolAdvertisingTests(_Base):
    async def test_tools_are_advertised_to_clients(self):
        mcp = build_mcp(self.gum, sanitize=False)
        names = {t.name for t in await mcp.list_tools()}
        self.assertEqual(names, {"gather_context", "recent_context"})


if __name__ == "__main__":
    unittest.main()
