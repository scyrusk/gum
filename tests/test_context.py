# test_context.py
#
# Stdlib-only (unittest) tests for the shared GUM context-assembly module
# (gum/context.py). Runnable without pytest or a live model:
#     python -m unittest tests.test_context
#
# This module is the single grounding path both the MCP server and the execution
# bridge use (spec #4). These tests drive it directly against a real temp
# database and assert the retrieval, egress pseudonymization, and the rendered
# grounding block a dispatched agent receives.

from __future__ import annotations

import tempfile
import unittest
import uuid

from gum import gum as Gum
from gum.context import _focus_terms, gather_context, render_context
from gum.models import Proposition


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
    """Deterministic stand-in for the PII model (no torch/transformers needed)."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def sanitize(self, text: str) -> str:
        return self.sanitize_map(text)[0]

    def sanitize_map(self, text: str) -> tuple[str, dict[str, str]]:
        aliases: dict[str, str] = {}
        for raw, pseudo in self._mapping.items():
            if raw in text:
                aliases[raw] = pseudo
            text = text.replace(raw, pseudo)
        return text, aliases


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
    async def test_focus_terms_strip_instruction_verbs(self):
        self.assertEqual(
            _focus_terms("draft a grant proposal for the Schmidt Foundation"),
            "grant proposal schmidt foundation",
        )
        # All-stopword topic falls back to the raw terms rather than emptying.
        self.assertEqual(_focus_terms("please help me write it"), "please help me write it")

    async def test_gather_returns_only_relevant_propositions(self):
        await self._seed(
            _prop("Omar is applying for a Schmidt Foundation research grant on privacy", 9),
            _prop("Omar prefers dark roast coffee in the morning", 6),
        )
        result = await gather_context(self.gum, "Schmidt grant proposal")

        self.assertEqual(result["count"], 1)
        self.assertFalse(result["sanitized"])
        self.assertEqual(result["query_aliases"], {})
        self.assertIn("Schmidt Foundation research grant", result["propositions"][0]["text"])

    async def test_empty_topic_falls_back_to_recent(self):
        await self._seed(_prop("Omar drafted an outline today", 7))
        result = await gather_context(self.gum, "   ")
        self.assertEqual(result["topic"], "")
        self.assertEqual(result["count"], 1)

    async def test_limit_is_clamped(self):
        await self._seed(*[_prop(f"Omar did task {i}", 5) for i in range(5)])
        result = await gather_context(self.gum, "task", limit=999)
        self.assertLessEqual(result["count"], 50)

    async def test_sanitizer_pseudonymizes_and_bridges_aliases(self):
        await self._seed(_prop("Omar is applying for a Schmidt Foundation grant", 9))
        fake = _FakeSanitizer({"Schmidt": "[ORG_1]", "Omar": "[PERSON_1]"})

        result = await gather_context(
            self.gum, "grant proposal for Schmidt", sanitizer=fake
        )

        self.assertTrue(result["sanitized"])
        text = result["propositions"][0]["text"]
        self.assertIn("[ORG_1]", text)
        self.assertNotIn("Schmidt", text)
        # Only the entities named in the topic are bridged to their pseudo-IDs.
        self.assertEqual(result["query_aliases"], {"Schmidt": "[ORG_1]"})


class RenderContextTests(unittest.TestCase):
    def test_render_lists_propositions_with_confidence(self):
        result = {
            "sanitized": False,
            "query_aliases": {},
            "propositions": [
                {"text": "Omar is applying for a Schmidt grant", "confidence": 9},
                {"text": "Omar prefers dark roast", "confidence": 7},
            ],
        }
        text = render_context(result)
        self.assertIn("(confidence 9/10) Omar is applying for a Schmidt grant", text)
        self.assertIn("(confidence 7/10) Omar prefers dark roast", text)

    def test_render_empty_is_honest(self):
        text = render_context({"propositions": [], "sanitized": False})
        self.assertIn("no confident context", text.lower())

    def test_render_flags_pseudonymization_but_omits_raw_aliases(self):
        # The rendered block may go to an off-device agent, so it flags the
        # pseudonymization contract but must NOT leak the real->pseudo map (which
        # carries raw entity names). Only pseudonymized text reaches the prompt.
        result = {
            "sanitized": True,
            "query_aliases": {"Schmidt": "[ORG_1]"},
            "propositions": [{"text": "[PERSON_1] is applying to [ORG_1]", "confidence": 9}],
        }
        text = render_context(result)
        self.assertIn("pseudonymized", text.lower())
        self.assertIn("[ORG_1]", text)
        self.assertNotIn("Schmidt", text)


if __name__ == "__main__":
    unittest.main()
