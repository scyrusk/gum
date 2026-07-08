# test_api_suggestions.py
#
# Stdlib-only (unittest) tests for the GUMBO `/suggestions` REST endpoint.
# Runnable without a live model:  python -m unittest tests.test_api_suggestions
#
# The text model is stubbed (patched structured_completion), so these tests
# drive the read-only FastAPI app end-to-end through a TestClient while staying
# fully offline and deterministic.

from __future__ import annotations

import tempfile
import unittest
import uuid
from unittest import mock

from fastapi.testclient import TestClient

from gum import gum as Gum
from gum.api import create_app
from gum.models import Proposition
from gum.schemas import SuggestionItem, SuggestionSchema


def _prop(text: str, confidence: int) -> Proposition:
    return Proposition(
        text=text,
        reasoning=f"because of {text}",
        confidence=confidence,
        decay=5,
        revision_group=uuid.uuid4().hex,
        version=1,
    )


# One high-value/low-intrusion suggestion (surfaced) and one noisy one (withheld).
_FAKE_SUGGESTIONS = SuggestionSchema(suggestions=[
    SuggestionItem(
        title="Rent a suit in Chicago",
        description="Found three suit-rental shops near the venue.",
        rationale="Wedding + no formal wear.",
        probability_useful=9, benefit=9, cost_if_wrong=2, cost_if_missed=7,
    ),
    SuggestionItem(
        title="Reorganize your desktop icons",
        description="Tidy the desktop.",
        rationale="Loosely related.",
        probability_useful=2, benefit=2, cost_if_wrong=8, cost_if_missed=1,
    ),
])


async def _fake_completion(client, model, messages, schema, **kwargs):
    return _FAKE_SUGGESTIONS


class SuggestionsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db")
        await self.gum.connect_db()
        async with self.gum._session() as s:
            s.add_all([
                _prop("Omar is going to a friend's wedding in Chicago", 8),
                _prop("Omar doesn't own suitable formal wear", 7),
                _prop("Omar is idly browsing social media", 3),  # below the bar
            ])

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def test_suggestions_ranked_and_scored(self):
        app = create_app(self.gum)
        with mock.patch("gum.gumbo.structured_completion", side_effect=_fake_completion):
            with TestClient(app) as client:
                resp = client.get("/suggestions")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        sugs = body["suggestions"]
        # Ranked by expected utility: the high-value one is first and surfaced.
        self.assertEqual(len(sugs), 2)
        self.assertEqual(sugs[0]["title"], "Rent a suit in Chicago")
        self.assertTrue(sugs[0]["should_surface"])
        self.assertFalse(sugs[1]["should_surface"])
        self.assertGreaterEqual(sugs[0]["expected_utility"], sugs[1]["expected_utility"])
        # The scored fields are present for a client UI to render.
        for key in ("probability_useful", "benefit", "cost_if_wrong", "cost_if_missed"):
            self.assertIn(key, sugs[0])

    def test_surfaced_only_applies_mixed_initiative_filter(self):
        app = create_app(self.gum)
        with mock.patch("gum.gumbo.structured_completion", side_effect=_fake_completion):
            with TestClient(app) as client:
                resp = client.get("/suggestions", params={"surfaced_only": "true"})
        sugs = resp.json()["suggestions"]
        self.assertEqual(len(sugs), 1)
        self.assertEqual(sugs[0]["title"], "Rent a suit in Chicago")
        self.assertTrue(sugs[0]["should_surface"])

    def test_limit_caps_results(self):
        app = create_app(self.gum)
        with mock.patch("gum.gumbo.structured_completion", side_effect=_fake_completion):
            with TestClient(app) as client:
                resp = client.get("/suggestions", params={"limit": 1})
        self.assertEqual(len(resp.json()["suggestions"]), 1)

    def test_sanitize_scrubs_suggestion_text(self):
        # Under --sanitize, the model-written text is pseudonymized on the way out
        # while numeric scores pass through unchanged.
        fake_sanitizer = mock.Mock()
        fake_sanitizer.load = mock.Mock()
        fake_sanitizer.sanitize = mock.Mock(side_effect=lambda t: t.replace("Chicago", "[CITY]"))
        with mock.patch("gum.sanitize.get_sanitizer", return_value=fake_sanitizer):
            app = create_app(self.gum, sanitize=True)
            with mock.patch("gum.gumbo.structured_completion", side_effect=_fake_completion):
                with TestClient(app) as client:
                    resp = client.get("/suggestions")
        sugs = resp.json()["suggestions"]
        self.assertEqual(sugs[0]["title"], "Rent a suit in [CITY]")
        self.assertNotIn("Chicago", sugs[0]["description"])
        # Numeric scores are untouched by the scrubber.
        self.assertEqual(sugs[0]["probability_useful"], 9)


if __name__ == "__main__":
    unittest.main()
