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
from gum.models import Observation, Proposition
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

    def test_gumbo_page_served(self):
        # The desktop-style GUMBO assistant page is served as static HTML and
        # wires itself to the /suggestions endpoint (project tabs pass `focus`).
        app = create_app(self.gum)
        with TestClient(app) as client:
            resp = client.get("/gumbo")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        body = resp.text
        self.assertIn("GUMBO", body)
        self.assertIn("/suggestions?", body)
        # Project tabs and the mixed-initiative surfacing toggle are present.
        self.assertIn("surfaced_only", body)
        self.assertIn("Add a project", body)

    async def test_feedback_recorded_as_observation(self):
        # A thumbs vote is fed back into the GUM as an observation (paper §4.3)
        # so future suggestions reflect what the user found useful. Exercised
        # against the real batcher on the same thread the gum was built on.
        ok = await self.gum.add_suggestion_feedback(
            title="Rent a suit in Chicago",
            vote="down",
            description="Found three suit-rental shops near the venue.",
            focus="Wedding",
        )
        self.assertTrue(ok)
        item = self.gum.batcher._queue.get()
        self.assertEqual(item["observer_name"], "gumbo_feedback")
        self.assertIn("did not find helpful", item["content"])
        self.assertIn("Rent a suit in Chicago", item["content"])
        self.assertIn("Wedding", item["content"])

    async def test_feedback_bad_vote_records_nothing(self):
        ok = await self.gum.add_suggestion_feedback(title="x", vote="sideways")
        self.assertFalse(ok)
        self.assertEqual(self.gum.batcher.size(), 0)

    def test_feedback_endpoint_wires_to_gum(self):
        # The POST endpoint hands the vote to the GUM and reports the result.
        # (batcher.push is patched: its SQLite queue is bound to the gum's
        # creation thread, while TestClient runs endpoints on another thread.)
        app = create_app(self.gum)
        with mock.patch.object(self.gum.batcher, "push") as push:
            with TestClient(app) as client:
                good = client.post("/suggestions/feedback", json={
                    "title": "Rent a suit in Chicago", "vote": "up", "focus": "Wedding",
                })
                bad = client.post("/suggestions/feedback", json={"title": "x", "vote": "nope"})
        self.assertTrue(good.json()["ok"])
        self.assertFalse(bad.json()["ok"])
        push.assert_called_once()
        self.assertEqual(push.call_args.kwargs["observer_name"], "gumbo_feedback")
        self.assertIn("found helpful", push.call_args.kwargs["content"])

    def test_gumbo_page_has_feedback_controls(self):
        app = create_app(self.gum)
        with TestClient(app) as client:
            body = client.get("/gumbo").text
        self.assertIn("/suggestions/feedback", body)
        self.assertIn("Was this useful?", body)

    async def test_memory_lists_propositions_with_support(self):
        # The Memory page (paper Fig 3B) browses the raw propositions, each
        # annotated with its "support" — the number of observations backing it.
        async with self.gum._session() as s:
            prop = _prop("Omar is planning a Chicago trip", 9)
            prop.observations = {
                Observation(observer_name="screen", content="booked flight to ORD", content_type="text"),
                Observation(observer_name="screen", content="compared downtown hotels", content_type="text"),
            }
            s.add(prop)
        app = create_app(self.gum)
        with TestClient(app) as client:
            resp = client.get("/memory")
        self.assertEqual(resp.status_code, 200)
        props = resp.json()["propositions"]
        self.assertTrue(props)
        # Memory is NOT confidence-filtered — every proposition shows, each with a
        # support count and confidence for the table.
        for p in props:
            self.assertIn("support", p)
            self.assertIn("confidence", p)
        planned = next(p for p in props if p["text"].startswith("Omar is planning a Chicago trip"))
        self.assertEqual(planned["support"], 2)

    def test_memory_search_filters_and_keeps_support(self):
        app = create_app(self.gum)
        with TestClient(app) as client:
            resp = client.get("/memory", params={"q": "wedding"})
        self.assertEqual(resp.status_code, 200)
        props = resp.json()["propositions"]
        self.assertTrue(props)
        self.assertTrue(any("wedding" in p["text"].lower() for p in props))
        self.assertTrue(all("support" in p for p in props))

    def test_gumbo_page_has_memory_view(self):
        # The desktop app exposes a Memory section wired to /memory, with the
        # Support column from the paper.
        app = create_app(self.gum)
        with TestClient(app) as client:
            body = client.get("/gumbo").text
        self.assertIn("/memory?", body)
        self.assertIn('data-view="memory"', body)
        self.assertIn("Support", body)

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
