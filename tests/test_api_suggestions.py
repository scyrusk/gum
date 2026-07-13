# test_api_suggestions.py
#
# Stdlib-only (unittest) tests for the GUMBO `/suggestions` REST endpoint.
# Runnable without a live model:  python -m unittest tests.test_api_suggestions
#
# The text model is stubbed (patched structured_completion), so these tests
# drive the read-only FastAPI app end-to-end through a TestClient while staying
# fully offline and deterministic.

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from unittest import mock

from fastapi.testclient import TestClient

from gum import gum as Gum
from gum.api import create_app
from gum.models import Observation, Proposition
from gum.schemas import (
    CommitmentItem,
    CommitmentSchema,
    SuggestionItem,
    SuggestionSchema,
)


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

    def test_rate_limited_caps_surfacing_across_requests(self):
        # Two suggestions clear the mixed-initiative bar, but the token-bucket
        # rate limit (paper §4.3.2, ~1/min) lets only one surface — and its state
        # is shared across requests, so a second immediate poll surfaces nothing.
        two_surfaced = SuggestionSchema(suggestions=[
            SuggestionItem(
                title="Rent a suit in Chicago",
                description="Found three suit-rental shops near the venue.",
                rationale="Wedding + no formal wear.",
                probability_useful=9, benefit=9, cost_if_wrong=2, cost_if_missed=7,
            ),
            SuggestionItem(
                title="Book a hotel near the venue",
                description="Reserve a room close to the ceremony.",
                rationale="Travel logistics.",
                probability_useful=8, benefit=8, cost_if_wrong=2, cost_if_missed=6,
            ),
        ])

        async def _fake(client, model, messages, schema, **kwargs):
            return two_surfaced

        app = create_app(self.gum)
        with mock.patch("gum.gumbo.structured_completion", side_effect=_fake):
            with TestClient(app) as client:
                first = client.get("/suggestions", params={"rate_limited": "true"})
                second = client.get("/suggestions", params={"rate_limited": "true"})
        first_sugs = first.json()["suggestions"]
        self.assertEqual(len(first_sugs), 1)  # only one of two worthy surfaces
        self.assertEqual(first_sugs[0]["title"], "Rent a suit in Chicago")
        # Bucket drained (well under a minute of real time elapsed) → nothing more.
        self.assertEqual(second.json()["suggestions"], [])

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

    async def test_memory_delete_removes_proposition(self):
        # Curating the model (paper Fig 3B): the user removes a proposition, and
        # it disappears from Memory. A backing observation must not block the
        # delete (the junction cascades).
        async with self.gum._session() as s:
            prop = _prop("Omar secretly dislikes cilantro", 8)
            prop.observations = {
                Observation(observer_name="screen", content="searched cilantro substitute", content_type="text"),
            }
            s.add(prop)
        async with self.gum._session() as s:
            row = next(p for p in (await self.gum.recent(limit=50))
                       if p.text.startswith("Omar secretly dislikes cilantro"))
            target_id = row.id

        app = create_app(self.gum)
        with TestClient(app) as client:
            resp = client.delete(f"/memory/{target_id}")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["ok"])
            # It's gone from the Memory listing.
            remaining = client.get("/memory").json()["propositions"]
        self.assertFalse(any(p["id"] == target_id for p in remaining))

    async def test_memory_delete_missing_returns_not_ok(self):
        app = create_app(self.gum)
        with TestClient(app) as client:
            resp = client.delete("/memory/999999")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["ok"])

    def test_gumbo_page_has_forget_control(self):
        # The Memory table exposes a per-row Forget action wired to DELETE /memory/.
        app = create_app(self.gum)
        with TestClient(app) as client:
            body = client.get("/gumbo").text
        self.assertIn("mem-forget", body)
        self.assertIn("/memory/", body)
        self.assertIn("DELETE", body)

    async def test_memory_edit_updates_proposition(self):
        # Curating the model (paper Fig 3B): the user corrects a close-but-wrong
        # proposition instead of deleting it. The edit persists and search reflects
        # the new text (the FTS AFTER UPDATE trigger keeps the index in sync).
        async with self.gum._session() as s:
            s.add(_prop("Omar prefers tea over coffee", 6))
        async with self.gum._session() as s:
            row = next(p for p in (await self.gum.recent(limit=50))
                       if p.text.startswith("Omar prefers tea"))
            target_id = row.id

        app = create_app(self.gum)
        with TestClient(app) as client:
            resp = client.patch(
                f"/memory/{target_id}",
                json={"text": "Omar prefers coffee over tea", "confidence": 9},
            )
            self.assertEqual(resp.status_code, 200)
            payload = resp.json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["proposition"]["text"], "Omar prefers coffee over tea")
            self.assertEqual(payload["proposition"]["confidence"], 9)
            # Reasoning was not passed, so it is left untouched.
            self.assertEqual(payload["proposition"]["reasoning"],
                             "because of Omar prefers tea over coffee")
            # The change is durable and the FTS index tracks the new wording.
            hits = client.get("/memory", params={"q": "coffee"}).json()["propositions"]
        self.assertTrue(any(p["id"] == target_id and p["text"] == "Omar prefers coffee over tea"
                            for p in hits))

    async def test_memory_edit_rejects_blank_and_missing(self):
        async with self.gum._session() as s:
            s.add(_prop("Omar has a cat", 5))
        async with self.gum._session() as s:
            target_id = next(p for p in (await self.gum.recent(limit=50))
                             if p.text.startswith("Omar has a cat")).id

        app = create_app(self.gum)
        with TestClient(app) as client:
            # Blank proposition text is meaningless and is rejected, unchanged.
            blank = client.patch(f"/memory/{target_id}", json={"text": "   "})
            self.assertFalse(blank.json()["ok"])
            # An unknown id reports not-ok rather than erroring.
            missing = client.patch("/memory/999999", json={"text": "anything"})
            self.assertFalse(missing.json()["ok"])
            # The original proposition is intact.
            still = client.get("/memory").json()["propositions"]
        self.assertTrue(any(p["id"] == target_id and p["text"] == "Omar has a cat"
                            for p in still))

    def test_gumbo_page_has_edit_control(self):
        # The Memory table exposes a per-row Edit action wired to PATCH /memory/.
        app = create_app(self.gum)
        with TestClient(app) as client:
            body = client.get("/gumbo").text
        self.assertIn("mem-edit", body)
        self.assertIn("PATCH", body)

    def test_chat_replies_grounded_in_propositions(self):
        # "Start Chat" (paper §4.3.3): the local text model answers grounded in
        # the user's high-confidence propositions and the suggestion in scope.
        captured = {}

        async def fake_chat(client, model, messages, **kwargs):
            captured["messages"] = messages
            return "Here are three suit-rental shops near the venue."

        app = create_app(self.gum)
        with mock.patch("gum.gumbo.text_completion", side_effect=fake_chat):
            with TestClient(app) as client:
                resp = client.post("/suggestions/chat", json={
                    "messages": [{"role": "user", "content": "Where can I rent one?"}],
                    "suggestion": {
                        "title": "Rent a suit in Chicago",
                        "description": "Found three suit-rental shops near the venue.",
                        "rationale": "Wedding + no formal wear.",
                    },
                })
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertIn("suit-rental", body["reply"])
        msgs = captured["messages"]
        # A system turn grounds GUMBO; the user's question is passed through.
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("wedding", msgs[0]["content"].lower())
        self.assertIn("Rent a suit in Chicago", msgs[0]["content"])
        self.assertEqual(msgs[-1], {"role": "user", "content": "Where can I rent one?"})

    def test_chat_requires_a_user_message(self):
        app = create_app(self.gum)
        with mock.patch("gum.gumbo.text_completion") as tc:
            with TestClient(app) as client:
                resp = client.post("/suggestions/chat", json={"messages": []})
        self.assertFalse(resp.json()["ok"])
        tc.assert_not_called()

    def test_chat_sanitizes_reply(self):
        fake_sanitizer = mock.Mock()
        fake_sanitizer.load = mock.Mock()
        fake_sanitizer.sanitize = mock.Mock(side_effect=lambda t: t.replace("Chicago", "[CITY]"))

        async def fake_chat(client, model, messages, **kwargs):
            return "Try the shops in downtown Chicago."

        with mock.patch("gum.sanitize.get_sanitizer", return_value=fake_sanitizer):
            app = create_app(self.gum, sanitize=True)
            with mock.patch("gum.gumbo.text_completion", side_effect=fake_chat):
                with TestClient(app) as client:
                    resp = client.post("/suggestions/chat", json={
                        "messages": [{"role": "user", "content": "where?"}],
                    })
        self.assertEqual(resp.json()["reply"], "Try the shops in downtown [CITY].")

    def test_gumbo_page_has_start_chat(self):
        app = create_app(self.gum)
        with TestClient(app) as client:
            body = client.get("/gumbo").text
        self.assertIn("/suggestions/chat", body)
        self.assertIn("Start Chat", body)

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


class AgendaEndpointTests(unittest.IsolatedAsyncioTestCase):
    """The Commitment & Deadline Radar exposed over HTTP as `GET /agenda`.

    Same ranked list the CLI `gum agenda` and MCP `agenda` surfaces build. The
    text model is stubbed (patched structured_completion) so these drive the
    read-only FastAPI app end-to-end while staying offline and deterministic. A
    far-future due date keeps `days_until_due` positive regardless of the real
    "now" the endpoint computes.
    """

    async def asyncSetUp(self):
        # Only the extraction call is stubbed; disable the second-pass
        # verification so it doesn't hit the same stub (covered by
        # tests.test_agenda.VerificationPassTests).
        self.enterContext(mock.patch.dict(os.environ, {"GUM_AGENDA_VERIFY": "0"}))
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db")
        await self.gum.connect_db()
        async with self.gum._session() as s:
            s.add_all([
                _prop("Omar has a grant proposal deadline for the Schmidt Foundation", 9),
                _prop("Omar promised to send reviewer comments back to a colleague", 7),
            ])

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def _fake_completion(self):
        async def fake(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(source_index=1,
                               title="Submit the Schmidt grant proposal",
                               due_date="2999-07-20", source="Schmidt",
                               status_guess="in progress"),
                CommitmentItem(source_index=2, title="Send reviewer comments",
                               due_date=None, source="a colleague",
                               status_guess="not started"),
            ])
        return fake

    def test_agenda_ranked_and_shaped(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake_completion()):
            with TestClient(app) as client:
                resp = client.get("/agenda")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 2)
        self.assertFalse(body["sanitized"])
        self.assertIsNone(body["window_days"])
        commitments = body["commitments"]
        # A dated commitment always outranks an undated one.
        self.assertEqual(commitments[0]["title"], "Submit the Schmidt grant proposal")
        self.assertEqual(commitments[0]["due_date"], "2999-07-20")
        self.assertIsNone(commitments[1]["due_date"])
        # Provenance + ranking fields are present for a client to render.
        for key in ("urgency", "days_until_due", "proposition_id", "status_guess"):
            self.assertIn(key, commitments[0])

    def test_window_excludes_far_future_commitments(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake_completion()):
            with TestClient(app) as client:
                resp = client.get("/agenda", params={"window_days": 30})
        body = resp.json()
        self.assertEqual(body["window_days"], 30)
        # The year-2999 deadline is far outside a 30-day horizon; the undated
        # commitment has no date and is always kept.
        titles = [c["title"] for c in body["commitments"]]
        self.assertEqual(titles, ["Send reviewer comments"])

    def test_limit_caps_results(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake_completion()):
            with TestClient(app) as client:
                resp = client.get("/agenda", params={"limit": 1})
        self.assertEqual(len(resp.json()["commitments"]), 1)

    def test_sanitize_scrubs_commitment_text(self):
        # Under --sanitize the model-written text fields are pseudonymized on the
        # way out while numeric/date/ranking fields pass through unchanged.
        fake_sanitizer = mock.Mock()
        fake_sanitizer.load = mock.Mock()
        fake_sanitizer.sanitize = mock.Mock(side_effect=lambda t: t.replace("Schmidt", "[ORG]"))
        # title/source use the carrier-context fragment path; this fake is
        # context-independent so it scrubs a fragment the same as a sentence.
        fake_sanitizer.sanitize_fragment = mock.Mock(side_effect=lambda t: t.replace("Schmidt", "[ORG]"))
        with mock.patch("gum.sanitize.get_sanitizer", return_value=fake_sanitizer):
            app = create_app(self.gum, sanitize=True)
            with mock.patch("gum.agenda.structured_completion", side_effect=self._fake_completion()):
                with TestClient(app) as client:
                    resp = client.get("/agenda")
        body = resp.json()
        self.assertTrue(body["sanitized"])
        top = body["commitments"][0]
        self.assertEqual(top["title"], "Submit the [ORG] grant proposal")
        self.assertNotIn("Schmidt", top["source"])
        # The date is untouched by the scrubber.
        self.assertEqual(top["due_date"], "2999-07-20")


class AgendaEditEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Editing/dismissing agenda items over REST (GUMBO Agenda page).

    The extraction model is stubbed; each edit persists as an override that the
    `GET /agenda` merge overlays, and propagates back into the GUM (proposition
    rewrite + correction observation). The batcher's `push` is patched so the
    correction observations don't kick off real inference during the test.
    """

    async def asyncSetUp(self):
        self.enterContext(mock.patch.dict(os.environ, {"GUM_AGENDA_VERIFY": "0"}))
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db")
        await self.gum.connect_db()
        async with self.gum._session() as s:
            s.add_all([
                # One proposition carries exactly one absolute date (rewritable),
                # the other is undated.
                _prop("Omar must submit the Schmidt grant proposal by 2026-07-20.", 9),
                _prop("Omar promised to send reviewer comments back to a colleague.", 7),
            ])

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def _fake(self):
        async def fake(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(source_index=1, title="Submit the Schmidt grant proposal",
                               due_date="2026-07-20", source="Schmidt", status_guess="in progress"),
                CommitmentItem(source_index=2, title="Send reviewer comments",
                               due_date=None, source="a colleague", status_guess="not started"),
            ])
        return fake

    def _get(self, client):
        return client.get("/agenda?limit=50").json()

    def test_today_anchor_and_editable(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push"):
                with TestClient(app) as client:
                    body = self._get(client)
        self.assertIn("today", body)
        self.assertRegex(body["today"]["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertIn("weekday", body["today"])
        self.assertTrue(all(c["editable"] for c in body["commitments"]))

    def test_edit_overrides_due_date_and_persists(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push"):
                with TestClient(app) as client:
                    before = self._get(client)
                    # Target the dated item by its proposition text (robust to the
                    # model's source-index ordering).
                    target = [c for c in before["commitments"]
                              if "2026-07-20" in c["proposition_text"]][0]
                    pid = target["proposition_id"]
                    r = client.patch(f"/agenda/{pid}", json={"due_date": "2026-08-15"}).json()
                    self.assertTrue(r["ok"])
                    after = self._get(client)
        edited = [c for c in after["commitments"] if c["proposition_id"] == pid][0]
        self.assertEqual(edited["due_date"], "2026-08-15")

    def test_edit_rewrites_single_date_proposition_text(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push"):
                with TestClient(app) as client:
                    before = self._get(client)
                    target = [c for c in before["commitments"]
                              if "2026-07-20" in c["proposition_text"]][0]
                    pid = target["proposition_id"]
                    client.patch(f"/agenda/{pid}", json={"due_date": "2026-08-15"})
                    # The date inside the source proposition was rewritten in place
                    # (so the deterministic MCP upcoming_deadlines scan sees the fix
                    # too), visible through the Memory endpoint.
                    mem = client.get("/memory").json()
        text = [p["text"] for p in mem["propositions"] if p["id"] == pid][0]
        self.assertIn("2026-08-15", text)
        self.assertNotIn("2026-07-20", text)

    def test_edit_normalizes_non_iso_due_date_to_iso(self):
        # `_parse_due` accepts a few lenient formats (e.g. mm/dd/yyyy) so a local
        # model's drift still lands on the radar. But a parseable-but-non-ISO value
        # must be canonicalized to YYYY-MM-DD before it is stored or spliced into
        # the proposition text — otherwise the `\d{4}-\d{2}-\d{2}` deadline scans
        # (and a later rewrite) would no longer see the date.
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push"):
                with TestClient(app) as client:
                    before = self._get(client)
                    target = [c for c in before["commitments"]
                              if "2026-07-20" in c["proposition_text"]][0]
                    pid = target["proposition_id"]
                    r = client.patch(f"/agenda/{pid}", json={"due_date": "07/20/2026"}).json()
                    self.assertTrue(r["ok"])
                    after = self._get(client)
                    mem = client.get("/memory").json()
        edited = [c for c in after["commitments"] if c["proposition_id"] == pid][0]
        self.assertEqual(edited["due_date"], "2026-07-20")
        text = [p["text"] for p in mem["propositions"] if p["id"] == pid][0]
        self.assertIn("2026-07-20", text)
        self.assertNotIn("07/20/2026", text)

    def test_edit_title_persists_over_model_output(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push"):
                with TestClient(app) as client:
                    before = self._get(client)
                    pid = before["commitments"][0]["proposition_id"]
                    client.patch(f"/agenda/{pid}", json={"title": "Finish the proposal"})
                    after = self._get(client)
        edited = [c for c in after["commitments"] if c["proposition_id"] == pid][0]
        self.assertEqual(edited["title"], "Finish the proposal")

    def test_edit_pushes_correction_observation(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push") as push:
                with TestClient(app) as client:
                    before = self._get(client)
                    pid = before["commitments"][0]["proposition_id"]
                    client.patch(f"/agenda/{pid}", json={"status": "blocked"})
        self.assertTrue(push.called)
        self.assertEqual(push.call_args.kwargs["observer_name"], "gumbo_agenda_edit")

    def test_bad_date_rejected(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push"):
                with TestClient(app) as client:
                    before = self._get(client)
                    pid = before["commitments"][0]["proposition_id"]
                    r = client.patch(f"/agenda/{pid}", json={"due_date": "next Friday"}).json()
        self.assertFalse(r["ok"])

    def test_edit_missing_proposition_returns_not_found(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push"):
                with TestClient(app) as client:
                    r = client.patch("/agenda/99999", json={"title": "x"}).json()
        self.assertFalse(r["ok"])

    def test_dismiss_hides_item_but_keeps_proposition(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push") as push:
                with TestClient(app) as client:
                    before = self._get(client)
                    pid = [c for c in before["commitments"]
                           if "reviewer comments" in c["proposition_text"]][0]["proposition_id"]
                    r = client.post(f"/agenda/{pid}/dismiss").json()
                    self.assertTrue(r["ok"])
                    after = self._get(client)
                    # Dismissed item is gone from the radar…
                    self.assertNotIn(pid, [c["proposition_id"] for c in after["commitments"]])
                    # …but the proposition is NOT deleted (dismiss != forget).
                    mem = client.get("/memory").json()
        self.assertIn(pid, [p["id"] for p in mem["propositions"]])
        self.assertEqual(push.call_args.kwargs["observer_name"], "gumbo_agenda_dismiss")

    def test_undo_restores_dismissed_item(self):
        app = create_app(self.gum)
        with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
            with mock.patch.object(self.gum.batcher, "push"):
                with TestClient(app) as client:
                    before = self._get(client)
                    pid = before["commitments"][0]["proposition_id"]
                    client.post(f"/agenda/{pid}/dismiss")
                    self.assertNotIn(pid, [c["proposition_id"] for c in self._get(client)["commitments"]])
                    client.post(f"/agenda/{pid}/undo")
                    restored = self._get(client)
        self.assertIn(pid, [c["proposition_id"] for c in restored["commitments"]])

    def test_override_persists_when_model_drops_item(self):
        app = create_app(self.gum)
        with mock.patch.object(self.gum.batcher, "push"):
            with mock.patch("gum.agenda.structured_completion", side_effect=self._fake()):
                with TestClient(app) as client:
                    before = self._get(client)
                    pid = before["commitments"][0]["proposition_id"]
                    client.patch(f"/agenda/{pid}", json={"title": "Sticky edit"})
                # Next load: the model now extracts nothing, but the edit must stick.
                async def empty(client_, model, messages, schema, **kwargs):
                    return CommitmentSchema(commitments=[])
                with mock.patch("gum.agenda.structured_completion", side_effect=empty):
                    with TestClient(app) as client:
                        after = self._get(client)
        titles = {c["proposition_id"]: c["title"] for c in after["commitments"]}
        self.assertEqual(titles.get(pid), "Sticky edit")


class AgendaAddedItemEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Explicitly-added agenda items (e.g. from the MCP add tool) show up in
    `GET /agenda` as first-class, editable commitments keyed by item_id, and edit
    directly (no proposition, no correction observation)."""

    async def asyncSetUp(self):
        self.enterContext(mock.patch.dict(os.environ, {"GUM_AGENDA_VERIFY": "0"}))
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db")
        await self.gum.connect_db()
        self.iid = await self.gum.add_agenda_item(
            title="Submit the Q3 report", due_date="2999-07-20",
            status="in progress", source="added by an assistant", created_by="mcp",
        )

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def _empty_model(self):
        async def fake(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[])
        return fake

    def _client(self):
        return TestClient(create_app(self.gum))

    def test_added_item_appears_editable_with_item_id(self):
        with mock.patch("gum.agenda.structured_completion", side_effect=self._empty_model()):
            with self._client() as client:
                body = client.get("/agenda").json()
        items = [c for c in body["commitments"] if c.get("item_id") == self.iid]
        self.assertEqual(len(items), 1)
        c = items[0]
        self.assertTrue(c["editable"])
        self.assertIsNone(c["proposition_id"])
        self.assertEqual(c["title"], "Submit the Q3 report")
        self.assertEqual(c["due_date"], "2999-07-20")

    def test_edit_item_endpoint(self):
        with mock.patch("gum.agenda.structured_completion", side_effect=self._empty_model()):
            with self._client() as client:
                r = client.patch(f"/agenda/item/{self.iid}",
                                 json={"status": "blocked", "due_date": "2999-08-01"}).json()
                self.assertTrue(r["ok"])
                c = [x for x in client.get("/agenda").json()["commitments"]
                     if x.get("item_id") == self.iid][0]
        self.assertEqual(c["status_guess"], "blocked")
        self.assertEqual(c["due_date"], "2999-08-01")

    def test_edit_item_bad_date_and_missing(self):
        with mock.patch("gum.agenda.structured_completion", side_effect=self._empty_model()):
            with self._client() as client:
                self.assertFalse(client.patch(f"/agenda/item/{self.iid}",
                                              json={"due_date": "soon"}).json()["ok"])
                self.assertFalse(client.patch("/agenda/item/999999",
                                              json={"title": "x"}).json()["ok"])

    def test_dismiss_and_undo_item(self):
        with mock.patch("gum.agenda.structured_completion", side_effect=self._empty_model()):
            with self._client() as client:
                self.assertTrue(client.post(f"/agenda/item/{self.iid}/dismiss").json()["ok"])
                gone = client.get("/agenda").json()["commitments"]
                self.assertNotIn(self.iid, [c.get("item_id") for c in gone])
                self.assertTrue(client.post(f"/agenda/item/{self.iid}/undo").json()["ok"])
                back = client.get("/agenda").json()["commitments"]
        self.assertIn(self.iid, [c.get("item_id") for c in back])


class AgendaPageTests(unittest.TestCase):
    def test_gumbo_page_has_agenda_view(self):
        # The static page must expose the Agenda nav + container the tray deep-links
        # to (/gumbo#agenda). Assert directly against the shipped file.
        from gum.api import _STATIC_DIR
        html = (_STATIC_DIR / "gumbo.html").read_text()
        self.assertIn('data-view="agenda"', html)
        self.assertIn('id="agenda"', html)
        self.assertIn("#agenda", html)  # deep-link handling


if __name__ == "__main__":
    unittest.main()
