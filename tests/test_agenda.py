# test_agenda.py
#
# Stdlib-only (unittest) tests for the Commitment & Deadline Radar (gum agenda).
# Runnable without pytest or a live model:  python -m unittest tests.test_agenda
#
# The text model is stubbed out (patched structured_completion) so these tests
# exercise candidate selection, prompt assembly, due-date parsing, urgency
# ranking, the horizon window, and provenance mapping deterministically/offline.

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from unittest import mock

from gum import gum as Gum
from gum.agenda import (
    Commitment,
    CommitmentRadar,
    _dedupe_key,
    _parse_due,
    build_agenda,
    urgency_score,
)
from gum.schemas import (
    CommitmentItem,
    CommitmentSchema,
    CommitmentVerdictSchema,
)

# A fixed "now" so proximity/recency math is deterministic across the suite.
NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)


class ParseDueTests(unittest.TestCase):
    def test_iso(self):
        self.assertEqual(_parse_due("2026-07-20").isoformat(), "2026-07-20")

    def test_alternate_formats(self):
        self.assertEqual(_parse_due("2026/07/20").isoformat(), "2026-07-20")
        self.assertEqual(_parse_due("07/20/2026").isoformat(), "2026-07-20")

    def test_none_and_blank(self):
        self.assertIsNone(_parse_due(None))
        self.assertIsNone(_parse_due("   "))

    def test_unparseable_is_undated(self):
        self.assertIsNone(_parse_due("next Friday"))


class UrgencyScoreTests(unittest.TestCase):
    def test_overdue_scores_maximum(self):
        # Due yesterday, high confidence → top of the dated band (~2.0).
        v = urgency_score(days_until_due=-1, confidence=10, decay=5, age_days=0.0)
        self.assertAlmostEqual(v, 2.0, places=6)

    def test_dated_always_outranks_undated(self):
        far = urgency_score(days_until_due=365, confidence=1, decay=1, age_days=0.0)
        best_undated = urgency_score(
            days_until_due=None, confidence=10, decay=10, age_days=0.0
        )
        self.assertGreater(far, 1.0)          # dated band floor
        self.assertLessEqual(best_undated, 1.0)  # undated band ceiling
        self.assertGreater(far, best_undated)

    def test_closer_deadline_ranks_higher(self):
        soon = urgency_score(2, confidence=8, decay=5, age_days=0.0)
        later = urgency_score(30, confidence=8, decay=5, age_days=0.0)
        self.assertGreater(soon, later)

    def test_confidence_breaks_ties_between_equal_deadlines(self):
        sure = urgency_score(5, confidence=9, decay=5, age_days=0.0)
        shaky = urgency_score(5, confidence=2, decay=5, age_days=0.0)
        self.assertGreater(sure, shaky)

    def test_undated_recent_beats_stale_and_decay_sets_fade(self):
        recent = urgency_score(None, confidence=8, decay=5, age_days=0.0)
        stale = urgency_score(None, confidence=8, decay=5, age_days=60.0)
        self.assertGreater(recent, stale)
        # Higher decay keeps an equally-old item warmer.
        durable = urgency_score(None, confidence=8, decay=10, age_days=10.0)
        fleeting = urgency_score(None, confidence=8, decay=1, age_days=10.0)
        self.assertGreater(durable, fleeting)


class DedupeKeyTests(unittest.TestCase):
    def test_folds_case_and_punctuation(self):
        self.assertEqual(
            _dedupe_key("Submit the NSF grant!"),
            _dedupe_key("submit  the  nsf  grant"),
        )

    def test_distinct_titles_differ(self):
        self.assertNotEqual(
            _dedupe_key("Submit the NSF grant"), _dedupe_key("Pay the electric bill")
        )

    def test_empty_when_no_alphanumeric_content(self):
        self.assertEqual(_dedupe_key("  —  !! "), "")


class PromptGuidanceTests(unittest.TestCase):
    """The extraction prompt is the sole precision lever for the local model, so
    guard that its hard-won exclusion guidance stays in place. A negated/hedged
    proposition ("is likely not involved in X") must not be inverted into a
    positive commitment, while a deliverable-negation ("has not yet submitted X")
    must still count — the distinction the model has to make."""

    def test_prompt_excludes_negated_but_keeps_not_yet_done(self):
        from gum.prompts.gum import AGENDA_PROMPT

        lowered = AGENDA_PROMPT.lower()
        self.assertIn("negated", lowered)
        self.assertIn("unlikely to", lowered)
        self.assertIn("no longer", lowered)
        # The counter-example that keeps genuine open commitments in scope.
        self.assertIn("has not yet submitted", lowered)


def _prop(text: str, confidence: int, *, decay: int = 5, created_at=None) -> object:
    from gum.models import Proposition

    kwargs = dict(
        text=text,
        reasoning=f"because of {text}",
        confidence=confidence,
        decay=decay,
        revision_group=uuid.uuid4().hex,
        version=1,
    )
    if created_at is not None:
        kwargs["created_at"] = created_at
    return Proposition(**kwargs)


class CommitmentRadarTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # These tests stub only the extraction call, so switch off the second-pass
        # verification (which would otherwise hit the same stub). The verification
        # pass has its own dedicated coverage in VerificationPassTests.
        self.enterContext(mock.patch.dict(os.environ, {"GUM_AGENDA_VERIFY": "0"}))
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum(
            "Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db"
        )
        await self.gum.connect_db()
        async with self.gum._session() as s:
            s.add_all([
                _prop(
                    "Omar has a grant proposal deadline for the NSF on July 20",
                    9,
                    created_at=datetime(2026, 7, 8, tzinfo=timezone.utc),
                ),
                _prop(
                    "Omar promised to send reviewer comments back to a colleague",
                    7,
                    created_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
                ),
                _prop("Omar prefers dark mode in his editor", 8),  # not a commitment
                _prop("Omar is idly browsing social media", 2),    # below the bar
            ])

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    async def test_select_candidates_filters_by_confidence_and_dedupes(self):
        radar = CommitmentRadar(self.gum, min_confidence=3)
        props = await radar.select_candidates()
        texts = {p.text for p in props}
        self.assertIn("Omar has a grant proposal deadline for the NSF on July 20", texts)
        self.assertNotIn("Omar is idly browsing social media", texts)  # conf 2 < 3
        # BM25 hit + recency backfill must not double-count a proposition.
        ids = [p.id for p in props]
        self.assertEqual(len(ids), len(set(ids)))

    async def test_build_extracts_ranks_and_grounds_prompt(self):
        captured = {}

        def _index_of(prompt: str, needle: str) -> int:
            # Candidate order is BM25-then-recency, not insertion order, so read
            # the 1-based number the extractor actually saw for this proposition.
            for line in prompt.splitlines():
                if needle in line:
                    return int(line.split(".", 1)[0].strip())
            raise AssertionError(f"{needle!r} not found in prompt")

        async def fake_completion(client, model, messages, schema, **kwargs):
            prompt = messages[0]["content"]
            captured["messages"] = messages
            return CommitmentSchema(commitments=[
                CommitmentItem(
                    source_index=_index_of(prompt, "NSF"),
                    title="Submit the NSF grant proposal",
                    due_date="2026-07-20",
                    source="NSF",
                    status_guess="in progress",
                ),
                CommitmentItem(
                    source_index=_index_of(prompt, "reviewer comments"),
                    title="Send reviewer comments to a colleague",
                    due_date=None,
                    source="a colleague",
                    status_guess="not started",
                ),
            ])

        radar = CommitmentRadar(self.gum, min_confidence=3)
        with mock.patch("gum.agenda.structured_completion", side_effect=fake_completion):
            commitments = await radar.build(now=NOW)

        self.assertEqual(len(commitments), 2)
        # Dated commitment ranks above the undated one.
        self.assertEqual(commitments[0].title, "Submit the NSF grant proposal")
        self.assertEqual(commitments[0].due_date, "2026-07-20")
        self.assertEqual(commitments[0].days_until_due, 9)
        self.assertGreater(commitments[0].urgency, commitments[1].urgency)
        self.assertIsInstance(commitments[0], Commitment)

        # Provenance is reused from the source proposition, not re-derived.
        self.assertEqual(commitments[0].confidence, 9)
        self.assertEqual(commitments[0].decay, 5)
        self.assertIsNotNone(commitments[0].proposition_id)
        self.assertIn("NSF", commitments[0].proposition_text)

        # The undated one is ranked by confidence*recency, not proximity.
        self.assertIsNone(commitments[1].due_date)
        self.assertIsNone(commitments[1].days_until_due)

        # The prompt is grounded: user name, today's date, and the candidate text.
        prompt = captured["messages"][0]["content"]
        self.assertIn("Omar", prompt)
        self.assertIn("2026-07-11", prompt)          # today
        self.assertIn("grant proposal deadline", prompt)

    async def test_build_drops_out_of_range_source_index(self):
        async def fake_completion(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(
                    source_index=999,  # no such proposition was offered
                    title="Phantom task",
                    due_date=None,
                    source="unknown",
                    status_guess="unknown",
                ),
            ])

        radar = CommitmentRadar(self.gum, min_confidence=3)
        with mock.patch("gum.agenda.structured_completion", side_effect=fake_completion):
            commitments = await radar.build(now=NOW)
        self.assertEqual(commitments, [])

    async def test_window_excludes_far_future_keeps_overdue_and_undated(self):
        async def fake_completion(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(source_index=1, title="Far future", due_date="2026-12-31",
                               source="x", status_guess="not started"),
                CommitmentItem(source_index=1, title="Overdue", due_date="2026-07-01",
                               source="x", status_guess="not started"),
                CommitmentItem(source_index=2, title="Undated", due_date=None,
                               source="x", status_guess="not started"),
            ])

        radar = CommitmentRadar(self.gum, min_confidence=3, window_days=7)
        with mock.patch("gum.agenda.structured_completion", side_effect=fake_completion):
            commitments = await radar.build(now=NOW)
        titles = {c.title for c in commitments}
        self.assertNotIn("Far future", titles)  # beyond the 7-day horizon
        self.assertIn("Overdue", titles)         # overdue always kept
        self.assertIn("Undated", titles)         # no date to compare → kept

    async def test_limit_caps_results(self):
        async def fake_completion(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(source_index=1, title="A", due_date="2026-07-12",
                               source="x", status_guess="unknown"),
                CommitmentItem(source_index=2, title="B", due_date="2026-07-13",
                               source="x", status_guess="unknown"),
            ])

        radar = CommitmentRadar(self.gum, min_confidence=3, limit=1)
        with mock.patch("gum.agenda.structured_completion", side_effect=fake_completion):
            commitments = await radar.build(now=NOW)
        self.assertEqual(len(commitments), 1)

    async def test_build_agenda_helper_and_to_dict(self):
        async def fake_completion(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(source_index=1, title="Submit the NSF grant proposal",
                               due_date="2026-07-20", source="NSF", status_guess="in progress"),
            ])

        with mock.patch("gum.agenda.structured_completion", side_effect=fake_completion):
            commitments = await build_agenda(self.gum, now=NOW)
        self.assertEqual(len(commitments), 1)
        d = commitments[0].to_dict()
        for key in ("title", "due_date", "source", "status_guess", "urgency",
                    "days_until_due", "proposition_id", "confidence", "decay"):
            self.assertIn(key, d)

    async def test_build_uses_greedy_decoding(self):
        # Commitment extraction is a classification task: the same GUM state must
        # yield the same radar, and the sampling noise of a nonzero temperature is
        # what lets an ongoing-activity proposition slip through as a false
        # positive on some runs. Guard that the extraction is pinned to greedy
        # decoding (temperature=0), matching the other decision calls in gum.gum.
        captured = {}

        async def fake_completion(client, model, messages, schema, **kwargs):
            captured["kwargs"] = kwargs
            return CommitmentSchema(commitments=[])

        radar = CommitmentRadar(self.gum, min_confidence=3)
        with mock.patch("gum.agenda.structured_completion", side_effect=fake_completion):
            await radar.build(now=NOW)

        self.assertEqual(captured["kwargs"].get("temperature"), 0)

    async def test_build_dedupes_near_duplicate_titles_keeping_most_urgent(self):
        # The GUM re-infers overlapping propositions, so the extractor emits the
        # same commitment twice with cosmetically-different titles and dates.
        async def fake_completion(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(source_index=1, title="Submit the NSF grant proposal!",
                               due_date="2026-07-25", source="NSF",
                               status_guess="in progress"),
                CommitmentItem(source_index=1, title="submit the  NSF grant  proposal",
                               due_date="2026-07-13", source="NSF",
                               status_guess="in progress"),
                CommitmentItem(source_index=2, title="Pay the electric bill",
                               due_date="2026-07-20", source="utility",
                               status_guess="not started"),
            ])

        radar = CommitmentRadar(self.gum, min_confidence=3)
        with mock.patch("gum.agenda.structured_completion", side_effect=fake_completion):
            commitments = await radar.build(now=NOW)

        titles = [c.title for c in commitments]
        # The two NSF variants collapse to one; the distinct bill survives.
        self.assertEqual(len(commitments), 2)
        self.assertEqual(sum("nsf" in t.lower() for t in titles), 1)
        self.assertIn("Pay the electric bill", titles)
        # The surviving NSF instance is the more urgent (sooner-dated) one.
        nsf = next(c for c in commitments if "nsf" in c.title.lower())
        self.assertEqual(nsf.due_date, "2026-07-13")

    async def test_empty_when_no_candidates(self):
        radar = CommitmentRadar(self.gum, min_confidence=11)  # nothing qualifies
        with mock.patch("gum.agenda.structured_completion") as sc:
            commitments = await radar.build(now=NOW)
        self.assertEqual(commitments, [])
        sc.assert_not_called()  # no model call when there's nothing to extract from


class VerifyPromptGuidanceTests(unittest.TestCase):
    """The verification prompt is the precision lever of the second pass; guard
    that its isolation/ongoing-activity guidance stays in place."""

    def test_verify_prompt_has_isolation_and_ongoing_guidance(self):
        from gum.prompts.gum import AGENDA_VERIFY_PROMPT

        lowered = AGENDA_VERIFY_PROMPT.lower()
        self.assertIn("in isolation", lowered)
        self.assertIn("discrete completion point", lowered)
        self.assertIn("ongoing", lowered)
        # A specific scheduled meeting must survive verification, whatever its topic.
        self.assertIn("scheduled meeting", lowered)
        self.assertIn("is_commitment", AGENDA_VERIFY_PROMPT)


class VerificationPassTests(unittest.IsolatedAsyncioTestCase):
    """The second-pass verification that re-judges each extracted commitment in
    isolation, dropping ongoing-activity false positives the pooled extraction
    over-promotes. Both LLM calls are stubbed via a schema-dispatching fake, so
    the pass is exercised deterministically and offline (verification ON here)."""

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum(
            "Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db"
        )
        await self.gum.connect_db()
        async with self.gum._session() as s:
            s.add_all([
                _prop("Omar promised to send reviewer comments back to a colleague", 8),
                _prop("Omar manages research-related activities across several apps", 9),
            ])

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def _index_of(self, prompt: str, needle: str) -> int:
        for line in prompt.splitlines():
            if needle in line and line.strip()[:1].isdigit():
                return int(line.split(".", 1)[0].strip())
        raise AssertionError(f"{needle!r} not found in prompt")

    def _dispatch_stub(self, verdicts=None, verify_error=False, calls=None):
        """A structured_completion fake that answers by schema.

        The extraction call (CommitmentSchema) returns one genuine + one ongoing
        commitment; the verification call (CommitmentVerdictSchema) answers each
        proposition using *verdicts* (drop the 'manages' habit by default).
        """
        async def fake(client, model, messages, schema, **kwargs):
            prompt = messages[0]["content"]
            if schema is CommitmentSchema:
                return CommitmentSchema(commitments=[
                    CommitmentItem(
                        source_index=self._index_of(prompt, "reviewer comments"),
                        title="Send reviewer comments to a colleague",
                        due_date=None, source="a colleague",
                        status_guess="not started"),
                    CommitmentItem(
                        source_index=self._index_of(prompt, "manages research"),
                        title="Manage research-related activities",
                        due_date=None, source="unknown", status_guess="unknown"),
                ])
            # verification call
            if calls is not None:
                calls.append(prompt)
            if verify_error:
                raise RuntimeError("model unavailable")
            keep = not ("manages research" in prompt)
            if verdicts is not None:
                keep = verdicts(prompt)
            return CommitmentVerdictSchema(
                is_commitment=keep, reason="stub verdict")
        return fake

    async def test_verification_drops_ongoing_activity(self):
        calls: list[str] = []
        radar = CommitmentRadar(self.gum, min_confidence=3)  # verify on by default
        with mock.patch("gum.agenda.structured_completion",
                        side_effect=self._dispatch_stub(calls=calls)):
            commitments = await radar.build(now=NOW)

        titles = [c.title for c in commitments]
        self.assertEqual(titles, ["Send reviewer comments to a colleague"])
        # One verification call per extracted commitment (2), judged individually.
        self.assertEqual(len(calls), 2)

    async def test_verification_disabled_keeps_everything(self):
        radar = CommitmentRadar(self.gum, min_confidence=3, verify=False)
        with mock.patch("gum.agenda.structured_completion",
                        side_effect=self._dispatch_stub()):
            commitments = await radar.build(now=NOW)
        self.assertEqual(len(commitments), 2)  # ongoing item is NOT dropped

    async def test_verification_fails_open_on_error(self):
        # A verification error must never silently empty the radar — keep the item.
        radar = CommitmentRadar(self.gum, min_confidence=3)
        with mock.patch("gum.agenda.structured_completion",
                        side_effect=self._dispatch_stub(verify_error=True)):
            commitments = await radar.build(now=NOW)
        self.assertEqual(len(commitments), 2)

    async def test_verify_env_toggle_off(self):
        with mock.patch.dict(os.environ, {"GUM_AGENDA_VERIFY": "0"}):
            radar = CommitmentRadar(self.gum, min_confidence=3)
        self.assertFalse(radar.verify)


if __name__ == "__main__":
    unittest.main()
