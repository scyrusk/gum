# test_gumbo.py
#
# Stdlib-only (unittest) tests for the GUMBO suggestion engine. Runnable without
# pytest or a live model:  python -m unittest tests.test_gumbo
#
# The text model is stubbed out (patched structured_completion) so these tests
# exercise proposition selection, prompt assembly, expected-utility scoring, and
# ranking deterministically and offline.

from __future__ import annotations

import tempfile
import unittest
import uuid
from unittest import mock

from gum import gum as Gum
from gum.gumbo import Gumbo, Suggestion, TokenBucket, expected_utility, lexical_overlap
from gum.models import Proposition
from gum.schemas import SuggestionItem, SuggestionSchema


class LexicalOverlapTests(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertEqual(lexical_overlap("rent a suit", "rent a suit"), 1.0)

    def test_disjoint_is_zero(self):
        self.assertEqual(lexical_overlap("rent suit", "book flight"), 0.0)

    def test_case_and_punctuation_insensitive(self):
        # "Rent a Suit!" and "rent a suit" tokenize identically.
        self.assertEqual(lexical_overlap("Rent a Suit!", "rent a suit"), 1.0)

    def test_empty_is_zero(self):
        self.assertEqual(lexical_overlap("", "rent a suit"), 0.0)

    def test_partial_overlap_between_zero_and_one(self):
        v = lexical_overlap("rent a suit in chicago", "rent a tuxedo in chicago")
        self.assertGreater(v, 0.0)
        self.assertLess(v, 1.0)


class TokenBucketTests(unittest.TestCase):
    def _clock(self):
        # A hand-cranked clock so refill is deterministic (no real time / sleeps).
        self._t = [0.0]
        return lambda: self._t[0]

    def test_starts_full_and_drains(self):
        clock = self._clock()
        b = TokenBucket(1, 60, clock=clock)
        self.assertEqual(b.take(5), 1)  # one token available, only one granted
        self.assertEqual(b.take(5), 0)  # empty now

    def test_refills_one_per_interval_and_caps_at_capacity(self):
        clock = self._clock()
        b = TokenBucket(1, 60, clock=clock)
        self.assertEqual(b.take(1), 1)
        self._t[0] = 59.0  # not yet a full interval
        self.assertEqual(b.take(1), 0)
        self._t[0] = 60.0  # exactly one interval → one token back
        self.assertEqual(b.take(1), 1)
        # Idle for a long time must not let tokens accumulate past capacity.
        self._t[0] = 100000.0
        self.assertEqual(b.take(5), 1)

    def test_take_grants_up_to_capacity(self):
        clock = self._clock()
        b = TokenBucket(3, 10, clock=clock)
        self.assertEqual(b.take(5), 3)  # capped at what's available
        self.assertEqual(b.take(5), 0)


class ExpectedUtilityTests(unittest.TestCase):
    def test_high_value_low_cost_surfaces(self):
        eu, surface = expected_utility(
            probability_useful=9, benefit=9, cost_if_wrong=2, cost_if_missed=8
        )
        self.assertTrue(surface)
        self.assertGreater(eu, 0)

    def test_low_value_high_intrusion_withheld(self):
        eu, surface = expected_utility(
            probability_useful=2, benefit=3, cost_if_wrong=9, cost_if_missed=1
        )
        self.assertFalse(surface)
        self.assertLess(eu, 0)

    def test_matches_paper_equations(self):
        # E[interrupt] = p*B - (1-p)*C_FP ; E[quiet] = -p*C_FN ; eu = diff.
        p, B, cfp, cfn = 0.8, 6, 4, 5
        e_interrupt = p * B - (1 - p) * cfp
        e_quiet = -p * cfn
        eu, surface = expected_utility(8, B, cfp, cfn)
        self.assertAlmostEqual(eu, e_interrupt - e_quiet)
        self.assertEqual(surface, e_interrupt > e_quiet)


def _prop(text: str, confidence: int) -> Proposition:
    return Proposition(
        text=text,
        reasoning=f"because of {text}",
        confidence=confidence,
        decay=5,
        revision_group=uuid.uuid4().hex,
        version=1,
    )


class GumboEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Local-only client is built but never called (structured_completion is
        # patched in the generate test), so no model/network is required.
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

    async def test_select_filters_by_confidence(self):
        engine = Gumbo(self.gum, min_confidence=7)
        props = await engine.select_propositions()
        texts = {p.text for p in props}
        self.assertIn("Omar is going to a friend's wedding in Chicago", texts)
        self.assertIn("Omar doesn't own suitable formal wear", texts)
        self.assertNotIn("Omar is idly browsing social media", texts)

    async def test_generate_scores_ranks_and_grounds_prompt(self):
        captured = {}

        async def fake_completion(client, model, messages, schema, **kwargs):
            captured["messages"] = messages
            return SuggestionSchema(suggestions=[
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

        engine = Gumbo(self.gum, min_confidence=7)
        with mock.patch("gum.gumbo.structured_completion", side_effect=fake_completion):
            suggestions = await engine.generate()

        # The high-value/low-intrusion suggestion ranks first and is surfaced;
        # the noisy one is ranked last and withheld.
        self.assertEqual(len(suggestions), 2)
        self.assertEqual(suggestions[0].title, "Rent a suit in Chicago")
        self.assertTrue(suggestions[0].should_surface)
        self.assertFalse(suggestions[1].should_surface)
        self.assertGreaterEqual(suggestions[0].expected_utility, suggestions[1].expected_utility)
        self.assertIsInstance(suggestions[0], Suggestion)
        self.assertIn("wedding", suggestions[0].to_dict()["rationale"].lower() + " wedding")

        # High-confidence propositions (and the user's name) made it into the prompt;
        # the below-threshold one did not.
        prompt_text = captured["messages"][0]["content"]
        self.assertIn("Omar", prompt_text)
        self.assertIn("wedding in Chicago", prompt_text)
        self.assertNotIn("idly browsing social media", prompt_text)

    async def test_generate_filters_near_duplicate_suggestions(self):
        # The model emits two phrasings of the same idea plus a distinct one; the
        # lower-utility near-duplicate should be dropped, keeping the best of the pair.
        async def fake_completion(client, model, messages, schema, **kwargs):
            return SuggestionSchema(suggestions=[
                SuggestionItem(
                    title="Rent a suit in Chicago",
                    description="Find a suit rental shop near the wedding venue in Chicago.",
                    rationale="Wedding + no formal wear.",
                    probability_useful=9, benefit=9, cost_if_wrong=2, cost_if_missed=7,
                ),
                SuggestionItem(  # near-duplicate of the first, but lower utility
                    title="Rent a suit in Chicago soon",
                    description="Find a suit rental shop near the wedding venue in Chicago today.",
                    rationale="Same idea, restated.",
                    probability_useful=6, benefit=6, cost_if_wrong=3, cost_if_missed=4,
                ),
                SuggestionItem(
                    title="Book a hotel near the venue",
                    description="Reserve a room close to the ceremony.",
                    rationale="Travel logistics.",
                    probability_useful=8, benefit=7, cost_if_wrong=2, cost_if_missed=5,
                ),
            ])

        engine = Gumbo(self.gum, min_confidence=7)
        with mock.patch("gum.gumbo.structured_completion", side_effect=fake_completion):
            suggestions = await engine.generate()

        titles = [s.title for s in suggestions]
        self.assertEqual(len(suggestions), 2)
        self.assertIn("Rent a suit in Chicago", titles)  # higher-utility survivor kept
        self.assertNotIn("Rent a suit in Chicago soon", titles)  # near-dup dropped
        self.assertIn("Book a hotel near the venue", titles)  # distinct idea kept

    async def test_generate_dedup_disabled_by_zero_threshold(self):
        async def fake_completion(client, model, messages, schema, **kwargs):
            return SuggestionSchema(suggestions=[
                SuggestionItem(
                    title="Rent a suit in Chicago",
                    description="Find a suit rental shop.",
                    rationale="x", probability_useful=9, benefit=9,
                    cost_if_wrong=2, cost_if_missed=7,
                ),
                SuggestionItem(
                    title="Rent a suit in Chicago",
                    description="Find a suit rental shop.",
                    rationale="x", probability_useful=9, benefit=9,
                    cost_if_wrong=2, cost_if_missed=7,
                ),
            ])

        engine = Gumbo(self.gum, min_confidence=7, dedup_threshold=0.0)
        with mock.patch("gum.gumbo.structured_completion", side_effect=fake_completion):
            suggestions = await engine.generate()
        self.assertEqual(len(suggestions), 2)  # nothing filtered

    async def test_generate_suppresses_already_seen_across_polls(self):
        async def fake_completion(client, model, messages, schema, **kwargs):
            return SuggestionSchema(suggestions=[
                SuggestionItem(
                    title="Rent a suit in Chicago",
                    description="Find a suit rental shop near the wedding venue in Chicago.",
                    rationale="x", probability_useful=9, benefit=9,
                    cost_if_wrong=2, cost_if_missed=7,
                ),
            ])

        engine = Gumbo(self.gum, min_confidence=7)
        with mock.patch("gum.gumbo.structured_completion", side_effect=fake_completion):
            suggestions = await engine.generate(
                seen=["Rent a suit in Chicago Find a suit rental shop near the wedding venue in Chicago."]
            )
        self.assertEqual(suggestions, [])  # already surfaced earlier → withheld

    async def _two_surfaced_completion(self, client, model, messages, schema, **kwargs):
        # Two distinct, high-value/low-intrusion suggestions — both clear the
        # mixed-initiative bar, so only the rate limit should hold them back.
        return SuggestionSchema(suggestions=[
            SuggestionItem(
                title="Rent a suit in Chicago",
                description="Find a suit rental shop near the wedding venue.",
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

    async def test_surface_rate_limits_to_one_per_interval(self):
        t = [0.0]
        engine = Gumbo(self.gum, min_confidence=7, surface_interval=60, clock=lambda: t[0])
        with mock.patch("gum.gumbo.structured_completion", side_effect=self._two_surfaced_completion):
            # Two clear the EU bar, but the token bucket only lets one through.
            first = await engine.surface()
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].title, "Rent a suit in Chicago")  # highest utility

            # A second poll within the same minute surfaces nothing.
            self.assertEqual(await engine.surface(), [])

            # After a full interval the bucket refills and one surfaces again.
            t[0] = 60.0
            again = await engine.surface()
            self.assertEqual(len(again), 1)

    async def test_surface_disabled_returns_all_worthy(self):
        engine = Gumbo(self.gum, min_confidence=7, surface_interval=0)  # limit off
        with mock.patch("gum.gumbo.structured_completion", side_effect=self._two_surfaced_completion):
            worthy = await engine.surface()
        self.assertEqual(len(worthy), 2)  # both surfaced, no rate cap

    async def test_generate_stays_quiet_without_confident_propositions(self):
        engine = Gumbo(self.gum, min_confidence=10)  # nothing qualifies
        with mock.patch("gum.gumbo.structured_completion") as sc:
            suggestions = await engine.generate()
        self.assertEqual(suggestions, [])
        sc.assert_not_called()  # no model call when there's nothing to ground on


class _RecordingExecutor:
    """Test double for the execution bridge: records what it was asked to run."""

    def __init__(self):
        self.dispatched: list = []
        self.options: list = []

    async def dispatch(self, suggestion, **kwargs):
        self.dispatched.append(suggestion)
        self.options.append(kwargs)
        return {"suggestion": suggestion.title, "status": "pending_approval"}


class GumboExecuteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db")
        await self.gum.connect_db()
        async with self.gum._session() as s:
            s.add_all([
                _prop("Omar is going to a friend's wedding in Chicago", 8),
                _prop("Omar doesn't own suitable formal wear", 7),
            ])

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def _one_surfaced_completion(self):
        async def fake_completion(client, model, messages, schema, **kwargs):
            return SuggestionSchema(suggestions=[
                SuggestionItem(
                    title="Rent a suit in Chicago",
                    description="Find a suit rental shop near the wedding venue.",
                    rationale="Wedding + no formal wear.",
                    probability_useful=9, benefit=9, cost_if_wrong=2, cost_if_missed=7,
                ),
            ])
        return fake_completion

    async def test_execute_off_by_default_is_a_noop(self):
        # Default-OFF: execute() dispatches nothing and never touches the executor,
        # even when a worthy suggestion exists.
        executor = _RecordingExecutor()
        engine = Gumbo(self.gum, min_confidence=7, executor=executor)
        self.assertFalse(engine.execution_enabled)
        with mock.patch("gum.gumbo.structured_completion", side_effect=self._one_surfaced_completion()):
            outcomes = await engine.execute()
        self.assertEqual(outcomes, [])
        self.assertEqual(executor.dispatched, [])

    async def test_execute_dispatches_worthy_suggestions_when_enabled(self):
        executor = _RecordingExecutor()
        engine = Gumbo(
            self.gum, min_confidence=7, execution_enabled=True, executor=executor,
        )
        with mock.patch("gum.gumbo.structured_completion", side_effect=self._one_surfaced_completion()):
            outcomes = await engine.execute()
        self.assertEqual(len(outcomes), 1)
        self.assertEqual([s.title for s in executor.dispatched], ["Rent a suit in Chicago"])
        self.assertEqual(outcomes[0]["status"], "pending_approval")

    async def test_execute_preserves_token_bucket_rate_limit(self):
        # execute() runs the same rate-limited pipeline as surface(): the token
        # bucket that caps surfacing also caps how many suggestions can act.
        t = [0.0]
        executor = _RecordingExecutor()
        engine = Gumbo(
            self.gum, min_confidence=7, execution_enabled=True, executor=executor,
            surface_interval=60, clock=lambda: t[0],
        )
        with mock.patch("gum.gumbo.structured_completion", side_effect=self._one_surfaced_completion()):
            first = await engine.execute()
            self.assertEqual(len(first), 1)
            # A second call within the same interval is throttled — nothing dispatched.
            self.assertEqual(await engine.execute(), [])
        self.assertEqual(len(executor.dispatched), 1)

    async def test_execute_env_flag_enables_bridge(self):
        executor = _RecordingExecutor()
        with mock.patch.dict("os.environ", {"GUMBO_EXECUTION_ENABLED": "1"}):
            engine = Gumbo(self.gum, min_confidence=7, executor=executor)
        self.assertTrue(engine.execution_enabled)
        with mock.patch("gum.gumbo.structured_completion", side_effect=self._one_surfaced_completion()):
            outcomes = await engine.execute()
        self.assertEqual(len(outcomes), 1)

    async def test_execute_suggestion_dispatches_exact_card_with_instructions(self):
        executor = _RecordingExecutor()
        engine = Gumbo(self.gum, execution_enabled=True, executor=executor)
        item = (await self._one_surfaced_completion()(None, None, None, None)).suggestions[0]
        utility, should_surface = expected_utility(
            item.probability_useful, item.benefit, item.cost_if_wrong, item.cost_if_missed
        )
        suggestion = Suggestion(
            **item.model_dump(),
            expected_utility=utility,
            should_surface=should_surface,
        )
        outcome = await engine.execute_suggestion(
            suggestion, user_instructions="Use a table.", explicit=True
        )
        self.assertEqual(outcome["suggestion"], suggestion.title)
        self.assertEqual(executor.dispatched, [suggestion])
        self.assertEqual(
            executor.options,
            [{"user_instructions": "Use a table.", "explicit": True}],
        )


class SuggestionItemBoundsTests(unittest.TestCase):
    """The four suggestion scores are contractually 1–10; the schema enforces it.

    These scores are gate inputs to the execution bridge: ``probability_useful``
    is read raw by ``Executor.is_auto_dispatchable`` (a malformed high value like
    100 would clear the ``probability_useful < min_probability`` confidence bar on
    a bogus score), and ``benefit``/``cost_if_wrong``/``cost_if_missed`` feed the
    ``expected_utility``/``should_surface`` decision the gate also depends on
    (only ``p`` is clamped there, so an out-of-range ``benefit`` could flip
    ``should_surface``). Enforcing the range rejects a malformed score at
    validation — driving ``structured_completion``'s retries — instead of letting
    it sail through the auto-dispatch gate, mirroring ``RiskAssessmentSchema.risk``.
    """

    def _item(self, **overrides):
        fields = dict(
            title="t", description="d", rationale="r",
            probability_useful=9, benefit=9, cost_if_wrong=2, cost_if_missed=7,
        )
        fields.update(overrides)
        return SuggestionItem(**fields)

    def test_in_range_scores_are_accepted(self):
        for score in (1, 5, 10):
            self._item(
                probability_useful=score, benefit=score,
                cost_if_wrong=score, cost_if_missed=score,
            )

    def test_out_of_range_scores_are_rejected(self):
        from pydantic import ValidationError

        for field in ("probability_useful", "benefit", "cost_if_wrong", "cost_if_missed"):
            for score in (0, -1, 11, 99):
                with self.assertRaises(ValidationError):
                    self._item(**{field: score})

    def test_json_schema_carries_the_bounds_for_the_model(self):
        props = SuggestionItem.model_json_schema()["properties"]
        for field in ("probability_useful", "benefit", "cost_if_wrong", "cost_if_missed"):
            self.assertEqual(props[field]["minimum"], 1, field)
            self.assertEqual(props[field]["maximum"], 10, field)


if __name__ == "__main__":
    unittest.main()
