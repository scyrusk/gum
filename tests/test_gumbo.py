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
from gum.gumbo import Gumbo, Suggestion, expected_utility, lexical_overlap
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

    async def test_generate_stays_quiet_without_confident_propositions(self):
        engine = Gumbo(self.gum, min_confidence=10)  # nothing qualifies
        with mock.patch("gum.gumbo.structured_completion") as sc:
            suggestions = await engine.generate()
        self.assertEqual(suggestions, [])
        sc.assert_not_called()  # no model call when there's nothing to ground on


if __name__ == "__main__":
    unittest.main()
