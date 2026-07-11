# test_executor.py
#
# Stdlib-only (unittest) tests for the GUMBO execution bridge's safety gate
# (spec #4). Runnable without pytest or a live model:
#     python -m unittest tests.test_executor
#
# The text model is stubbed out (patched structured_completion) so these tests
# exercise the risk assessment and the auto-dispatch decision deterministically
# and offline. The gate is the safety core of the bridge: a high-confidence,
# read-only/reversible, low-risk suggestion may auto-dispatch; everything else
# stays proposal-only.

from __future__ import annotations

import tempfile
import unittest
import uuid
from unittest import mock

from gum import gum as Gum
from gum.executor import (
    DEFAULT_MAX_RISK,
    DEFAULT_MIN_PROBABILITY,
    Executor,
    RiskAssessment,
)
from gum.gumbo import Suggestion, expected_utility
from gum.schemas import RiskAssessmentSchema


def _suggestion(
    *,
    title: str = "Research suit-rental shops near the venue",
    description: str = "Find three formalwear rental options in Chicago for review.",
    probability_useful: int = 9,
    benefit: int = 8,
    cost_if_wrong: int = 2,
    cost_if_missed: int = 7,
) -> Suggestion:
    eu, surface = expected_utility(
        probability_useful, benefit, cost_if_wrong, cost_if_missed
    )
    return Suggestion(
        title=title,
        description=description,
        rationale="wedding + no formal wear",
        probability_useful=probability_useful,
        benefit=benefit,
        cost_if_wrong=cost_if_wrong,
        cost_if_missed=cost_if_missed,
        expected_utility=eu,
        should_surface=surface,
    )


class RiskAssessmentDataclassTests(unittest.TestCase):
    def test_read_only_and_reversible_are_reversible(self):
        self.assertTrue(RiskAssessment("read_only", 1, "r").is_reversible)
        self.assertTrue(RiskAssessment("reversible", 2, "r").is_reversible)

    def test_irreversible_is_not_reversible(self):
        self.assertFalse(RiskAssessment("irreversible", 2, "r").is_reversible)


class ExecutorGateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # The local-only client is built but never called (structured_completion
        # is patched), so no model/network is required.
        self.gum = Gum(
            "Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db"
        )

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def _patch_assessment(self, reversibility: str, risk: int):
        async def fake_completion(client, model, messages, schema, **kwargs):
            self.assertIs(schema, RiskAssessmentSchema)
            return RiskAssessmentSchema(
                reversibility=reversibility, risk=risk, rationale="stub"
            )

        return mock.patch(
            "gum.executor.structured_completion", side_effect=fake_completion
        )

    async def test_assess_risk_grounds_prompt_and_returns_assessment(self):
        captured = {}

        async def fake_completion(client, model, messages, schema, **kwargs):
            captured["messages"] = messages
            return RiskAssessmentSchema(
                reversibility="read_only", risk=2, rationale="only researches options"
            )

        ex = Executor(self.gum)
        sug = _suggestion()
        with mock.patch(
            "gum.executor.structured_completion", side_effect=fake_completion
        ):
            assessment = await ex.assess_risk(sug)

        self.assertEqual(assessment.reversibility, "read_only")
        self.assertEqual(assessment.risk, 2)
        self.assertTrue(assessment.is_reversible)
        # The suggestion (and the user's name) grounded the risk prompt.
        prompt = captured["messages"][0]["content"]
        self.assertIn("Omar", prompt)
        self.assertIn(sug.title, prompt)
        self.assertIn(sug.description, prompt)

    async def test_high_confidence_reversible_low_risk_dispatches(self):
        ex = Executor(self.gum)
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("reversible", 2):
            assessment = await ex.assess_risk(sug)
        self.assertTrue(ex.is_auto_dispatchable(sug, assessment))

    async def test_irreversible_action_stays_proposal_only(self):
        ex = Executor(self.gum)
        # Even a maximally confident suggestion must not auto-run an irreversible
        # action (e.g. sending a message on the user's behalf).
        sug = _suggestion(
            title="Email the reviewers your response",
            probability_useful=10,
            benefit=10,
            cost_if_wrong=1,
        )
        with self._patch_assessment("irreversible", 2):
            assessment = await ex.assess_risk(sug)
        self.assertFalse(ex.is_auto_dispatchable(sug, assessment))

    async def test_high_risk_reversible_action_stays_proposal_only(self):
        ex = Executor(self.gum)
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("reversible", DEFAULT_MAX_RISK + 1):
            assessment = await ex.assess_risk(sug)
        self.assertFalse(ex.is_auto_dispatchable(sug, assessment))

    async def test_low_confidence_suggestion_stays_proposal_only(self):
        ex = Executor(self.gum)
        # Reversible and harmless, but the model isn't confident it's useful.
        sug = _suggestion(
            probability_useful=DEFAULT_MIN_PROBABILITY - 1,
            benefit=6,
            cost_if_wrong=2,
            cost_if_missed=6,
        )
        with self._patch_assessment("read_only", 1):
            assessment = await ex.assess_risk(sug)
        self.assertFalse(ex.is_auto_dispatchable(sug, assessment))

    async def test_non_surfaced_suggestion_stays_proposal_only(self):
        ex = Executor(self.gum)
        # A noisy suggestion the mixed-initiative decision would withhold: high
        # false-positive cost, low value. It must not act even if reversible.
        sug = _suggestion(
            probability_useful=8, benefit=1, cost_if_wrong=10, cost_if_missed=1
        )
        self.assertFalse(sug.should_surface)
        with self._patch_assessment("read_only", 1):
            assessment = await ex.assess_risk(sug)
        self.assertFalse(ex.is_auto_dispatchable(sug, assessment))

    async def test_thresholds_are_configurable(self):
        # A stricter executor (only read_only, only risk 1) rejects a reversible
        # risk-2 action the default would accept.
        strict = Executor(self.gum, max_risk=1)
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("reversible", 2):
            assessment = await strict.assess_risk(sug)
        self.assertFalse(strict.is_auto_dispatchable(sug, assessment))
        # And the default accepts the same case, confirming the knob is what moved.
        default = Executor(self.gum)
        self.assertTrue(default.is_auto_dispatchable(sug, assessment))


if __name__ == "__main__":
    unittest.main()
