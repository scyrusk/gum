# test_cli_execute.py
#
# Stdlib-only (unittest) tests for the `gum execute --review` approve/reject
# surface (spec #4, Step 3). Runnable offline without a model or the claude CLI:
#   python -m unittest tests.test_cli_execute
#
# The review loop is driven with injected prompt/out functions and a fake GUM
# that records add_suggestion_feedback calls, so we verify the accept/reject
# signal flows back through the existing suggestion-feedback plumbing without
# touching stdin/stdout or a database.

from __future__ import annotations

import unittest

from gum.cli import review_outcomes
from gum.executor import (
    STATUS_FAILED,
    STATUS_PENDING_APPROVAL,
    STATUS_PROPOSAL_ONLY,
    AgentResult,
    ExecutionOutcome,
    RiskAssessment,
)
from gum.gumbo import Suggestion


def _suggestion(title: str, description: str = "do the thing") -> Suggestion:
    return Suggestion(
        title=title,
        description=description,
        rationale="because",
        probability_useful=9,
        benefit=8,
        cost_if_wrong=2,
        cost_if_missed=6,
        expected_utility=5.0,
        should_surface=True,
    )


def _pending(title: str, output: str = "here is your draft") -> ExecutionOutcome:
    return ExecutionOutcome(
        suggestion=_suggestion(title),
        status=STATUS_PENDING_APPROVAL,
        assessment=RiskAssessment(reversibility="reversible", risk=2, rationale="ok"),
        context="(grounding)",
        result=AgentResult(ok=True, output=output),
    )


class FakeGum:
    """Records add_suggestion_feedback calls the way the real GUM would receive them."""

    def __init__(self) -> None:
        self.feedback: list[dict] = []

    async def add_suggestion_feedback(self, *, title, vote, description=None, focus=None):
        self.feedback.append(
            {"title": title, "vote": vote, "description": description, "focus": focus}
        )
        return vote in ("up", "down")


class _Prompter:
    """A canned-answer stand-in for input()."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, message: str = "") -> str:
        self.prompts.append(message)
        return self._answers.pop(0)


async def _run(gum, outcomes, answers, *, interactive=True):
    out_lines: list[str] = []
    prompter = _Prompter(answers)
    recorded = await review_outcomes(
        gum,
        outcomes,
        interactive=interactive,
        prompt=prompter,
        out=out_lines.append,
    )
    return recorded, out_lines, prompter


class ReviewOutcomesTests(unittest.IsolatedAsyncioTestCase):
    async def test_approve_records_thumbs_up(self):
        gum = FakeGum()
        recorded, _out, _p = await _run(gum, [_pending("Draft an email")], ["a"])
        self.assertEqual(recorded, [{"title": "Draft an email", "vote": "up"}])
        self.assertEqual(len(gum.feedback), 1)
        self.assertEqual(gum.feedback[0]["vote"], "up")
        self.assertEqual(gum.feedback[0]["title"], "Draft an email")
        # The suggestion description rides along so the feedback observation is rich.
        self.assertEqual(gum.feedback[0]["description"], "do the thing")

    async def test_reject_records_thumbs_down(self):
        gum = FakeGum()
        recorded, _out, _p = await _run(gum, [_pending("Risky thing")], ["reject"])
        self.assertEqual(recorded, [{"title": "Risky thing", "vote": "down"}])
        self.assertEqual(gum.feedback[0]["vote"], "down")

    async def test_skip_records_nothing(self):
        gum = FakeGum()
        recorded, _out, _p = await _run(gum, [_pending("Meh")], ["s"])
        self.assertEqual(recorded, [])
        self.assertEqual(gum.feedback, [])

    async def test_reprompts_on_invalid_answer(self):
        gum = FakeGum()
        recorded, out, prompter = await _run(gum, [_pending("Draft")], ["huh?", "y"])
        self.assertEqual(recorded, [{"title": "Draft", "vote": "up"}])
        # Two prompts were needed (first answer was rejected).
        self.assertEqual(len(prompter.prompts), 2)
        self.assertTrue(any("Please answer" in line for line in out))

    async def test_only_pending_approval_is_prompted(self):
        gum = FakeGum()
        proposal = ExecutionOutcome(
            suggestion=_suggestion("Held"),
            status=STATUS_PROPOSAL_ONLY,
            assessment=RiskAssessment(reversibility="irreversible", risk=8, rationale="x"),
            reason="did not clear the gate",
        )
        failed = ExecutionOutcome(
            suggestion=_suggestion("Broke"),
            status=STATUS_FAILED,
            result=AgentResult(ok=False, output="", error="boom"),
            reason="boom",
        )
        pending = _pending("Ready")
        # Only the pending one consumes an answer; the other two are display-only.
        recorded, _out, prompter = await _run(gum, [proposal, failed, pending], ["a"])
        self.assertEqual(recorded, [{"title": "Ready", "vote": "up"}])
        self.assertEqual(len(prompter.prompts), 1)

    async def test_non_interactive_records_nothing_but_renders_all(self):
        gum = FakeGum()
        recorded, out, _p = await _run(
            gum, [_pending("A"), _pending("B")], [], interactive=False
        )
        self.assertEqual(recorded, [])
        self.assertEqual(gum.feedback, [])
        # Both outcomes still show up in the rendered output.
        joined = "\n".join(out)
        self.assertIn("A", joined)
        self.assertIn("B", joined)

    async def test_multiple_outcomes_record_in_order(self):
        gum = FakeGum()
        recorded, _out, _p = await _run(
            gum, [_pending("First"), _pending("Second")], ["a", "r"]
        )
        self.assertEqual(
            recorded,
            [
                {"title": "First", "vote": "up"},
                {"title": "Second", "vote": "down"},
            ],
        )
        self.assertEqual([f["vote"] for f in gum.feedback], ["up", "down"])

    async def test_draft_output_is_rendered(self):
        gum = FakeGum()
        _recorded, out, _p = await _run(
            gum, [_pending("Email", output="Dear team, ...")], ["s"]
        )
        self.assertTrue(any("Dear team, ..." in line for line in out))


if __name__ == "__main__":
    unittest.main()
