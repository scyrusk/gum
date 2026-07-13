from __future__ import annotations

import re
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from gum import gum as Gum
from gum.models import Observation, Proposition
from gum.schemas import PropositionSchema, Update


class TemporalPropositionPromptTests(unittest.IsolatedAsyncioTestCase):
    """The propose/revise prompts must anchor today's date and instruct the
    model to preserve deadlines/dates, so downstream deadline extraction
    (gum.agenda) has absolute dates to rank instead of dropping them upstream.
    """

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name)
        await self.gum.connect_db()

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    @staticmethod
    def _empty_result(*args, **kwargs):
        return PropositionSchema(propositions=[])

    async def _propose_prompt(self) -> str:
        with mock.patch(
            "gum.gum.structured_completion", side_effect=self._empty_result
        ) as completion:
            await self.gum._construct_propositions(
                Update(content="Grant proposal due July 20", content_type="input_text")
            )
        return completion.call_args.args[2][-1]["content"]

    async def _revise_prompt(self) -> str:
        with mock.patch(
            "gum.gum.structured_completion", side_effect=self._empty_result
        ) as completion:
            await self.gum._revise_propositions(
                [
                    Observation(
                        observer_name="screen",
                        content="deadline next Friday",
                        content_type="input_text",
                    )
                ],
                [
                    Proposition(
                        text="Omar has a submission coming up",
                        reasoning="evidence",
                        confidence=5,
                        decay=5,
                    )
                ],
            )
        return completion.call_args.args[2][-1]["content"]

    async def test_propose_prompt_injects_todays_date(self):
        prompt = await self._propose_prompt()
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        self.assertIn(today, prompt)
        # No placeholder must survive un-substituted.
        self.assertNotIn("{today}", prompt)

    async def test_propose_prompt_has_temporal_grounding_rule(self):
        prompt = await self._propose_prompt()
        self.assertIn("Temporal grounding", prompt)
        self.assertIn("YYYY-MM-DD", prompt)
        # Dates/deadlines must be named as entities to preserve.
        self.assertRegex(prompt, r"dates,?\s+times,?\s+and\s+deadlines")

    async def test_revise_prompt_injects_todays_date_and_preserves_deadlines(self):
        prompt = await self._revise_prompt()
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        self.assertIn(today, prompt)
        self.assertNotIn("{today}", prompt)
        self.assertRegex(prompt, r"dates,?\s+times,?\s+and\s+deadlines")

    def test_today_str_is_local_iso_with_weekday(self):
        # Guards the format the prompts rely on (absolute ISO date + weekday).
        self.assertRegex(self.gum._today_str(), r"^\d{4}-\d{2}-\d{2} \(\w+\)$")


if __name__ == "__main__":
    unittest.main()
