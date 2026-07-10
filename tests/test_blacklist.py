from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gum import gum as Gum
from gum.models import Observation, Proposition
from gum.schemas import PropositionSchema, Update


class PropositionBlacklistTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.blacklist = Path(self._tmp.name) / "rules.txt"
        self.gum = Gum(
            "Omar",
            "dummy-model",
            data_directory=self._tmp.name,
            blacklist_file=str(self.blacklist),
        )
        await self.gum.connect_db()

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    @staticmethod
    def _empty_result(*args, **kwargs):
        return PropositionSchema(propositions=[])

    async def test_proposal_prompt_contains_each_active_rule(self):
        self.blacklist.write_text(
            "# Private topics\nDo not generate propositions about passwords.\n\n"
            "Do not generate propositions about credit cards.\n",
            encoding="utf-8",
        )

        with mock.patch(
            "gum.gum.structured_completion", side_effect=self._empty_result
        ) as completion:
            await self.gum._construct_propositions(
                Update(content="visible screen text", content_type="input_text")
            )

        prompt = completion.call_args.args[2][0]["content"]
        self.assertIn("1. Do not generate propositions about passwords.", prompt)
        self.assertIn("2. Do not generate propositions about credit cards.", prompt)
        self.assertNotIn("# Private topics", prompt)
        self.assertIn("return an empty `propositions` list", prompt)

    async def test_revision_prompt_enforces_rules_too(self):
        self.blacklist.write_text(
            "Do not generate propositions about adult content.\n", encoding="utf-8"
        )
        existing = Proposition(
            text="existing proposition",
            reasoning="existing evidence",
            confidence=5,
            decay=5,
        )
        observation = Observation(
            observer_name="screen", content="screen content", content_type="input_text"
        )

        with mock.patch(
            "gum.gum.structured_completion", side_effect=self._empty_result
        ) as completion:
            await self.gum._revise_propositions([observation], [existing])

        prompt = completion.call_args.args[2][0]["content"]
        self.assertIn("Do not generate propositions about adult content.", prompt)
        self.assertIn("These rules take priority", prompt)

    async def test_rules_are_reloaded_and_missing_file_is_allowed(self):
        with mock.patch(
            "gum.gum.structured_completion", side_effect=self._empty_result
        ) as completion:
            await self.gum._construct_propositions(
                Update(content="first", content_type="input_text")
            )
            first_prompt = completion.call_args.args[2][0]["content"]

            self.blacklist.write_text("Exclude financial credentials.\n", encoding="utf-8")
            await self.gum._construct_propositions(
                Update(content="second", content_type="input_text")
            )
            second_prompt = completion.call_args.args[2][0]["content"]

        self.assertNotIn("Proposition Content Blacklist", first_prompt)
        self.assertIn("Exclude financial credentials.", second_prompt)


if __name__ == "__main__":
    unittest.main()
