from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sqlalchemy import select

from gum import gum as Gum
from gum.models import Observation, Proposition
from gum.schemas import (
    BlacklistComplianceSchema,
    PropositionItem,
    PropositionSchema,
    Update,
)


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

        generated = PropositionSchema(
            propositions=[
                PropositionItem(
                    proposition="Omar visits adult websites",
                    reasoning="Adult content was visible",
                    confidence=8,
                    decay=5,
                )
            ]
        )
        with mock.patch(
            "gum.gum.structured_completion",
            side_effect=[generated, BlacklistComplianceSchema(allowed_indices=[])],
        ) as completion:
            result = await self.gum._revise_propositions([observation], [existing])

        prompt = completion.call_args_list[0].args[2][0]["content"]
        self.assertIn("Do not generate propositions about adult content.", prompt)
        self.assertIn("These rules take priority", prompt)
        self.assertEqual(result, [])
        self.assertEqual(completion.call_count, 2)

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

    async def test_noncompliant_model_output_is_removed_by_second_pass(self):
        self.blacklist.write_text(
            "Do not generate propositions about passwords.\n", encoding="utf-8"
        )
        generated = PropositionSchema(
            propositions=[
                PropositionItem(
                    proposition="Omar uses a password manager",
                    reasoning="A password manager was visible",
                    confidence=8,
                    decay=5,
                ),
                PropositionItem(
                    proposition="Omar uses a dark editor theme",
                    reasoning="The editor background was dark",
                    confidence=7,
                    decay=5,
                ),
            ]
        )

        with mock.patch(
            "gum.gum.structured_completion",
            side_effect=[
                generated,
                BlacklistComplianceSchema(allowed_indices=[1]),
            ],
        ) as completion:
            result = await self.gum._construct_propositions(
                Update(content="screen content", content_type="input_text")
            )

        self.assertEqual([item.proposition for item in result], ["Omar uses a dark editor theme"])
        compliance_messages = completion.call_args_list[1].args[2]
        self.assertEqual(
            [message["role"] for message in compliance_messages], ["system", "user"]
        )
        compliance_instructions = compliance_messages[0]["content"]
        candidate_data = compliance_messages[1]["content"]
        self.assertIn(
            "strict proposition-content compliance checker", compliance_instructions
        )
        self.assertIn(
            "Do not generate propositions about passwords.", compliance_instructions
        )
        self.assertNotIn("empty `propositions` list", compliance_instructions)
        self.assertNotIn("Omar uses a password manager", compliance_instructions)
        self.assertIn("Omar uses a password manager", candidate_data)
        self.assertNotIn("Do not generate propositions about passwords.", candidate_data)

    async def test_failed_compliance_check_suppresses_model_output(self):
        self.blacklist.write_text("Exclude passwords.\n", encoding="utf-8")
        generated = PropositionSchema(
            propositions=[
                PropositionItem(
                    proposition="Omar prefers dark mode",
                    reasoning="A dark interface was visible",
                    confidence=7,
                    decay=5,
                )
            ]
        )

        with mock.patch(
            "gum.gum.structured_completion",
            side_effect=[generated, RuntimeError("validator unavailable")],
        ):
            result = await self.gum._construct_propositions(
                Update(content="screen content", content_type="input_text")
            )

        self.assertEqual(result, [])

    async def test_only_compliant_propositions_are_persisted_for_a_batch(self):
        self.blacklist.write_text(
            "Do not generate propositions about passwords.\n", encoding="utf-8"
        )
        generated = PropositionSchema(
            propositions=[
                PropositionItem(
                    proposition="Omar's password is visible",
                    reasoning="A password field was shown",
                    confidence=9,
                    decay=5,
                ),
                PropositionItem(
                    proposition="Omar uses a dark editor theme",
                    reasoning="The editor background was dark",
                    confidence=7,
                    decay=5,
                ),
            ]
        )
        batch = [
            {
                "id": "batch-observation-1",
                "observer_name": "screen",
                "content": "screen content",
                "content_type": "input_text",
            }
        ]

        with (
            mock.patch(
                "gum.gum.structured_completion",
                side_effect=[
                    generated,
                    BlacklistComplianceSchema(allowed_indices=[1]),
                ],
            ),
            mock.patch.object(self.gum.batcher, "ack_batch") as ack_batch,
        ):
            await self.gum._process_batch(batch)

        ack_batch.assert_called_once_with(batch)

        async with self.gum._session() as session:
            persisted = list((await session.scalars(select(Proposition))).all())

        self.assertEqual(
            [proposition.text for proposition in persisted],
            ["Omar uses a dark editor theme"],
        )
        self.assertNotIn(
            "password",
            " ".join(
                f"{proposition.text} {proposition.reasoning}"
                for proposition in persisted
            ).lower(),
        )

    async def test_unreadable_blacklist_suppresses_proposition_writes(self):
        self.blacklist.write_text("Exclude passwords.\n", encoding="utf-8")

        with (
            mock.patch("builtins.open", side_effect=PermissionError("denied")),
            mock.patch("gum.gum.structured_completion") as completion,
        ):
            proposed = await self.gum._construct_propositions(
                Update(content="password visible", content_type="input_text")
            )
            revised = await self.gum._revise_propositions(
                [
                    Observation(
                        observer_name="screen",
                        content="password visible",
                        content_type="input_text",
                    )
                ],
                [
                    Proposition(
                        text="existing proposition",
                        reasoning="existing evidence",
                        confidence=5,
                        decay=5,
                    )
                ],
            )

        self.assertEqual(proposed, [])
        self.assertEqual(revised, [])
        completion.assert_not_called()

    async def test_blocked_revision_preserves_existing_proposition(self):
        self.blacklist.write_text(
            "Do not generate propositions about passwords.\n", encoding="utf-8"
        )
        async with self.gum._session() as session:
            existing = Proposition(
                text="Omar uses a password manager",
                reasoning="A password manager was visible",
                confidence=8,
                decay=5,
                revision_group="existing-group",
                version=1,
            )
            observation = Observation(
                observer_name="screen",
                content="password settings were visible",
                content_type="input_text",
            )
            session.add_all([existing, observation])
            await session.flush()
            existing_id = existing.id

        with mock.patch(
            "gum.gum.structured_completion", side_effect=self._empty_result
        ):
            async with self.gum._session() as session:
                existing = await session.get(Proposition, existing_id)
                new_observation = Observation(
                    observer_name="screen",
                    content="another password screen",
                    content_type="input_text",
                )
                session.add(new_observation)
                await session.flush()
                await self.gum._handle_similar(
                    session, [existing], [new_observation]
                )

        async with self.gum._session() as session:
            preserved = await session.get(Proposition, existing_id)
            self.assertIsNotNone(preserved)
            self.assertEqual(preserved.text, "Omar uses a password manager")

    async def test_only_compliant_revisions_replace_existing_propositions(self):
        self.blacklist.write_text(
            "Do not generate propositions about passwords.\n", encoding="utf-8"
        )
        async with self.gum._session() as session:
            existing = Proposition(
                text="Omar configures development tools",
                reasoning="A settings screen was visible",
                confidence=6,
                decay=5,
                revision_group="existing-group",
                version=1,
            )
            observation = Observation(
                observer_name="screen",
                content="updated settings screen",
                content_type="input_text",
            )
            session.add_all([existing, observation])
            await session.flush()
            existing_id = existing.id
            observation_id = observation.id

        revised = PropositionSchema(
            propositions=[
                PropositionItem(
                    proposition="Omar stores passwords in a development tool",
                    reasoning="A password setting was visible",
                    confidence=8,
                    decay=5,
                ),
                PropositionItem(
                    proposition="Omar uses a dark editor theme",
                    reasoning="The settings screen showed a dark theme",
                    confidence=7,
                    decay=5,
                ),
            ]
        )
        with mock.patch(
            "gum.gum.structured_completion",
            side_effect=[
                revised,
                BlacklistComplianceSchema(allowed_indices=[1]),
            ],
        ):
            async with self.gum._session() as session:
                existing = await session.get(Proposition, existing_id)
                observation = await session.get(Observation, observation_id)
                await self.gum._handle_similar(session, [existing], [observation])

        async with self.gum._session() as session:
            persisted = list((await session.scalars(select(Proposition))).all())

        self.assertEqual(
            [proposition.text for proposition in persisted],
            ["Omar uses a dark editor theme"],
        )
        self.assertNotIn(
            "password",
            " ".join(
                f"{proposition.text} {proposition.reasoning}"
                for proposition in persisted
            ).lower(),
        )


if __name__ == "__main__":
    unittest.main()
