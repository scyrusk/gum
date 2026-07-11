# test_cli_agenda.py
#
# Stdlib-only (unittest) tests for the `gum agenda` CLI surface (gum/cli.py).
# Runnable without pytest or a live model:  python -m unittest tests.test_cli_agenda
#
# The text model is stubbed (patched structured_completion) and cli.gum is
# pointed at a temp DB, so these exercise the command's rendering (pretty +
# --json), the --window/--limit flags, and the sanitize path deterministically
# and offline.

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from gum import cli
from gum import gum as Gum
from gum.schemas import CommitmentItem, CommitmentSchema

# A fixed "now" is not injectable through the CLI (it builds its own radar), so
# these tests assert on structure/labels rather than absolute day counts.


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


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        limit=10,
        window=None,
        json=False,
        user_name="Omar",
        text_model="dummy-model",
        sanitize=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class CmdAgendaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Only the extraction call is stubbed here; disable the second-pass
        # verification so it doesn't hit the same stub (it has dedicated coverage
        # in tests.test_agenda.VerificationPassTests).
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
                _prop("Omar is idly browsing social media", 2),  # below the bar
            ])
        # cmd_agenda builds its own gum from args; hand it our temp-DB instance.
        self._patch_gum = mock.patch("gum.cli.gum", return_value=self.gum)
        self._patch_gum.start()

    async def asyncTearDown(self):
        self._patch_gum.stop()
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def _fake_completion(self):
        async def fake(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(source_index=1, title="Submit the NSF grant proposal",
                               due_date="2999-07-20", source="NSF",
                               status_guess="in progress"),
                CommitmentItem(source_index=2, title="Send reviewer comments",
                               due_date=None, source="a colleague",
                               status_guess="not started"),
            ])
        return fake

    async def _run(self, args) -> str:
        buf = io.StringIO()
        with mock.patch("gum.agenda.structured_completion",
                        side_effect=self._fake_completion()):
            with redirect_stdout(buf):
                await cli.cmd_agenda(args)
        return buf.getvalue()

    async def test_pretty_output_lists_commitments(self):
        out = await self._run(_args())
        self.assertIn("Commitment radar", out)
        self.assertIn("Submit the NSF grant proposal", out)
        self.assertIn("Send reviewer comments", out)
        self.assertIn("due 2999-07-20", out)
        self.assertIn("in progress", out)

    async def test_json_output_is_valid_and_structured(self):
        out = await self._run(_args(json=True))
        data = json.loads(out)
        self.assertEqual(len(data), 2)
        titles = {row["title"] for row in data}
        self.assertIn("Submit the NSF grant proposal", titles)
        for key in ("due_date", "source", "status_guess", "urgency",
                    "days_until_due", "confidence"):
            self.assertIn(key, data[0])
        # Dated commitment ranks above the undated one.
        self.assertEqual(data[0]["title"], "Submit the NSF grant proposal")

    async def test_limit_caps_results(self):
        out = await self._run(_args(json=True, limit=1))
        self.assertEqual(len(json.loads(out)), 1)

    async def test_window_excludes_far_future(self):
        # The NSF deadline is far in the future; a tight window drops it but keeps
        # the undated commitment.
        out = await self._run(_args(json=True, window=7))
        titles = {row["title"] for row in json.loads(out)}
        self.assertNotIn("Submit the NSF grant proposal", titles)
        self.assertIn("Send reviewer comments", titles)

    async def test_sanitize_scrubs_text_fields(self):
        called = {}

        async def fake_scrub(text, enabled):
            called["enabled"] = enabled
            return "[REDACTED]" if enabled else text

        with mock.patch("gum.cli._scrub", side_effect=fake_scrub):
            out = await self._run(_args(sanitize=True, json=True))
        self.assertTrue(called["enabled"])
        data = json.loads(out)
        self.assertTrue(all(row["title"] == "[REDACTED]" for row in data))

    async def test_empty_radar_message(self):
        async def empty(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[])

        buf = io.StringIO()
        with mock.patch("gum.agenda.structured_completion", side_effect=empty):
            with redirect_stdout(buf):
                await cli.cmd_agenda(_args())
        self.assertIn("Nothing on the radar", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
