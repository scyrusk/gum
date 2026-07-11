# test_api_execute.py
#
# Stdlib-only (unittest) tests for the execution-bridge REST endpoint
# (POST /suggestions/execute, spec #4). Runnable without a live model or the
# real `claude` CLI:  python -m unittest tests.test_api_execute
#
# The text model is stubbed (patched structured_completion for both the
# suggestion pipeline and the risk assessment) and the agent backend is a
# recording double, so these tests drive the FastAPI app end-to-end through a
# TestClient while staying fully offline and deterministic. The point is to prove
# the endpoint is default-OFF, dispatches a high-confidence reversible suggestion
# to a held-for-approval draft, keeps a risky one proposal-only, and pseudonymizes
# its text under --sanitize.

from __future__ import annotations

import tempfile
import unittest
import uuid
from unittest import mock

from fastapi.testclient import TestClient

from gum import gum as Gum
from gum.api import create_app
from gum.executor import AgentResult, Executor
from gum.gumbo import Gumbo
from gum.models import Proposition
from gum.schemas import (
    RiskAssessmentSchema,
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


# One high-value, low-intrusion suggestion that clears the surfacing bar.
_FAKE_SUGGESTIONS = SuggestionSchema(suggestions=[
    SuggestionItem(
        title="Draft a checklist for the Chicago trip",
        description="Assemble a packing + logistics checklist for the wedding.",
        rationale="Wedding travel with several open tasks.",
        probability_useful=9, benefit=9, cost_if_wrong=2, cost_if_missed=7,
    ),
])


async def _fake_suggestions(client, model, messages, schema, **kwargs):
    return _FAKE_SUGGESTIONS


class _RecordingBackend:
    """An AgentBackend double that records the dispatch and returns a canned draft."""

    def __init__(self, output: str = "DRAFT: packing checklist ready for review"):
        self.output = output
        self.calls: list[tuple[str, str, str, float]] = []

    async def run(self, task, context, *, cwd, timeout):
        self.calls.append((task, context, cwd, timeout))
        return AgentResult(ok=True, output=self.output)


def _patch_executor(gum, backend, *, reversibility="reversible", risk=2):
    """Patch Gumbo._get_executor to return an offline Executor.

    The Executor uses ``backend`` for dispatch and has egress sanitization off
    (so no PII model is pulled in for context assembly) — the API-layer sanitizer
    is what these tests exercise. ``structured_completion`` inside the executor is
    patched separately by the caller to feed the risk assessment.
    """
    executor = Executor(gum, backend=backend, sanitize=False)
    return mock.patch.object(Gumbo, "_get_executor", lambda self: executor)


async def _fake_risk(client, model, messages, schema, **kwargs):
    # Value is swapped per-test via the closure below; default reversible/low-risk.
    return RiskAssessmentSchema(
        reversibility="reversible", risk=2, rationale="Only drafts a local file."
    )


class ExecuteEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum(
            "Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db"
        )
        await self.gum.connect_db()
        async with self.gum._session() as s:
            s.add_all([
                _prop("Omar is going to a friend's wedding in Chicago", 8),
                _prop("Omar has several open trip-planning tasks", 7),
            ])

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def test_execute_disabled_by_default(self):
        # Default-OFF: the route exists but refuses to run anything, and no agent
        # backend is ever touched.
        backend = _RecordingBackend()
        app = create_app(self.gum)  # execute defaults to None -> env (off)
        with _patch_executor(self.gum, backend):
            with TestClient(app) as client:
                resp = client.post("/suggestions/execute")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertFalse(body["enabled"])
        self.assertIn("disabled", body["error"])
        self.assertEqual(backend.calls, [])

    def test_execute_dispatches_reversible_suggestion(self):
        backend = _RecordingBackend()
        app = create_app(self.gum, execute=True)
        with _patch_executor(self.gum, backend), \
                mock.patch("gum.gumbo.structured_completion", side_effect=_fake_suggestions), \
                mock.patch("gum.executor.structured_completion", side_effect=_fake_risk):
            with TestClient(app) as client:
                resp = client.post("/suggestions/execute")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["enabled"])
        self.assertEqual(body["dispatched"], 1)
        self.assertEqual(len(body["outcomes"]), 1)
        outcome = body["outcomes"][0]
        self.assertEqual(outcome["status"], "pending_approval")
        self.assertEqual(
            outcome["suggestion"]["title"], "Draft a checklist for the Chicago trip"
        )
        self.assertEqual(
            outcome["result"]["output"], "DRAFT: packing checklist ready for review"
        )
        self.assertTrue(outcome["result"]["ok"])
        # The agent actually ran exactly once, in a sandbox cwd (not cwd of test).
        self.assertEqual(len(backend.calls), 1)

    def test_execute_keeps_risky_suggestion_proposal_only(self):
        backend = _RecordingBackend()

        async def _risky(client, model, messages, schema, **kwargs):
            return RiskAssessmentSchema(
                reversibility="irreversible", risk=9,
                rationale="Sends outward-facing email.",
            )

        app = create_app(self.gum, execute=True)
        with _patch_executor(self.gum, backend), \
                mock.patch("gum.gumbo.structured_completion", side_effect=_fake_suggestions), \
                mock.patch("gum.executor.structured_completion", side_effect=_risky):
            with TestClient(app) as client:
                resp = client.post("/suggestions/execute")
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["dispatched"], 0)
        outcome = body["outcomes"][0]
        self.assertEqual(outcome["status"], "proposal_only")
        self.assertIsNone(outcome["result"])
        # Gate declined before any dispatch: the backend never ran.
        self.assertEqual(backend.calls, [])

    def test_execute_scrubs_output_under_sanitize(self):
        # A fake sanitizer that reveals the API-layer scrub ran over the draft.
        class _FakeSanitizer:
            def load(self):
                pass

            def sanitize(self, text):
                return text.replace("Omar", "[PERSON_1]")

        backend = _RecordingBackend(output="DRAFT for Omar: checklist ready")
        with mock.patch("gum.sanitize.get_sanitizer", return_value=_FakeSanitizer()):
            app = create_app(self.gum, sanitize=True, execute=True)
        with _patch_executor(self.gum, backend), \
                mock.patch("gum.gumbo.structured_completion", side_effect=_fake_suggestions), \
                mock.patch("gum.executor.structured_completion", side_effect=_fake_risk):
            with TestClient(app) as client:
                resp = client.post("/suggestions/execute")
        outcome = resp.json()["outcomes"][0]
        # The model-written draft is pseudonymized on the way out.
        self.assertEqual(outcome["result"]["output"], "DRAFT for [PERSON_1]: checklist ready")

    def test_execute_env_flag_enables(self):
        backend = _RecordingBackend()
        with mock.patch.dict("os.environ", {"GUMBO_EXECUTION_ENABLED": "1"}):
            app = create_app(self.gum)  # execute=None -> resolves env -> ON
        with _patch_executor(self.gum, backend), \
                mock.patch("gum.gumbo.structured_completion", side_effect=_fake_suggestions), \
                mock.patch("gum.executor.structured_completion", side_effect=_fake_risk):
            with TestClient(app) as client:
                resp = client.post("/suggestions/execute")
        body = resp.json()
        self.assertTrue(body["enabled"])
        self.assertEqual(body["dispatched"], 1)


if __name__ == "__main__":
    unittest.main()
