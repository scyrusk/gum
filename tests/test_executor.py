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

import asyncio
import os
import shlex
import signal
import tempfile
import unittest
import uuid
from unittest import mock

from gum import gum as Gum
from gum.executor import (
    DEFAULT_MAX_RISK,
    DEFAULT_MIN_PROBABILITY,
    STATUS_FAILED,
    STATUS_PENDING_APPROVAL,
    STATUS_PROPOSAL_ONLY,
    AgentResult,
    ClaudeCLIBackend,
    Executor,
    RiskAssessment,
)
from gum.gumbo import Suggestion, expected_utility
from gum.models import Proposition
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


class _FakeSanitizer:
    """Deterministic stand-in for the PII model (no torch/transformers needed)."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def sanitize(self, text: str) -> str:
        return self.sanitize_map(text)[0]

    def sanitize_map(self, text: str) -> tuple[str, dict[str, str]]:
        aliases: dict[str, str] = {}
        for raw, pseudo in self._mapping.items():
            if raw in text:
                aliases[raw] = pseudo
            text = text.replace(raw, pseudo)
        return text, aliases


def _prop(text: str, confidence: int) -> Proposition:
    return Proposition(
        text=text,
        reasoning=f"because of {text}",
        confidence=confidence,
        decay=5,
        revision_group=uuid.uuid4().hex,
        version=1,
    )


class AssembleContextTests(unittest.IsolatedAsyncioTestCase):
    """The executor grounds a dispatched task on the SAME assembly the MCP uses.

    Spec #4 forbids a second grounding path, so these assert that
    ``assemble_context`` retrieves the relevant, high-confidence propositions and
    pseudonymizes them on egress (the backend shells out to an off-device agent).
    """

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum(
            "Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db"
        )
        await self.gum.connect_db()

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    async def _seed(self, *props: Proposition) -> None:
        async with self.gum._session() as s:
            s.add_all(list(props))

    async def test_context_grounds_on_the_suggestion(self):
        await self._seed(
            _prop("Omar is applying for a Schmidt Foundation research grant", 9),
            # A topically-unrelated fact sharing no query terms with the task.
            _prop("Weekend mountain hikes are a favorite pastime", 6),
        )
        ex = Executor(self.gum, sanitize=False)
        sug = _suggestion(
            title="Draft the Schmidt Foundation grant proposal",
            description="Prepare a first draft of the research grant application.",
        )
        block = await ex.assemble_context(sug)
        self.assertIn("Schmidt Foundation research grant", block)
        # The unrelated proposition is not retrieved for this task.
        self.assertNotIn("mountain hikes", block)

    async def test_context_is_pseudonymized_for_off_device_dispatch(self):
        # sanitize defaults ON, fail-closed; here we inject a deterministic fake
        # so no PII model is required. The backend gets pseudo-IDs, never raw PII.
        await self._seed(
            _prop("Omar is applying for a Schmidt Foundation research grant", 9),
        )
        fake = _FakeSanitizer({"Schmidt": "[ORG_1]", "Omar": "[PERSON_1]"})
        ex = Executor(self.gum, sanitizer=fake)
        sug = _suggestion(
            title="Draft the Schmidt grant proposal",
            description="Prepare a first draft for Omar.",
        )
        block = await ex.assemble_context(sug)
        self.assertIn("[ORG_1]", block)
        self.assertNotIn("Schmidt", block)
        self.assertNotIn("Omar", block)
        self.assertIn("pseudonymized", block.lower())

    async def test_thin_gum_yields_honest_empty_context(self):
        # Nothing confident to ground on → an honest "no context" block, not an
        # empty string, so the backend prompt reads correctly.
        ex = Executor(self.gum, sanitize=False)
        block = await ex.assemble_context(_suggestion())
        self.assertIn("no confident context", block.lower())


class _RecordingBackend:
    """A mock AgentBackend that records its inputs and returns a canned result."""

    def __init__(self, result: AgentResult):
        self._result = result
        self.calls: list[dict] = []

    async def run(self, task, context, *, cwd, timeout):
        # Record whether the sandbox cwd exists *at call time*: the executor tears
        # the per-dispatch workspace down once run() returns, so a post-dispatch
        # os.path.isdir() check would race the cleanup.
        self.calls.append(
            {
                "task": task,
                "context": context,
                "cwd": cwd,
                "cwd_isdir": os.path.isdir(cwd),
                "timeout": timeout,
            }
        )
        return self._result


class DispatchFlowTests(unittest.IsolatedAsyncioTestCase):
    """The end-to-end dispatch flow: gate → ground → run backend → hold artifact.

    The agent backend is mocked so no ``claude`` CLI or model is required. These
    assert that only gate-clearing suggestions reach the backend, that a run's
    output lands in a pending-approval outcome, and that a risky suggestion never
    dispatches.
    """

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum(
            "Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db"
        )
        await self.gum.connect_db()

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    def _patch_assessment(self, reversibility: str, risk: int):
        async def fake_completion(client, model, messages, schema, **kwargs):
            return RiskAssessmentSchema(
                reversibility=reversibility, risk=risk, rationale="stub"
            )

        return mock.patch(
            "gum.executor.structured_completion", side_effect=fake_completion
        )

    async def test_dispatch_runs_backend_and_holds_for_approval(self):
        backend = _RecordingBackend(AgentResult(ok=True, output="three rental shops: ..."))
        ex = Executor(self.gum, backend=backend, sanitize=False)
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("read_only", 1):
            outcome = await ex.dispatch(sug)

        self.assertEqual(outcome.status, STATUS_PENDING_APPROVAL)
        self.assertTrue(outcome.is_pending_approval)
        self.assertTrue(outcome.dispatched)
        self.assertEqual(outcome.result.output, "three rental shops: ...")
        # The backend actually ran, exactly once, confined to a workspace cwd with
        # the executor's timeout, and grounded on the assembled task.
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual(call["timeout"], ex.timeout)
        self.assertTrue(call["cwd_isdir"])  # the sandbox existed during the run
        self.assertIn(sug.title, call["task"])
        self.assertIn("review", call["task"].lower())
        # The per-dispatch sandbox is torn down once the run returns, so nothing
        # accumulates and no run's scratch state leaks into the next dispatch.
        self.assertFalse(os.path.exists(call["cwd"]))

    async def test_dispatched_prompt_is_pseudonymized_for_off_device_backend(self):
        # The shipped backend ships the whole instruction off-device, so every
        # identity in it must be pseudonymized on egress — not only the grounding
        # context, but the two identities the prompt itself reintroduces: the
        # GUM-generated suggestion text (which may embed real names/projects) and
        # the user's own name (stamped in by EXECUTION_AGENT_PROMPT). Leaving
        # either raw leaks it and lets the agent tie a pseudo-ID back to the user.
        await self._seed_prop("Omar is applying for a Schmidt Foundation grant", 9)
        fake = _FakeSanitizer({"Schmidt": "[ORG_1]", "Omar": "[PERSON_1]"})
        backend = _RecordingBackend(AgentResult(ok=True, output="draft"))
        ex = Executor(self.gum, backend=backend, sanitizer=fake)
        sug = _suggestion(
            title="Draft the Schmidt grant proposal",
            description="Prepare a first draft for Omar.",
            probability_useful=9,
        )
        with self._patch_assessment("read_only", 1):
            outcome = await ex.dispatch(sug)

        self.assertEqual(outcome.status, STATUS_PENDING_APPROVAL)
        task = backend.calls[0]["task"]
        # The suggestion text AND the user's name reach the backend only as their
        # stable pseudo-IDs — the same ones the grounding context carries.
        self.assertIn("[ORG_1]", task)
        self.assertIn("[PERSON_1]", task)
        self.assertNotIn("Schmidt", task)
        self.assertNotIn("Omar", task)

    async def _seed_prop(self, text: str, confidence: int) -> None:
        async with self.gum._session() as s:
            s.add(_prop(text, confidence))

    async def test_each_dispatch_gets_an_isolated_cleaned_workspace(self):
        backend = _RecordingBackend(AgentResult(ok=True, output="draft"))
        ex = Executor(self.gum, backend=backend, sanitize=False)
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("read_only", 1):
            await ex.dispatch(sug)
            await ex.dispatch(sug)

        # Two runs, two *distinct* sandboxes, each removed after its run — so one
        # dispatch's scratch state can never bleed into the next, and nothing piles
        # up under the workspace root.
        cwds = [c["cwd"] for c in backend.calls]
        self.assertEqual(len(set(cwds)), 2)
        for c in backend.calls:
            self.assertTrue(c["cwd_isdir"])
            self.assertFalse(os.path.exists(c["cwd"]))
        # Both sandboxes lived under the shared workspace root, which is left empty.
        root = ex._ensure_workspace_root()
        for c in cwds:
            self.assertEqual(os.path.dirname(c), root)
        self.assertEqual(os.listdir(root), [])

    async def test_workspace_is_cleaned_up_even_when_backend_fails(self):
        backend = _RecordingBackend(
            AgentResult(ok=False, output="", error="agent timed out after 120s")
        )
        ex = Executor(self.gum, backend=backend, sanitize=False)
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("reversible", 2):
            outcome = await ex.dispatch(sug)

        self.assertEqual(outcome.status, STATUS_FAILED)
        # A failed/timed-out run must not leave its sandbox behind.
        self.assertFalse(os.path.exists(backend.calls[0]["cwd"]))
        self.assertEqual(os.listdir(ex._ensure_workspace_root()), [])

    async def test_risky_suggestion_never_dispatches(self):
        backend = _RecordingBackend(AgentResult(ok=True, output="should not run"))
        ex = Executor(self.gum, backend=backend, sanitize=False)
        sug = _suggestion(title="Email the reviewers", probability_useful=10)
        with self._patch_assessment("irreversible", 2):
            outcome = await ex.dispatch(sug)

        self.assertEqual(outcome.status, STATUS_PROPOSAL_ONLY)
        self.assertFalse(outcome.dispatched)
        self.assertIsNone(outcome.result)
        self.assertEqual(backend.calls, [])  # the agent was never invoked

    async def test_backend_failure_lands_failed_not_pending(self):
        backend = _RecordingBackend(
            AgentResult(ok=False, output="", error="agent timed out after 120s")
        )
        ex = Executor(self.gum, backend=backend, sanitize=False)
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("reversible", 2):
            outcome = await ex.dispatch(sug)

        self.assertEqual(outcome.status, STATUS_FAILED)
        self.assertFalse(outcome.is_pending_approval)
        self.assertIn("timed out", outcome.reason)

    async def test_assessment_failure_fails_closed_without_dispatching(self):
        # A flaky/failed safety classifier (model error, malformed response) must
        # NOT dispatch and must NOT propagate: an un-assessable action is exactly
        # what the gate exists to hold, and one bad assessment must not abort a
        # whole execute() batch. It fails closed to proposal-only for this one.
        backend = _RecordingBackend(AgentResult(ok=True, output="should not run"))
        ex = Executor(self.gum, backend=backend, sanitize=False)
        sug = _suggestion(probability_useful=10)

        async def boom(*args, **kwargs):
            raise RuntimeError("local model unavailable")

        with mock.patch("gum.executor.structured_completion", side_effect=boom):
            outcome = await ex.dispatch(sug)

        self.assertEqual(outcome.status, STATUS_PROPOSAL_ONLY)
        self.assertFalse(outcome.dispatched)
        self.assertIsNone(outcome.assessment)  # never obtained
        self.assertIsNone(outcome.result)
        self.assertIn("assessment failed", outcome.reason)
        self.assertEqual(backend.calls, [])  # the agent was never invoked

    async def test_grounding_failure_fails_closed_without_dispatching(self):
        # The gate cleared, but building the grounded, pseudonymized prompt fails
        # (a transient retrieval error, or the fail-closed egress sanitizer refusing
        # to load). We must NOT dispatch an un-grounded/un-sanitized prompt to the
        # off-device backend, and must NOT propagate — one setup failure must not
        # abort a whole execute() batch. It fails closed to proposal-only, keeping
        # the assessment the gate already obtained.
        backend = _RecordingBackend(AgentResult(ok=True, output="should not run"))
        ex = Executor(self.gum, backend=backend, sanitize=False)
        sug = _suggestion(probability_useful=10)

        async def boom(self_, suggestion):
            raise RuntimeError("sanitizer PII model unavailable")

        with self._patch_assessment("read_only", 1), mock.patch.object(
            Executor, "assemble_context", boom
        ):
            outcome = await ex.dispatch(sug)

        self.assertEqual(outcome.status, STATUS_PROPOSAL_ONLY)
        self.assertFalse(outcome.dispatched)
        self.assertIsNotNone(outcome.assessment)  # the gate ran and cleared
        self.assertIsNone(outcome.result)
        self.assertIn("could not build grounded prompt", outcome.reason)
        self.assertEqual(backend.calls, [])  # the off-device agent was never invoked

    async def test_missing_backend_stays_proposal_only(self):
        ex = Executor(self.gum, sanitize=False)  # no backend configured
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("read_only", 1):
            outcome = await ex.dispatch(sug)
        self.assertEqual(outcome.status, STATUS_PROPOSAL_ONLY)
        self.assertIn("backend", outcome.reason)

    async def test_outcome_to_dict_is_json_friendly(self):
        backend = _RecordingBackend(AgentResult(ok=True, output="draft"))
        ex = Executor(self.gum, backend=backend, sanitize=False)
        sug = _suggestion(probability_useful=9)
        with self._patch_assessment("read_only", 1):
            outcome = await ex.dispatch(sug)
        d = outcome.to_dict()
        self.assertEqual(d["status"], STATUS_PENDING_APPROVAL)
        self.assertEqual(d["suggestion"]["title"], sug.title)
        self.assertEqual(d["result"]["output"], "draft")
        self.assertEqual(d["assessment"]["reversibility"], "read_only")


class ClaudeCLIBackendTests(unittest.IsolatedAsyncioTestCase):
    """The shipped backend: subprocess mechanics, sandboxing, and failure modes.

    Uses a stand-in executable (``cat``/a missing binary) instead of the real
    ``claude`` CLI so these run offline and deterministically.
    """

    async def test_missing_cli_returns_error_not_raises(self):
        backend = ClaudeCLIBackend(command="definitely-not-a-real-binary-xyz")
        with tempfile.TemporaryDirectory() as cwd:
            result = await backend.run("do it", "ctx", cwd=cwd, timeout=5)
        self.assertFalse(result.ok)
        self.assertIn("not found", result.error)

    async def test_captures_stdout_and_feeds_prompt_on_stdin(self):
        # `cat` echoes stdin to stdout, so it stands in for a CLI that reads the
        # prompt and prints a result: we can assert the prompt reached the process
        # and its output was captured.
        backend = ClaudeCLIBackend(command="cat", extra_args=[], permission_mode=None)
        with tempfile.TemporaryDirectory() as cwd:
            result = await backend.run("draft the reply", "## context block", cwd=cwd, timeout=10)
        self.assertTrue(result.ok)
        self.assertIn("draft the reply", result.output)
        self.assertIn("## context block", result.output)

    async def test_timeout_kills_and_reports(self):
        # `sleep 30` never produces output; the backend must kill it and report a
        # timeout rather than hang.
        backend = ClaudeCLIBackend(command="sleep", extra_args=["30"], permission_mode=None)
        with tempfile.TemporaryDirectory() as cwd:
            result = await backend.run("x", "", cwd=cwd, timeout=0.5)
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.error)

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    async def test_timeout_kills_the_whole_process_tree(self):
        # The real `claude` CLI spawns its own children (bash-tool commands, MCP
        # servers). A plain kill of the direct child would orphan those past the
        # timeout; the backend must tear down the whole process group. Stand in a
        # shell that forks a long-lived grandchild and records its pid, then assert
        # the grandchild is dead once the run times out.
        #
        # The grandchild redirects its own stdout/stderr away from the inherited
        # pipe so it does NOT keep the backend's `communicate()` open — that makes
        # the buggy (direct-kill-only) path fail *fast* here (the grandchild is
        # still alive when we check) instead of hanging until it exits on its own.
        child_pid = None
        try:
            with tempfile.TemporaryDirectory() as cwd:
                pidfile = os.path.join(cwd, "child.pid")
                script = (
                    f"sleep 30 >/dev/null 2>&1 & "
                    f"echo $! > {shlex.quote(pidfile)}; wait"
                )
                backend = ClaudeCLIBackend(
                    command="sh", extra_args=["-c", script], permission_mode=None
                )
                result = await backend.run("x", "", cwd=cwd, timeout=0.5)
                self.assertFalse(result.ok)
                self.assertIn("timed out", result.error)
                with open(pidfile) as fh:
                    child_pid = int(fh.read().strip())
            # Give the OS a moment to reap the SIGKILLed grandchild, then confirm it
            # is gone: os.kill(pid, 0) raises ProcessLookupError for a dead process.
            await asyncio.sleep(0.2)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            child_pid = None  # confirmed dead; nothing to clean up
        finally:
            # If the backend failed to tear the tree down (the bug), don't leave a
            # real orphan `sleep` lingering for 30s after the test.
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    async def test_nonzero_exit_is_a_failure(self):
        # `false` exits 1 with no output.
        backend = ClaudeCLIBackend(command="false", extra_args=[], permission_mode=None)
        with tempfile.TemporaryDirectory() as cwd:
            result = await backend.run("x", "", cwd=cwd, timeout=5)
        self.assertFalse(result.ok)
        self.assertTrue(result.error)

    def test_defaults_to_read_only_permission_mode(self):
        # Defence-in-depth: the shipped backend runs the CLI in a read/research-only
        # permission mode by default, so the tool layer itself refuses edits/Bash/
        # outward actions — the "produce a draft, never act" contract is enforced,
        # not merely requested in the prompt.
        argv = ClaudeCLIBackend()._build_argv()
        self.assertEqual(argv, ["claude", "-p", "--permission-mode", "plan"])

    def test_permission_mode_survives_custom_extra_args(self):
        # A deployment tightening `extra_args` must not accidentally drop the
        # safety posture: permission mode is a separate, first-class property.
        argv = ClaudeCLIBackend(extra_args=["--print", "--model", "haiku"])._build_argv()
        self.assertEqual(argv[-2:], ["--permission-mode", "plan"])

    def test_permission_mode_is_overridable_and_disablable(self):
        # An explicit mode is honoured; None (a fully-trusted, opted-out backend)
        # emits no permission flag at all.
        self.assertIn(
            "acceptEdits",
            ClaudeCLIBackend(permission_mode="acceptEdits")._build_argv(),
        )
        self.assertNotIn(
            "--permission-mode", ClaudeCLIBackend(permission_mode=None)._build_argv()
        )

    def test_permission_mode_env_override(self):
        with mock.patch.dict(os.environ, {"GUM_EXECUTOR_PERMISSION_MODE": ""}):
            self.assertNotIn(
                "--permission-mode", ClaudeCLIBackend()._build_argv()
            )
        with mock.patch.dict(os.environ, {"GUM_EXECUTOR_PERMISSION_MODE": "acceptEdits"}):
            self.assertEqual(
                ClaudeCLIBackend()._build_argv()[-2:],
                ["--permission-mode", "acceptEdits"],
            )


if __name__ == "__main__":
    unittest.main()
