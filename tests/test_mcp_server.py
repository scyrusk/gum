# test_mcp_server.py
#
# Stdlib-only (unittest) tests for the MCP server that exposes the GUM to a local
# executing agent (paper: the "gumcp"). Runnable without pytest or a live model:
#     python -m unittest tests.test_mcp_server
#
# These drive the FastMCP tools end-to-end against a real temp database and assert
# both what an agent would receive (the structured result) and the egress
# sanitization contract that keeps raw PII off-device.

from __future__ import annotations

import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from unittest import mock

from mcp.shared.memory import create_connected_server_and_client_session

from gum import gum as Gum
from gum.models import Observation, Proposition
from gum.mcp_server import build_mcp, _focus_terms
from gum.schemas import CommitmentItem, CommitmentSchema


def _prop(text: str, confidence: int) -> Proposition:
    return Proposition(
        text=text,
        reasoning=f"because of {text}",
        confidence=confidence,
        decay=5,
        revision_group=uuid.uuid4().hex,
        version=1,
    )


def _obs(content: str) -> Observation:
    return Observation(
        observer_name="Screen",
        content=content,
        content_type="input_text",
    )


class _FakeSanitizer:
    """Deterministic stand-in for the PII model, so tests need no torch/transformers."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def sanitize(self, text: str) -> str:
        return self.sanitize_map(text)[0]

    def sanitize_map(self, text: str) -> tuple[str, dict[str, str]]:
        aliases: dict[str, str] = {}
        for raw, pseudo in self._mapping.items():
            if raw in text:
                aliases[raw] = pseudo
            text = text.replace(raw, pseudo)
        return text, aliases


async def _call(mcp, name: str, args: dict) -> dict:
    """Invoke a FastMCP tool and return its structured (dict) result."""
    _blocks, structured = await mcp.call_tool(name, args)
    return structured


class _Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.gum = Gum("Omar", "dummy-model", data_directory=self._tmp.name, db_name="test.db")
        await self.gum.connect_db()

    async def asyncTearDown(self):
        if self.gum.engine is not None:
            await self.gum.engine.dispose()
        self._tmp.cleanup()

    async def _seed(self, *props: Proposition) -> None:
        async with self.gum._session() as s:
            s.add_all(list(props))


class GatherContextTests(_Base):
    async def test_gather_returns_only_relevant_propositions(self):
        await self._seed(
            _prop("Omar is applying for a Schmidt Foundation research grant on privacy", 9),
            _prop("Omar prefers dark roast coffee in the morning", 6),
        )
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "gather_context", {"topic": "Schmidt grant proposal"})

        self.assertEqual(result["count"], 1)
        self.assertFalse(result["sanitized"])
        texts = [p["text"] for p in result["propositions"]]
        self.assertIn("Schmidt Foundation research grant", texts[0])
        # BM25 relevance ranking exposes a score for the agent to weigh.
        self.assertIn("score", result["propositions"][0])

    async def test_empty_topic_falls_back_to_recent(self):
        await self._seed(_prop("Omar drafted an outline today", 7))
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "gather_context", {"topic": "   "})

        self.assertEqual(result["topic"], "")
        self.assertEqual(result["count"], 1)

    async def test_limit_is_clamped(self):
        await self._seed(*[_prop(f"Omar did task {i}", 5) for i in range(5)])
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "gather_context", {"topic": "task", "limit": 999})

        self.assertLessEqual(result["count"], 50)

    async def test_instruction_verbs_do_not_pollute_retrieval(self):
        # An agent passes a whole task instruction. The imperative verb "draft"
        # must not drag in an unrelated proposition just because the user also
        # happens to draft other things; retrieval runs on the substantive terms.
        await self._seed(
            _prop("Omar is applying for a Schmidt Foundation research grant", 9),
            _prop("Omar frequently drafts and sends emails every morning", 6),
        )
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(
            mcp,
            "gather_context",
            {"topic": "draft a grant proposal for the Schmidt Foundation"},
        )

        # The task instruction is reduced to its content words before searching.
        self.assertEqual(result["search_terms"], "grant proposal schmidt foundation")
        texts = [p["text"] for p in result["propositions"]]
        self.assertTrue(any("Schmidt Foundation" in t for t in texts))
        self.assertFalse(any("drafts and sends emails" in t for t in texts))

    async def test_all_stopword_topic_falls_back_to_raw_terms(self):
        # A topic that is nothing but stopwords/verbs must still search on
        # something rather than degrading to an empty (match-everything) query.
        self.assertEqual(_focus_terms("please help me write it"), "please help me write it")
        self.assertEqual(_focus_terms("Schmidt grant"), "schmidt grant")


class RecentContextTests(_Base):
    async def test_recent_returns_latest_propositions(self):
        await self._seed(
            _prop("Omar reviewed a paper", 6),
            _prop("Omar answered email", 5),
        )
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "recent_context", {"limit": 10})

        self.assertEqual(result["count"], 2)
        self.assertFalse(result["sanitized"])


class InspectPropositionTests(_Base):
    async def _seed_prop_with_evidence(self, prop: Proposition, *contents: str) -> int:
        async with self.gum._session() as s:
            for c in contents:
                prop.observations.add(_obs(c))
            s.add(prop)
            await s.flush()
            return prop.id

    async def test_inspect_returns_supporting_observations(self):
        # The provenance path: an agent finds a relevant proposition, then drills
        # into the raw evidence to ground its work.
        pid = await self._seed_prop_with_evidence(
            _prop("Omar studies privacy-preserving ML", 8),
            "Omar typed 'differential privacy budget' into a paper draft",
            "Omar opened a Schmidt Foundation grant portal",
        )
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "inspect_proposition", {"proposition_id": pid})

        self.assertTrue(result["found"])
        self.assertEqual(result["proposition"]["id"], pid)
        contents = [o["content"] for o in result["evidence"]]
        self.assertEqual(len(contents), 2)
        self.assertTrue(any("differential privacy" in c for c in contents))

    async def test_inspect_missing_proposition_reports_not_found(self):
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "inspect_proposition", {"proposition_id": 424242})

        self.assertFalse(result["found"])
        self.assertEqual(result["evidence"], [])

    async def test_inspect_pseudonymizes_evidence(self):
        import gum.sanitize as sanitize_mod

        fake = _FakeSanitizer({"Schmidt": "[ORG_1]", "Omar": "[PERSON_1]"})
        original = sanitize_mod.get_sanitizer
        sanitize_mod.get_sanitizer = lambda: fake
        try:
            pid = await self._seed_prop_with_evidence(
                _prop("Omar studies privacy", 8),
                "Omar opened a Schmidt Foundation grant portal",
            )
            mcp = build_mcp(self.gum, sanitize=True)
            result = await _call(mcp, "inspect_proposition", {"proposition_id": pid})
        finally:
            sanitize_mod.get_sanitizer = original

        self.assertTrue(result["sanitized"])
        content = result["evidence"][0]["content"]
        self.assertIn("[ORG_1]", content)
        self.assertNotIn("Schmidt", content)
        self.assertNotIn("Omar", content)


class SanitizationTests(_Base):
    async def test_pii_is_pseudonymized_on_egress(self):
        # The whole point of the gumcp: an external agent must never see raw
        # identities. build_mcp(sanitize=True) loads the sanitizer fail-closed;
        # here we substitute a deterministic fake for it.
        import gum.sanitize as sanitize_mod

        fake = _FakeSanitizer({"Schmidt": "[ORG_1]", "Omar": "[PERSON_1]"})
        original = sanitize_mod.get_sanitizer
        sanitize_mod.get_sanitizer = lambda: fake
        try:
            await self._seed(
                _prop("Omar is applying for a Schmidt Foundation grant", 9),
            )
            mcp = build_mcp(self.gum, sanitize=True)
            result = await _call(mcp, "gather_context", {"topic": "Schmidt grant"})
        finally:
            sanitize_mod.get_sanitizer = original

        self.assertTrue(fake.loaded)  # loaded eagerly (fail-closed) at build time
        self.assertTrue(result["sanitized"])
        text = result["propositions"][0]["text"]
        self.assertIn("[ORG_1]", text)
        self.assertIn("[PERSON_1]", text)
        self.assertNotIn("Schmidt", text)
        self.assertNotIn("Omar", text)

    async def test_query_aliases_bridge_task_terms_to_pseudo_ids(self):
        # The returned propositions are pseudonymized, so "Schmidt" from the
        # agent's task shows up as "[ORG_1]" in the context. gather_context echoes
        # a query_aliases map (real -> pseudo) for the entities *in the topic* so
        # the agent can tell which pseudonymized propositions concern it.
        import gum.sanitize as sanitize_mod

        fake = _FakeSanitizer({"Schmidt": "[ORG_1]", "Omar": "[PERSON_1]"})
        original = sanitize_mod.get_sanitizer
        sanitize_mod.get_sanitizer = lambda: fake
        try:
            await self._seed(
                _prop("Omar is applying for a Schmidt Foundation grant", 9),
            )
            mcp = build_mcp(self.gum, sanitize=True)
            result = await _call(
                mcp, "gather_context", {"topic": "grant proposal for Schmidt"}
            )
        finally:
            sanitize_mod.get_sanitizer = original

        self.assertEqual(result["query_aliases"], {"Schmidt": "[ORG_1]"})
        # The alias must match how the entity appears in the returned context.
        self.assertIn("[ORG_1]", result["propositions"][0]["text"])

    async def test_query_aliases_absent_when_unsanitized(self):
        # With no sanitizer there are no pseudo-IDs to bridge; the map is empty.
        await self._seed(_prop("Omar is applying for a Schmidt grant", 9))
        mcp = build_mcp(self.gum, sanitize=False)

        result = await _call(mcp, "gather_context", {"topic": "Schmidt grant"})

        self.assertEqual(result["query_aliases"], {})


class AgendaTests(_Base):
    """The commitment & deadline radar exposed as the `agenda` MCP tool.

    build_agenda runs the text model to extract commitments; we patch
    ``structured_completion`` so these are deterministic and offline, mirroring
    the CLI agenda tests. A far-future due date keeps ``days_until_due`` positive
    regardless of the real "now" the tool uses.
    """

    def _fake_completion(self):
        async def fake(client, model, messages, schema, **kwargs):
            return CommitmentSchema(commitments=[
                CommitmentItem(source_index=1,
                               title="Submit the Schmidt grant proposal",
                               due_date="2999-07-20", source="Schmidt",
                               status_guess="in progress"),
                CommitmentItem(source_index=2, title="Send reviewer comments",
                               due_date=None, source="a colleague",
                               status_guess="not started"),
            ])
        return fake

    async def _seed_commitment_props(self) -> None:
        await self._seed(
            _prop("Omar has a grant proposal deadline for the Schmidt Foundation", 9),
            _prop("Omar promised to send reviewer comments back to a colleague", 7),
        )

    async def test_agenda_returns_ranked_commitments(self):
        await self._seed_commitment_props()
        mcp = build_mcp(self.gum, sanitize=False)

        with mock.patch("gum.agenda.structured_completion",
                        side_effect=self._fake_completion()):
            result = await _call(mcp, "agenda", {})

        self.assertEqual(result["count"], 2)
        self.assertFalse(result["sanitized"])
        self.assertIsNone(result["window_days"])
        titles = [c["title"] for c in result["commitments"]]
        self.assertIn("Submit the Schmidt grant proposal", titles)
        self.assertIn("Send reviewer comments", titles)
        # A dated commitment always outranks an undated one.
        self.assertEqual(result["commitments"][0]["title"],
                         "Submit the Schmidt grant proposal")
        self.assertEqual(result["commitments"][0]["due_date"], "2999-07-20")
        self.assertIsNone(result["commitments"][1]["due_date"])

    async def test_window_excludes_far_future_commitments(self):
        await self._seed_commitment_props()
        mcp = build_mcp(self.gum, sanitize=False)

        with mock.patch("gum.agenda.structured_completion",
                        side_effect=self._fake_completion()):
            result = await _call(mcp, "agenda", {"window_days": 30})

        # The year-2999 deadline is far outside a 30-day horizon; the undated
        # commitment has no date and is always kept.
        self.assertEqual(result["window_days"], 30)
        titles = [c["title"] for c in result["commitments"]]
        self.assertEqual(titles, ["Send reviewer comments"])

    async def test_limit_is_clamped(self):
        await self._seed_commitment_props()
        mcp = build_mcp(self.gum, sanitize=False)

        with mock.patch("gum.agenda.structured_completion",
                        side_effect=self._fake_completion()):
            result = await _call(mcp, "agenda", {"limit": 999})

        self.assertLessEqual(result["count"], 50)

    async def test_agenda_pseudonymizes_text_fields(self):
        # The radar's title/source/proposition_text are model-written from raw
        # propositions and carry PII; the MCP surface must scrub them, while the
        # numeric ranking fields pass through untouched.
        import gum.sanitize as sanitize_mod

        fake = _FakeSanitizer({"Schmidt": "[ORG_1]", "Omar": "[PERSON_1]"})
        original = sanitize_mod.get_sanitizer
        sanitize_mod.get_sanitizer = lambda: fake
        try:
            await self._seed_commitment_props()
            mcp = build_mcp(self.gum, sanitize=True)
            with mock.patch("gum.agenda.structured_completion",
                            side_effect=self._fake_completion()):
                result = await _call(mcp, "agenda", {})
        finally:
            sanitize_mod.get_sanitizer = original

        self.assertTrue(result["sanitized"])
        top = result["commitments"][0]
        self.assertEqual(top["title"], "Submit the [ORG_1] grant proposal")
        self.assertEqual(top["source"], "[ORG_1]")
        self.assertNotIn("Omar", top["proposition_text"])
        self.assertIn("[PERSON_1]", top["proposition_text"])
        # Numeric/date fields are not text and must not be mangled.
        self.assertEqual(top["due_date"], "2999-07-20")
        self.assertEqual(top["confidence"], 9)


class ToolAdvertisingTests(_Base):
    async def test_tools_are_advertised_to_clients(self):
        mcp = build_mcp(self.gum, sanitize=False)
        names = {t.name for t in await mcp.list_tools()}
        self.assertEqual(
            names,
            {"gather_context", "recent_context", "inspect_proposition", "agenda"},
        )


class WithUserContextPromptTests(_Base):
    async def test_prompt_is_advertised(self):
        mcp = build_mcp(self.gum, sanitize=False)
        prompts = await mcp.list_prompts()
        names = {p.name for p in prompts}
        self.assertIn("with_user_context", names)
        prompt = next(p for p in prompts if p.name == "with_user_context")
        # The task argument is required so a client can collect it from the user.
        arg_names = {a.name for a in (prompt.arguments or [])}
        self.assertIn("task", arg_names)

    async def test_prompt_expands_to_context_gathering_instruction(self):
        mcp = build_mcp(self.gum, sanitize=False)
        result = await mcp.get_prompt(
            "with_user_context",
            {"task": "draft a grant proposal for the Schmidt Foundation"},
        )
        self.assertEqual(len(result.messages), 1)
        msg = result.messages[0]
        self.assertEqual(msg.role, "user")
        text = msg.content.text
        # The task is threaded through and the workflow (gather -> inspect ->
        # execute) is spelled out so it triggers the tools even in clients that
        # do not surface the server's free-text instructions.
        self.assertIn("Schmidt Foundation", text)
        self.assertIn("gather_context", text)
        self.assertIn("inspect_proposition", text)

    async def test_prompt_closes_the_loop_with_rehydration(self):
        # The last mile of the motivating flow: a pseudonymized draft is unusable
        # to the user until `gum rehydrate` restores real names on-device, and the
        # agent must not guess those names itself. The prompt has to say so, or the
        # workflow dead-ends with [ORG_1]-laden text in the chat.
        mcp = build_mcp(self.gum, sanitize=False)
        result = await mcp.get_prompt(
            "with_user_context",
            {"task": "draft a grant proposal for the Schmidt Foundation"},
        )
        text = result.messages[0].content.text
        self.assertIn("gum rehydrate", text)
        # It must steer the agent away from inventing the real values.
        self.assertIn("guess", text.lower())


class ClientSessionE2ETests(_Base):
    """Drive the server the way a real MCP client (Claude/Codex) does.

    The other tests call ``mcp.call_tool`` in-process, which skips the actual
    client<->server protocol: the initialize handshake, the over-the-wire tool
    and prompt listings, structured-content serialization, and prompt expansion.
    Here we connect a real ``ClientSession`` to the server over in-memory streams
    so the whole plumbing an external agent relies on is exercised end-to-end.
    """

    async def test_agent_can_discover_and_call_over_the_protocol(self):
        await self._seed(
            _prop("Omar is applying for a Schmidt Foundation research grant on privacy", 9),
            _prop("Omar prefers dark roast coffee in the morning", 6),
        )
        mcp = build_mcp(self.gum, sanitize=False)

        async with create_connected_server_and_client_session(mcp) as client:
            init = await client.initialize()
            self.assertEqual(init.serverInfo.name, "gum-context")

            # Tools and the prompt are advertised across the wire.
            tools = {t.name for t in (await client.list_tools()).tools}
            self.assertEqual(
                tools,
                {"gather_context", "recent_context", "inspect_proposition", "agenda"},
            )
            prompts = {p.name for p in (await client.list_prompts()).prompts}
            self.assertIn("with_user_context", prompts)

            # gather_context returns structured content the agent can parse.
            called = await client.call_tool(
                "gather_context", {"topic": "draft a grant proposal for Schmidt"}
            )
            self.assertFalse(called.isError)
            self.assertEqual(called.structuredContent["count"], 1)
            self.assertFalse(called.structuredContent["sanitized"])
            prop = called.structuredContent["propositions"][0]
            self.assertIn("Schmidt Foundation", prop["text"])
            pid = prop["id"]

            # The provenance drill-down round-trips too (no evidence seeded).
            inspected = await client.call_tool(
                "inspect_proposition", {"proposition_id": pid}
            )
            self.assertTrue(inspected.structuredContent["found"])
            self.assertEqual(inspected.structuredContent["proposition"]["id"], pid)

            # The workflow prompt expands with the task threaded through.
            expanded = await client.get_prompt(
                "with_user_context",
                {"task": "draft a grant proposal for the Schmidt Foundation"},
            )
            text = expanded.messages[0].content.text
            self.assertIn("Schmidt Foundation", text)
            self.assertIn("gather_context", text)


if __name__ == "__main__":
    unittest.main()
