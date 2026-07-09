"""Tests for pseudo-ID rehydration (the inverse of egress sanitization).

The MCP server hands a local agent *pseudonymized* GUM context ([PERSON_1],
[ORG_1], …). A frontier model drafts an artifact that still carries those
placeholders; the final, on-device step swaps them back for real names so the
user gets a usable document. rehydrate() / EntityMap.raw_for() and the
`gum rehydrate` CLI implement that step. None of this loads the transformers
model — it is pure lookup against the local entity map.

Run (from the repo root, inside the venv):
    python -m unittest tests.test_rehydrate
"""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gum.sanitize import EntityMap, Sanitizer, _PSEUDO_ID_RE


def _sanitizer(tmp: Path) -> Sanitizer:
    # No pipeline is ever set: rehydrate() and pseudo_for() are pure DB lookups,
    # so a real model is never loaded here.
    return Sanitizer(entity_map=EntityMap(db_path=str(tmp / "entities.db")))


class EntityMapReverseTests(unittest.TestCase):
    def test_raw_for_returns_the_minted_original(self):
        with tempfile.TemporaryDirectory() as d:
            em = EntityMap(db_path=str(Path(d) / "e.db"))
            pseudo = em.pseudo_for("ORG", "Schmidt Foundation")
            self.assertEqual(em.raw_for(pseudo), "Schmidt Foundation")

    def test_raw_for_unknown_pseudo_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            em = EntityMap(db_path=str(Path(d) / "e.db"))
            self.assertIsNone(em.raw_for("[PERSON_99]"))


class RehydrateTests(unittest.TestCase):
    def test_round_trip_restores_every_entity(self):
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            p_org = s._entities.pseudo_for("ORG", "Schmidt Foundation")
            p_person = s._entities.pseudo_for("PERSON", "Omar Khan")
            draft = (
                f"Dear {p_person}, we are applying to the {p_org} for funding "
                f"and look forward to partnering with {p_person}."
            )
            restored, n = s.rehydrate(draft)
            self.assertNotIn("[", restored)
            self.assertIn("Schmidt Foundation", restored)
            self.assertIn("Omar Khan", restored)
            # Both occurrences of the repeated pseudo-ID count.
            self.assertEqual(n, 3)

    def test_unknown_pseudo_ids_are_left_verbatim(self):
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            known = s._entities.pseudo_for("PERSON", "Ada")
            text = f"{known} met [PERSON_404] at the lab."
            restored, n = s.rehydrate(text)
            self.assertEqual(restored, "Ada met [PERSON_404] at the lab.")
            self.assertEqual(n, 1)

    def test_markdown_links_are_not_treated_as_pseudo_ids(self):
        # The regex only matches [UPPER_<digits>], so ordinary bracketed text is
        # untouched even when no entity map entry exists.
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            text = "See [the docs](http://x) and [TODO] items."
            restored, n = s.rehydrate(text)
            self.assertEqual(restored, text)
            self.assertEqual(n, 0)

    def test_empty_text(self):
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            self.assertEqual(s.rehydrate(""), ("", 0))

    def test_regex_shape(self):
        self.assertTrue(_PSEUDO_ID_RE.fullmatch("[PERSON_1]"))
        self.assertTrue(_PSEUDO_ID_RE.fullmatch("[ORG_12]"))
        self.assertIsNone(_PSEUDO_ID_RE.fullmatch("[person_1]"))
        self.assertIsNone(_PSEUDO_ID_RE.fullmatch("[PERSON]"))
        self.assertIsNone(_PSEUDO_ID_RE.fullmatch("[link]"))


class RehydrateCliTests(unittest.TestCase):
    """Exercise cmd_rehydrate end-to-end against a real entity map."""

    def _seed(self, tmp: Path) -> Sanitizer:
        s = _sanitizer(tmp)
        s._entities.pseudo_for("ORG", "Schmidt Foundation")  # -> [ORG_1]
        s._entities.pseudo_for("PERSON", "Omar Khan")  # -> [PERSON_1]
        return s

    def test_in_place_file_rehydration(self):
        from gum.cli import cmd_rehydrate

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._seed(tmp)
            draft = tmp / "draft.md"
            draft.write_text("Proposal to [ORG_1], drafted by [PERSON_1].")

            args = SimpleNamespace(input=str(draft), output=None)
            with mock.patch("gum.sanitize.get_sanitizer", lambda: _sanitizer(tmp)):
                err = io.StringIO()
                with redirect_stderr(err):
                    cmd_rehydrate(args)

            self.assertEqual(
                draft.read_text(), "Proposal to Schmidt Foundation, drafted by Omar Khan."
            )
            # Status goes to stderr and reports only a count — never the PII.
            self.assertIn("Rehydrated 2", err.getvalue())
            self.assertNotIn("Schmidt", err.getvalue())

    def test_stdin_to_stdout(self):
        from gum.cli import cmd_rehydrate

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._seed(tmp)
            args = SimpleNamespace(input=None, output=None)
            out, err = io.StringIO(), io.StringIO()
            with mock.patch("gum.sanitize.get_sanitizer", lambda: _sanitizer(tmp)):
                with mock.patch("sys.stdin", io.StringIO("Hi [PERSON_1]")):
                    with redirect_stdout(out), redirect_stderr(err):
                        cmd_rehydrate(args)
            self.assertEqual(out.getvalue(), "Hi Omar Khan")


if __name__ == "__main__":
    unittest.main()
