"""Offline tests for date preservation in the sanitizer (see gum/sanitize.py).

privacy-filter tags calendar dates with a `date` label, and pseudonymizing them
to [DATE_n] destroys the one signal the deadline pipeline exists to surface: the
propositions carry absolute YYYY-MM-DD deadlines, and an off-device agent building
the user's daily agenda from the *sanitized* output would see "[DATE_5]" instead
of "2026-07-20" and be unable to reason about what is due when. So the sanitizer
PRESERVES dates by default (a bare calendar date is not a re-identifier), and only
pseudonymizes them when GUM_SANITIZE_REDACT_DATES is set / redact_dates=True.

These tests pin that behavior with a fake pipe that tags one PERSON span and one
DATE span, so no real model is loaded.

Run (from the repo root, inside the venv):
    python -m unittest tests.test_sanitize_dates
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gum.sanitize import EntityMap, Sanitizer

PERSON = "SECRETNAME"  # fake model tags this as a person
DATE = "2026-07-20"  # fake model tags this as a date


class PersonAndDatePipe:
    """Fake pipeline tagging PERSON as `person` and DATE as `date` wherever seen."""

    def _spans_for(self, window: str) -> list[dict]:
        spans = []
        for needle, group in ((PERSON, "person"), (DATE, "private_date")):
            start = 0
            while True:
                i = window.find(needle, start)
                if i < 0:
                    break
                spans.append(
                    {"start": i, "end": i + len(needle),
                     "entity_group": group, "score": 0.99}
                )
                start = i + len(needle)
        return sorted(spans, key=lambda s: s["start"])

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, str):
            return self._spans_for(inputs)
        return [self._spans_for(w) for w in inputs]


def _sanitizer(tmp: Path, **kw) -> Sanitizer:
    s = Sanitizer(entity_map=EntityMap(db_path=str(tmp / "entities.db")), **kw)
    s._pipeline = PersonAndDatePipe()  # bypass _ensure_pipeline; no real model
    return s


TEXT = f"{PERSON} must submit the grant by {DATE}."


class DatePreservationTests(unittest.TestCase):
    def test_dates_preserved_by_default_while_person_redacted(self):
        # The core fix: the calendar date survives verbatim (deadline signal) while
        # the name is still pseudonymized.
        with tempfile.TemporaryDirectory() as d:
            out = _sanitizer(Path(d)).sanitize(TEXT)
        self.assertIn(DATE, out)
        self.assertNotIn(PERSON, out)
        self.assertNotIn("[DATE_", out)
        self.assertIn("[PERSON_1]", out)

    def test_redact_dates_flag_pseudonymizes_the_date(self):
        # Opt-in restores the old behavior: the date becomes a pseudo-ID.
        with tempfile.TemporaryDirectory() as d:
            out = _sanitizer(Path(d), redact_dates=True).sanitize(TEXT)
        self.assertNotIn(DATE, out)
        self.assertIn("[DATE_1]", out)
        self.assertNotIn(PERSON, out)

    def test_env_var_enables_date_redaction(self):
        # GUM_SANITIZE_REDACT_DATES=1 is equivalent to redact_dates=True.
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"GUM_SANITIZE_REDACT_DATES": "1"}):
                s = Sanitizer(entity_map=EntityMap(db_path=str(Path(d) / "e.db")))
            s._pipeline = PersonAndDatePipe()
            out = s.sanitize(TEXT)
        self.assertIn("[DATE_1]", out)

    def test_env_var_default_off_preserves_dates(self):
        # With the env var unset (cleared), dates are preserved.
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GUM_SANITIZE_REDACT_DATES", None)
                s = Sanitizer(entity_map=EntityMap(db_path=str(Path(d) / "e.db")))
            s._pipeline = PersonAndDatePipe()
            out = s.sanitize(TEXT)
        self.assertIn(DATE, out)
        self.assertNotIn("[DATE_", out)

    def test_sanitize_map_aliases_exclude_preserved_date(self):
        # The returned alias map reflects only what was actually replaced, so a
        # preserved date does not appear as a mapping.
        with tempfile.TemporaryDirectory() as d:
            _, aliases = _sanitizer(Path(d)).sanitize_map(TEXT)
        self.assertNotIn(DATE, aliases)
        self.assertIn(PERSON, aliases)

    def test_fragment_path_also_preserves_dates(self):
        # sanitize_fragment builds on sanitize_map, so it inherits date
        # preservation — a terse "due 2026-07-20" field keeps its date.
        with tempfile.TemporaryDirectory() as d:
            out = _sanitizer(Path(d)).sanitize_fragment(f"due {DATE}")
        self.assertEqual(out, f"due {DATE}")


if __name__ == "__main__":
    unittest.main()
