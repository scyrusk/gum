"""Offline tests for Sanitizer.sanitize_fragment (see gum/sanitize.py).

The privacy-filter NER model needs sentence context to recognize a name: a bare
"Jacob Willie Agnew" passes straight through un-pseudonymized, while the same name
inside a sentence is tagged and replaced. Short model-written fields — a
commitment's owner/source, a one-line title — arrive as exactly these bare
fragments, so scrubbing them verbatim leaks PII on a surface that reports itself
`sanitized`. sanitize_fragment lends the value a fixed carrier sentence, scrubs,
and strips the carrier back off. These tests pin that behavior with a fake pipe
that reproduces the model's context-dependent recall, so no real model is loaded.

Run (from the repo root, inside the venv):
    python -m unittest tests.test_sanitize_fragment
"""

import tempfile
import unittest
from pathlib import Path

from gum.sanitize import (
    _FRAGMENT_CARRIER_PREFIX,
    _FRAGMENT_CARRIER_SUFFIX,
    EntityMap,
    Sanitizer,
)

NEEDLE = "SECRETNAME"  # what our fake model "detects" as a PERSON span


class ContextSensitivePipe:
    """Fake pipeline that reproduces the real NER's context-dependent recall.

    It only reports a NEEDLE occurrence as a person span when the needle has
    non-space text before it in the window — i.e. it detects the name in a
    sentence ("Regarding SECRETNAME, ...") but NOT as a bare, leading fragment
    ("SECRETNAME"). This is exactly the failure sanitize_fragment exists to fix.
    """

    def _spans_for(self, window: str) -> list[dict]:
        spans, start = [], 0
        while True:
            i = window.find(NEEDLE, start)
            if i < 0:
                break
            if window[:i].strip():  # only detected when it has left-side context
                spans.append(
                    {"start": i, "end": i + len(NEEDLE),
                     "entity_group": "person", "score": 0.99}
                )
            start = i + len(NEEDLE)
        return spans

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, str):
            return self._spans_for(inputs)
        return [self._spans_for(w) for w in inputs]


def _sanitizer(tmp: Path) -> Sanitizer:
    s = Sanitizer(entity_map=EntityMap(db_path=str(tmp / "entities.db")))
    s._pipeline = ContextSensitivePipe()  # bypass _ensure_pipeline; no real model
    return s


class SanitizeFragmentTests(unittest.TestCase):
    def test_bare_name_leaks_via_plain_sanitize(self):
        # Establishes the bug the method fixes: a lone name is NOT caught by the
        # context-dependent model when scrubbed verbatim.
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            self.assertEqual(s.sanitize(NEEDLE), NEEDLE)  # leaks

    def test_bare_name_is_redacted_via_carrier(self):
        # The fix: the carrier gives the name sentence context, so it is tagged,
        # and the carrier is stripped back off leaving just the pseudo-ID.
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            out = s.sanitize_fragment(NEEDLE)
        self.assertNotIn(NEEDLE, out)
        self.assertEqual(out, "[PERSON_1]")

    def test_carrier_is_fully_stripped(self):
        # No trace of the carrier scaffolding survives in the output.
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            out = s.sanitize_fragment(f"meet {NEEDLE} soon")
        self.assertNotIn(_FRAGMENT_CARRIER_PREFIX.strip(), out)
        self.assertNotIn(_FRAGMENT_CARRIER_SUFFIX.strip(), out)
        self.assertEqual(out, "meet [PERSON_1] soon")

    def test_non_pii_fragment_passes_through_unchanged(self):
        # A fragment with no detected entity is returned verbatim (no carrier
        # residue, no spurious redaction).
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            for value in ("unknown", "the grant portal", "a colleague"):
                self.assertEqual(s.sanitize_fragment(value), value)

    def test_pseudo_id_is_consistent_across_calls(self):
        # The entity map keys on the entity, not the carrier, so the same name
        # yields the same pseudo-ID whether scrubbed as a fragment or a sentence.
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            frag = s.sanitize_fragment(NEEDLE)
            sentence = s.sanitize(f"I will meet {NEEDLE} tomorrow")
        self.assertEqual(frag, "[PERSON_1]")
        self.assertIn("[PERSON_1]", sentence)

    def test_empty_and_whitespace_are_returned_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            s = _sanitizer(Path(d))
            self.assertEqual(s.sanitize_fragment(""), "")
            self.assertEqual(s.sanitize_fragment("   "), "   ")


class _CarrierDropsTokenPipe:
    """Fake reproducing the *other* real failure mode: the carrier UNDER-detects.

    On a bare "Meet FULLNAME" it tags the whole name; but wrapped in the carrier
    it drops the last token, tagging only "FULLNAM". Defense in depth must prefer
    the bare pass's more-complete span so nothing leaks.
    """

    FULL = "FULLNAME"

    def _spans(self, window: str) -> list[dict]:
        target = self.FULL[:-1] if _FRAGMENT_CARRIER_PREFIX.strip() in window else self.FULL
        i = window.find(target)
        if i < 0:
            return []
        return [{"start": i, "end": i + len(target),
                 "entity_group": "person", "score": 0.99}]

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, str):
            return self._spans(inputs)
        return [self._spans(w) for w in inputs]


class DefenseInDepthTests(unittest.TestCase):
    def test_union_prefers_more_complete_bare_span(self):
        # Bare pass finds "FULLNAME"; carrier pass finds only "FULLNAM". The union,
        # redacting the longest span first, must leave no fragment of the name.
        with tempfile.TemporaryDirectory() as d:
            s = Sanitizer(entity_map=EntityMap(db_path=str(Path(d) / "e.db")))
            s._pipeline = _CarrierDropsTokenPipe()
            out = s.sanitize_fragment("Meet FULLNAME")
        self.assertNotIn("FULLNAME", out)
        self.assertNotIn("FULLNAM", out)
        self.assertEqual(out, "Meet [PERSON_1]")


if __name__ == "__main__":
    unittest.main()
