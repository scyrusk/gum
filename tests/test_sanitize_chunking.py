"""Offline tests for the sanitizer's bounded-window inference (see gum/sanitize.py).

A token-classification forward pass is O(seq_len^2) in memory, and the privacy-filter
pipeline does not truncate. Feeding a whole ~50k-char screen observation in one call
ballooned the daily-summary process to tens of GB. sanitize() now runs the model in
windows capped at MAX_PIPE_CHARS and stitches the detected spans back by a global
offset. These tests pin that behavior without loading the real transformers model.

Run (from the repo root, inside the venv):
    python -m unittest tests.test_sanitize_chunking
"""

import tempfile
import unittest
from pathlib import Path

from gum.sanitize import MAX_PIPE_CHARS, EntityMap, Sanitizer, _iter_windows

NEEDLE = "SECRETNAME"  # what our fake model "detects" as a PERSON span


class FakePipe:
    """Stand-in for the transformers token-classification pipeline.

    Accepts a list of windows (the batched call sanitize() makes) and returns one
    span-list per window, reporting each occurrence of NEEDLE as a high-confidence
    person span with window-local char offsets — exactly like the real pipeline's
    aggregation_strategy='simple' output.
    """

    def __init__(self):
        self.windows: list[str] = []

    def _spans_for(self, window: str) -> list[dict]:
        self.windows.append(window)
        spans, start = [], 0
        while True:
            i = window.find(NEEDLE, start)
            if i < 0:
                break
            spans.append(
                {"start": i, "end": i + len(NEEDLE), "entity_group": "person", "score": 0.99}
            )
            start = i + len(NEEDLE)
        return spans

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, str):  # tolerate a bare string too
            return self._spans_for(inputs)
        return [self._spans_for(w) for w in inputs]

    @property
    def max_window(self) -> int:
        return max((len(w) for w in self.windows), default=0)


def _sanitizer_with(fake: FakePipe, tmp: Path) -> Sanitizer:
    s = Sanitizer(entity_map=EntityMap(db_path=str(tmp / "entities.db")))
    s._pipeline = fake  # bypass _ensure_pipeline so no real model is loaded
    return s


class IterWindowsTests(unittest.TestCase):
    def test_windows_are_bounded_and_lossless(self):
        text = ("word " * 8000).strip()  # ~40k chars, no line breaks
        windows = list(_iter_windows(text, MAX_PIPE_CHARS))
        self.assertTrue(len(windows) > 1, "long text should split into many windows")
        for _, w in windows:
            self.assertLessEqual(len(w), MAX_PIPE_CHARS)
        # Windows tile the text exactly — no character is dropped or duplicated.
        self.assertEqual("".join(w for _, w in windows), text)
        # Offsets are contiguous and correct.
        for base, w in windows:
            self.assertEqual(text[base : base + len(w)], w)

    def test_whitespaceless_run_is_hard_cut(self):
        text = "x" * (MAX_PIPE_CHARS * 3 + 7)  # no break opportunity anywhere
        windows = list(_iter_windows(text, MAX_PIPE_CHARS))
        for _, w in windows:
            self.assertLessEqual(len(w), MAX_PIPE_CHARS)
        self.assertEqual("".join(w for _, w in windows), text)


class SanitizeChunkingTests(unittest.TestCase):
    def test_no_window_exceeds_budget_on_huge_input(self):
        with tempfile.TemporaryDirectory() as d:
            fake = FakePipe()
            s = _sanitizer_with(fake, Path(d))
            # ~50k chars, the size that used to blow up the forward pass.
            s.sanitize(("lorem ipsum " * 4200))
            self.assertGreater(len(fake.windows), 1)
            self.assertLessEqual(fake.max_window, MAX_PIPE_CHARS)

    def test_pii_in_tail_is_redacted_via_offset_stitching(self):
        with tempfile.TemporaryDirectory() as d:
            fake = FakePipe()
            s = _sanitizer_with(fake, Path(d))
            # Put the PII far past the first window so only correct global-offset
            # stitching can redact it.
            text = ("filler " * 6000) + NEEDLE + " tail"
            self.assertGreater(len(text), MAX_PIPE_CHARS * 3)
            out = s.sanitize(text)
            self.assertNotIn(NEEDLE, out)
            self.assertIn("[PERSON_", out)
            # Everything else is preserved and nothing else was clobbered.
            self.assertTrue(out.endswith(" tail"))

    def test_multiple_spans_across_windows_all_redacted(self):
        with tempfile.TemporaryDirectory() as d:
            fake = FakePipe()
            s = _sanitizer_with(fake, Path(d))
            # Three occurrences spread across distinct windows.
            text = NEEDLE + (" pad " * 4000) + NEEDLE + (" pad " * 4000) + NEEDLE
            out = s.sanitize(text)
            self.assertNotIn(NEEDLE, out)
            self.assertEqual(out.count("[PERSON_1]"), 3)  # same entity → same pseudo-ID

    def test_short_text_is_single_window_unchanged_behavior(self):
        with tempfile.TemporaryDirectory() as d:
            fake = FakePipe()
            s = _sanitizer_with(fake, Path(d))
            out = s.sanitize(f"hello {NEEDLE} world")
            self.assertEqual(len(fake.windows), 1)
            self.assertNotIn(NEEDLE, out)
            self.assertEqual(out, "hello [PERSON_1] world")


if __name__ == "__main__":
    unittest.main()
