from __future__ import annotations

import re
import unittest

from gum.prompts.screen import COMBINED_PROMPT, SUMMARY_PROMPT


class TemporalObservationPromptTests(unittest.TestCase):
    """The screen observation prompts feed the (now date-aware) proposition
    model. If the action summary never surfaces on-screen dates/deadlines, an
    "due Friday" or a calendar entry is dropped at the observation stage — before
    PROPOSE/REVISE can ground it into an absolute YYYY-MM-DD deadline. These
    guards keep the temporal-signal instruction in both summary prompts.
    """

    _DEADLINE_RE = re.compile(
        r"dates,?\s+times,?\s+deadlines,?\s+or\s+scheduled\s+events", re.IGNORECASE
    )

    def test_combined_prompt_surfaces_deadlines(self):
        # The summary section (not just the verbatim transcription) must call out
        # temporal signals, because the transcription is scoped to the current
        # frame while the summary spans every frame.
        self.assertRegex(COMBINED_PROMPT, self._DEADLINE_RE)
        self.assertIn("## Summary", COMBINED_PROMPT)
        # The instruction must live in the summary section, after the heading.
        summary_section = COMBINED_PROMPT.split("## Summary", 1)[1]
        self.assertRegex(summary_section, self._DEADLINE_RE)
        # Quoting the exact on-screen wording keeps relative phrasing ("due
        # Friday") intact for the downstream model to resolve.
        self.assertIn("quote the exact wording", COMBINED_PROMPT)

    def test_summary_prompt_surfaces_deadlines(self):
        # The legacy two-call path (GUM_COMBINE_VISION=0) uses SUMMARY_PROMPT and
        # must carry the same temporal-signal instruction.
        self.assertRegex(SUMMARY_PROMPT, self._DEADLINE_RE)
        self.assertIn("quote the exact wording", SUMMARY_PROMPT)


if __name__ == "__main__":
    unittest.main()
