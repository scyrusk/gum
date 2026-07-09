# sanitize.py
#
# Local, egress-time PII sanitization for GUM output. A local token-classification
# model (openai/privacy-filter) detects PII spans, and each span is replaced with a
# *consistent* pseudo-ID (e.g. [PERSON_1]) so downstream frontier models can still
# reason about "the same person X" across observations and propositions without ever
# seeing the real identity.
#
# Nothing here touches the primary gum.db: raw data stays verbatim on disk and is
# only pseudonymized on the way out (CLI export / REST responses). The entity <->
# pseudo-ID map (the re-identification key) lives in its own sqlite file so it can be
# locked down or excluded from any export.

from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = "~/.cache/gum/entities.db"
DEFAULT_MODEL = "openai/privacy-filter"
DEFAULT_MIN_SCORE = 0.5

# A token-classification forward pass costs O(seq_len^2) in activation memory, and
# the privacy-filter pipeline does NOT truncate on its own (and truncation would be
# unacceptable here — it silently drops, and thus leaks, any PII past the cutoff).
# A single ~50k-char screen observation is ~12k tokens and ballooned the process to
# tens of GB. So sanitize() feeds the model in windows kept well under its 512-token
# limit (~4 chars/token) and stitches the detected spans back by a global offset.
# Tunable: smaller windows = flatter memory but more (cheap) forward passes.
MAX_PIPE_CHARS = 1200

# Windows are pushed through the pipeline this many at a time. Batching keeps a long
# observation from becoming dozens of serial forward passes, while the fixed batch
# size (rather than "all windows at once") keeps peak memory bounded regardless of
# how large the input is.
PIPE_BATCH_SIZE = 16

# privacy-filter emits labels for person, email, phone, address, account number,
# url, date, and secret. `aggregation_strategy="simple"` strips the BIOES prefixes,
# but the exact spelling of each group varies (e.g. "private_person", "phone_number"),
# so we map defensively: lowercase, split on separators, and match the first known
# token. Anything unrecognized falls back to ENTITY so a detected span is *never*
# left un-pseudonymized.
_CATEGORY_MAP = {
    "person": "PERSON",
    "name": "PERSON",
    "email": "EMAIL",
    "phone": "PHONE",
    "address": "ADDRESS",
    "account": "ACCOUNT",
    "url": "URL",
    "date": "DATE",
    "secret": "SECRET",
}


def _category_for(entity_group: str) -> str:
    """Map a privacy-filter entity group to our pseudo-ID category prefix."""
    key = (entity_group or "").strip().lower()
    for token in key.replace("-", "_").split("_"):
        if token in _CATEGORY_MAP:
            return _CATEGORY_MAP[token]
    return "ENTITY"


def _iter_windows(text: str, budget: int):
    """Yield ``(offset, window)`` pairs covering *text* in ≤ *budget*-char slices.

    Each window breaks on the last newline (else space) before the budget so a PII
    token is not split across a boundary; a run with no whitespace (e.g. one long
    token) is cut hard rather than allowed to overflow the model's context.
    """
    n = len(text)
    i = 0
    while i < n:
        end = min(i + budget, n)
        if end < n:
            brk = text.rfind("\n", i, end)
            if brk <= i:
                brk = text.rfind(" ", i, end)
            if brk > i:
                end = brk
        yield i, text[i:end]
        i = end


class EntityMap:
    """SQLite-backed, consistent entity → pseudo-ID map (the re-identification key).

    Keyed by (category, normalized surface form): the same real entity always
    resolves to the same pseudo-ID, across processes and across runs. Stored in its
    own file (default ``~/.cache/gum/entities.db``) so it stays isolated from the
    main GUM database that the CLI/API/MCP read.
    """

    def __init__(self, db_path: str | None = None):
        raw = db_path or os.getenv("GUM_SANITIZE_DB") or DEFAULT_DB_PATH
        self._path = Path(os.path.expanduser(raw))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False: the sanitizer runs inside asyncio worker threads;
        # every access is serialized by self._lock, so shared use is safe.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_map(
                category   TEXT NOT NULL,
                norm_text  TEXT NOT NULL,
                pseudo_id  TEXT NOT NULL,
                raw_text   TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (category, norm_text)
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _norm(text: str) -> str:
        """Collapse whitespace and casefold so surface variants share a pseudo-ID."""
        return " ".join(text.split()).casefold()

    def pseudo_for(self, category: str, raw_text: str) -> str:
        """Return the stable pseudo-ID for *raw_text*, minting one on first sight."""
        norm = self._norm(raw_text)
        if not norm:
            return raw_text
        with self._lock:
            row = self._conn.execute(
                "SELECT pseudo_id FROM entity_map WHERE category=? AND norm_text=?",
                (category, norm),
            ).fetchone()
            if row:
                return row[0]
            count = self._conn.execute(
                "SELECT COUNT(*) FROM entity_map WHERE category=?", (category,)
            ).fetchone()[0]
            pseudo = f"[{category}_{count + 1}]"
            self._conn.execute(
                "INSERT INTO entity_map(category, norm_text, pseudo_id, raw_text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (category, norm, pseudo, raw_text, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()
            return pseudo

    def raw_for(self, pseudo_id: str) -> str | None:
        """Return the original text a *pseudo_id* was minted for, or None.

        The inverse of :meth:`pseudo_for`: this reads the re-identification key so
        a fully-local, trusted step can turn ``[PERSON_1]`` back into the real
        name. Pseudo-IDs are globally unique (``[CATEGORY_N]``), so no category is
        needed to look one up.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT raw_text FROM entity_map WHERE pseudo_id=?",
                (pseudo_id,),
            ).fetchone()
        return row[0] if row else None


# A minted pseudo-ID is always ``[CATEGORY_N]`` with an uppercase category and a
# 1-based index (see EntityMap.pseudo_for). Matching that exact shape avoids
# touching unrelated bracketed text (e.g. Markdown ``[link]``) during rehydration.
_PSEUDO_ID_RE = re.compile(r"\[[A-Z]+_\d+\]")


def find_pseudo_ids(text: str) -> list[str]:
    """Return the distinct pseudo-IDs (``[CATEGORY_N]``) in *text*, first-seen order.

    Run on the *output* of :meth:`Sanitizer.rehydrate`, this surfaces the
    placeholders that could not be restored to a real value — either invented by
    a frontier model or absent from the local entity map — so a caller can warn
    the user that the finished artifact still carries them. Pseudo-IDs are not PII
    (they are the opaque stand-ins sanitization produced), so listing them is safe
    even on a channel a model could read.
    """
    seen: list[str] = []
    for match in _PSEUDO_ID_RE.finditer(text or ""):
        pid = match.group(0)
        if pid not in seen:
            seen.append(pid)
    return seen


class Sanitizer:
    """Detects PII spans with a local model and replaces them with pseudo-IDs.

    The model is loaded lazily on first use (or eagerly via :meth:`load`, which the
    API uses for fail-closed startup). Inference is CPU/GPU-bound and blocking;
    callers should invoke :meth:`sanitize` via ``asyncio.to_thread``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        min_score: float | None = None,
        entity_map: EntityMap | None = None,
    ):
        self._model_name = model or os.getenv("GUM_SANITIZE_MODEL") or DEFAULT_MODEL
        self._min_score = (
            min_score
            if min_score is not None
            else float(os.getenv("GUM_SANITIZE_MIN_SCORE", str(DEFAULT_MIN_SCORE)))
        )
        self._entities = entity_map or EntityMap()
        self._pipeline = None
        self._load_lock = threading.Lock()
        # HuggingFace pipelines are not safe to call concurrently on one instance;
        # serialize inference (it is fast — privacy-filter is ~50M active params).
        self._infer_lock = threading.Lock()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is None:
                try:
                    from transformers import pipeline
                except ImportError as exc:
                    raise RuntimeError(
                        "Sanitization requires the 'sanitize' extra. "
                        "Install it with: pip install 'gum-ai[sanitize]'"
                    ) from exc
                self._pipeline = pipeline(
                    "token-classification",
                    model=self._model_name,
                    aggregation_strategy="simple",
                )
        return self._pipeline

    def load(self) -> None:
        """Eagerly load the model so failures surface at startup, not first request."""
        self._ensure_pipeline()

    def sanitize(self, text: str) -> str:
        """Return *text* with every detected PII span replaced by a pseudo-ID."""
        return self.sanitize_map(text)[0]

    def sanitize_map(self, text: str) -> tuple[str, dict[str, str]]:
        """Sanitize *text* and also return the ``{raw_span: pseudo_id}`` map of what
        was replaced.

        Same output text as :meth:`sanitize`, plus the mapping of each detected
        real entity to the pseudo-ID it became. Callers use this to expose the
        pseudo-IDs for entities *they already supplied* (e.g. the terms in a
        search query) without leaking anything new — it reveals only mappings for
        text the caller passed in, never for entities elsewhere in the GUM.
        """
        aliases: dict[str, str] = {}
        if not text:
            return text, aliases
        pipe = self._ensure_pipeline()
        # Run inference in bounded windows (see MAX_PIPE_CHARS) so one oversized
        # observation can't blow up the O(seq_len^2) forward pass, and push the
        # windows through the pipeline in batches (PIPE_BATCH_SIZE) so a long
        # observation isn't dozens of serial forward passes. Short text is a single
        # window, so behavior is unchanged for the common case. Each span's offsets
        # are then lifted back into the full-text coordinate space.
        windows = list(_iter_windows(text, MAX_PIPE_CHARS))
        with self._infer_lock:
            results = list(pipe([w for _, w in windows], batch_size=PIPE_BATCH_SIZE))
        spans: list[tuple] = []
        for (base, _window), window_spans in zip(windows, results):
            for sp in window_spans:
                spans.append(
                    (
                        sp["start"] + base,
                        sp["end"] + base,
                        _category_for(sp.get("entity_group", "")),
                        float(sp.get("score", 0.0)),
                    )
                )

        # Keep confident spans, tagged with our category, ordered by position.
        kept = [
            (start, end, cat)
            for start, end, cat, score in spans
            if score >= self._min_score
        ]
        kept.sort(key=lambda x: x[0])

        # privacy-filter tags with BIOES, which HuggingFace's "simple" aggregation
        # does not fully coalesce — so a multi-token entity ("Alice Smith", a split
        # email/phone) comes back as several adjacent spans. Merge runs of the same
        # category separated only by whitespace so each real entity maps to ONE
        # consistent pseudo-ID instead of a garbled string of fragments.
        merged: list[list] = []
        for start, end, cat in kept:
            if merged and cat == merged[-1][2] and not text[merged[-1][1]:start].strip():
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end, cat])

        # Replace right-to-left so each replacement leaves earlier offsets valid.
        for start, end, cat in sorted(merged, key=lambda x: x[0], reverse=True):
            # The tokenizer often includes a leading/trailing space in a span; keep
            # that whitespace outside the pseudo-ID so surrounding text stays spaced.
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start >= end:
                continue
            raw = text[start:end]
            pseudo = self._entities.pseudo_for(cat, raw)
            aliases[raw] = pseudo
            text = text[:start] + pseudo + text[end:]
        return text, aliases

    def rehydrate(self, text: str) -> tuple[str, int]:
        """Replace every known pseudo-ID in *text* with its original value.

        The inverse of :meth:`sanitize`, and the final step of the
        sanitized-context workflow: an agent gathers pseudonymized context, a
        frontier model drafts an artifact still carrying ``[PERSON_1]`` /
        ``[ORG_1]`` placeholders, and this turns them back into real names so the
        *user* gets a usable document. It is pure DB lookup against the local
        entity map — no model is loaded — and must only be run in a trusted,
        on-device step (never fed back to a frontier model, or the PII the
        pseudonymization protected would leak).

        Pseudo-IDs with no entry in the map (e.g. one the model invented) are left
        verbatim. Returns ``(rehydrated_text, n_substitutions)``.
        """
        if not text:
            return text, 0
        count = 0

        def _restore(match: re.Match) -> str:
            nonlocal count
            raw = self._entities.raw_for(match.group(0))
            if raw is None:
                return match.group(0)
            count += 1
            return raw

        return _PSEUDO_ID_RE.sub(_restore, text), count


_SINGLETON: Sanitizer | None = None
_SINGLETON_LOCK = threading.Lock()


def get_sanitizer() -> Sanitizer:
    """Return the process-wide sanitizer (one loaded model + one entities.db).

    Sharing a single instance across the CLI process or API server keeps pseudo-IDs
    consistent within a run and avoids reloading the model per call.
    """
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = Sanitizer()
    return _SINGLETON
