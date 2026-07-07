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
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = "~/.cache/gum/entities.db"
DEFAULT_MODEL = "openai/privacy-filter"
DEFAULT_MIN_SCORE = 0.5

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
        if not text:
            return text
        pipe = self._ensure_pipeline()
        with self._infer_lock:
            spans = pipe(text)

        # Keep confident spans, tagged with our category, ordered by position.
        kept = [
            (sp["start"], sp["end"], _category_for(sp.get("entity_group", "")))
            for sp in spans
            if float(sp.get("score", 0.0)) >= self._min_score
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
            pseudo = self._entities.pseudo_for(cat, text[start:end])
            text = text[:start] + pseudo + text[end:]
        return text


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
