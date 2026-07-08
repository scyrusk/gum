# api.py
#
# A small, localhost-only REST API so *any* local application (in any language)
# can build on what the GUM has learned. It is served inside the listening
# daemon and shares the same live `gum` instance, so reads always reflect the
# current model. It is strictly read-only and binds to 127.0.0.1.
#
# When launched with sanitize=True (`gum start --sanitize`), every response is
# pseudonymized on the way out: PII is replaced with consistent pseudo-IDs so
# downstream / off-device consumers (e.g. a frontier model behind the MCP) never
# see raw identities. This posture is fail-closed — if the sanitizer cannot load,
# the server refuses to start rather than serving raw data.

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .gum import gum
from .gumbo import Gumbo, Suggestion
from .models import FEEDBACK_RATINGS, Observation, Proposition

_STATIC_DIR = Path(__file__).parent / "static"


class ReviewIn(BaseModel):
    proposition_id: int
    rating: str  # one of FEEDBACK_RATINGS: accurate | partial | inaccurate
    note: str | None = None  # optional free-text context from the user


async def _scrub(text: str | None, sanitizer) -> str | None:
    """Pseudonymize *text* when a sanitizer is active, else return it unchanged."""
    if sanitizer is None or not text:
        return text
    return await asyncio.to_thread(sanitizer.sanitize, text)


async def _serialize_proposition(prop: Proposition, sanitizer, score: float | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": prop.id,
        "text": await _scrub(prop.text, sanitizer),
        "reasoning": await _scrub(prop.reasoning, sanitizer),
        "confidence": prop.confidence,
        "decay": prop.decay,
        "created_at": prop.created_at.isoformat() if prop.created_at else None,
        "updated_at": prop.updated_at.isoformat() if prop.updated_at else None,
    }
    if score is not None:
        data["score"] = score
    return data


async def _serialize_suggestion(sug: Suggestion, sanitizer) -> dict[str, Any]:
    """Serialize a scored suggestion for the API.

    The mixed-initiative scores and derived fields are numeric and carry no PII,
    so they pass through unchanged; only the model-written text (which was
    generated from raw propositions) is pseudonymized when a sanitizer is active.
    """
    data = sug.to_dict()
    for field in ("title", "description", "rationale"):
        data[field] = await _scrub(data[field], sanitizer)
    return data


async def _serialize_observation(obs: Observation, sanitizer) -> dict[str, Any]:
    return {
        "id": obs.id,
        "observer_name": obs.observer_name,
        "content": await _scrub(obs.content, sanitizer),
        "content_type": obs.content_type,
        "created_at": obs.created_at.isoformat() if obs.created_at else None,
    }


def create_app(gum_instance: gum, *, sanitize: bool = False) -> FastAPI:
    """Build the FastAPI app backed by a live `gum` instance.

    When *sanitize* is True the sanitizer is loaded eagerly (fail-closed): if the
    model or its dependencies are missing, this raises so the server never comes up
    serving raw PII.
    """
    sanitizer = None
    if sanitize:
        from .sanitize import get_sanitizer
        sanitizer = get_sanitizer()
        sanitizer.load()

    app = FastAPI(
        title="GUM Local API",
        description="Read-only local interface to your General User Model.",
        version="1.0.0",
    )

    # GUMBO suggestion engine over the same live GUM. Cheap to construct and does
    # no I/O until asked, so one shared instance is reused across requests.
    gumbo = Gumbo(gum_instance)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "user": gum_instance.user_name, "sanitized": sanitizer is not None}

    @app.get("/query")
    async def query(
        q: str = Query("", description="Free-text query (BM25). Empty = recent."),
        limit: int = Query(10, ge=1, le=100),
        mode: str = Query("OR", description="OR | AND | PHRASE"),
    ) -> dict[str, Any]:
        results = await gum_instance.query(q, limit=limit, mode=mode)
        return {
            "query": q,
            "results": [await _serialize_proposition(p, sanitizer, score) for p, score in results],
        }

    @app.get("/recent")
    async def recent(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
        props = await gum_instance.recent(limit=limit)
        return {"results": [await _serialize_proposition(p, sanitizer) for p in props]}

    @app.get("/observations")
    async def observations(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
        obs = await gum_instance.recent_observations(limit=limit)
        return {"results": [await _serialize_observation(o, sanitizer) for o in obs]}

    # ── GUMBO proactive suggestions ───────────────────────────────────────
    @app.get("/suggestions")
    async def suggestions(
        focus: str = Query(
            "", description="Optional topic (e.g. a project tab) to focus suggestions on."
        ),
        surfaced_only: bool = Query(
            False,
            description="If true, return only suggestions the mixed-initiative filter "
            "would surface (expected utility of interrupting > staying quiet).",
        ),
        limit: int = Query(10, ge=1, le=50),
    ) -> dict[str, Any]:
        results = await gumbo.generate(focus=focus.strip() or None)
        if surfaced_only:
            results = [s for s in results if s.should_surface]
        results = results[:limit]
        return {
            "focus": focus,
            "suggestions": [await _serialize_suggestion(s, sanitizer) for s in results],
        }

    # ── proposition review ────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def review_page() -> str:
        return (_STATIC_DIR / "review.html").read_text()

    @app.get("/review/next")
    async def review_next(
        skip: str = Query("", description="Comma-separated proposition ids to defer"),
    ) -> dict[str, Any]:
        exclude = {int(x) for x in skip.split(",") if x.strip().isdigit()}
        total, reviewed = await gum_instance.review_progress()
        result = await gum_instance.next_for_review(exclude_ids=exclude or None)
        if result is None:
            return {"done": True, "total": total, "reviewed": reviewed}
        prop, obs = result
        # The review UI is the user's own local tool for judging whether
        # propositions about them are accurate, so it always shows raw content
        # (sanitizer=None) — pseudonymized text would defeat the review — even when
        # the machine-facing endpoints above are sanitized under `--sanitize`.
        return {
            "done": False,
            "total": total,
            "reviewed": reviewed,
            "proposition": await _serialize_proposition(prop, None),
            "observations": [await _serialize_observation(o, None) for o in obs],
        }

    @app.post("/review")
    async def review_submit(body: ReviewIn) -> dict[str, Any]:
        if body.rating not in FEEDBACK_RATINGS:
            return {"ok": False, "error": f"rating must be one of {FEEDBACK_RATINGS}"}
        note = (body.note or "").strip() or None
        ok = await gum_instance.add_review(body.proposition_id, body.rating, note)
        return {"ok": ok}

    return app


def build_server(gum_instance: gum, host: str = "127.0.0.1", port: int = 8422, *, sanitize: bool = False):
    """Build a uvicorn Server for the API (caller awaits ``server.serve()``).

    Host defaults to 127.0.0.1 so the API is never exposed off-machine. When
    *sanitize* is True, all responses are pseudonymized (fail-closed at build).
    """
    import uvicorn

    config = uvicorn.Config(
        create_app(gum_instance, sanitize=sanitize),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(config)
