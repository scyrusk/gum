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


class SuggestionFeedbackIn(BaseModel):
    title: str  # the suggestion the user reacted to (for the observation text)
    vote: str  # "up" | "down"
    description: str | None = None  # suggestion body, for richer context
    focus: str | None = None  # active project tab, if any


class PropositionEditIn(BaseModel):
    # All fields optional: the user edits whichever they want. text is the
    # proposition statement, reasoning the justification, confidence the 1-10 pill.
    text: str | None = None
    reasoning: str | None = None
    confidence: int | None = None


class AgendaEditIn(BaseModel):
    # A user's edit to a generated agenda item (GUMBO Agenda page). All fields
    # optional: only the ones set are applied. `clear_due_date` explicitly marks
    # the item as having no fixed date (distinct from "date not overridden").
    title: str | None = None
    due_date: str | None = None  # ISO YYYY-MM-DD
    status: str | None = None
    clear_due_date: bool = False


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class SuggestionChatIn(BaseModel):
    messages: list[ChatMessage]  # running conversation (user/assistant turns)
    suggestion: dict[str, str] | None = None  # {title, description, rationale} in scope
    focus: str | None = None  # active project tab, if any


async def _scrub(text: str | None, sanitizer) -> str | None:
    """Pseudonymize *text* when a sanitizer is active, else return it unchanged."""
    if sanitizer is None or not text:
        return text
    return await asyncio.to_thread(sanitizer.sanitize, text)


async def _scrub_fragment(text: str | None, sanitizer) -> str | None:
    """Pseudonymize a short, context-free field (a bare name, a terse title).

    Uses the sanitizer's carrier-context path (see
    :meth:`gum.sanitize.Sanitizer.sanitize_fragment`), which the NER model needs
    to reliably tag a lone name that a plain :func:`_scrub` would leak.
    """
    if sanitizer is None or not text:
        return text
    return await asyncio.to_thread(sanitizer.sanitize_fragment, text)


async def _serialize_proposition(
    prop: Proposition,
    sanitizer,
    score: float | None = None,
    *,
    include_support: bool = False,
) -> dict[str, Any]:
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
    if include_support:
        # "Support" (paper Fig 3B) = how many observations back this proposition.
        # Requires the observations relationship to have been eager-loaded by the
        # caller (selectin), so this stays a cheap len() with no extra I/O.
        data["support"] = len(prop.observations)
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


async def _serialize_commitment(commitment, sanitizer) -> dict[str, Any]:
    """Serialize a ranked commitment for the machine surfaces (REST/MCP).

    Only the model-written text fields (``title``, ``source``,
    ``proposition_text``) are generated from raw propositions and carry PII, so
    they pass through the sanitizer; the numeric/date/ranking fields never do and
    are emitted unchanged. Mirrors :func:`_serialize_suggestion`.

    ``title`` and ``source`` are terse, context-free fragments (often a bare
    name), which the NER model under-detects — so they go through the
    carrier-context :func:`_scrub_fragment` rather than the plain :func:`_scrub`
    used for the full-sentence ``proposition_text``.
    """
    data = commitment.to_dict()
    for field in ("title", "source"):
        data[field] = await _scrub_fragment(data[field], sanitizer)
    data["proposition_text"] = await _scrub(data["proposition_text"], sanitizer)
    # The GUMBO Agenda page can edit items with known provenance: a
    # proposition-backed commitment (via its override key) or an explicitly-added
    # item (via its item_id). An item the model couldn't attribute to either has
    # no handle and is read-only.
    data["editable"] = (
        commitment.proposition_id is not None or commitment.item_id is not None
    )
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

    # ── memory (raw propositions) ─────────────────────────────────────────
    @app.get("/memory")
    async def memory(
        q: str = Query("", description="Optional search text; empty = most recent."),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        # The Memory page (paper Fig 3B) lets the user browse the raw
        # propositions in their GUM, each annotated with its "support" — the
        # number of observations backing it. Searching reuses the same BM25 query
        # the rest of the API uses; an empty query browses the most recent.
        if q.strip():
            results = await gum_instance.query(q.strip(), limit=limit)
            props = [p for p, _ in results]
        else:
            props = await gum_instance.recent(limit=limit, include_observations=True)
        return {
            "query": q,
            "propositions": [
                await _serialize_proposition(p, sanitizer, include_support=True)
                for p in props
            ],
        }

    @app.patch("/memory/{proposition_id}")
    async def memory_edit(
        proposition_id: int, body: PropositionEditIn
    ) -> dict[str, Any]:
        # Curate the model (paper Fig 3B): the user corrects a proposition that is
        # close-but-wrong instead of deleting it. Blank text/reasoning are rejected
        # (an empty proposition is meaningless); confidence is clamped to 1-10 to
        # match the model's own scale.
        text = body.text.strip() if body.text is not None else None
        reasoning = body.reasoning.strip() if body.reasoning is not None else None
        if text == "" or reasoning == "":
            return {"ok": False, "error": "text and reasoning cannot be blank"}
        confidence = body.confidence
        if confidence is not None:
            confidence = max(1, min(10, confidence))
        prop = await gum_instance.update_proposition(
            proposition_id, text=text, reasoning=reasoning, confidence=confidence
        )
        if prop is None:
            return {"ok": False, "error": "not found"}
        return {
            "ok": True,
            "proposition": await _serialize_proposition(prop, sanitizer),
        }

    @app.delete("/memory/{proposition_id}")
    async def memory_delete(proposition_id: int) -> dict[str, Any]:
        # Curate the model (paper Fig 3B): the user removes a proposition they
        # judge wrong or unwanted. This mutates the GUM, but it is the user acting
        # on their own model from their own machine — the same spirit as the
        # existing review/feedback write routes.
        ok = await gum_instance.delete_proposition(proposition_id)
        return {"ok": ok}

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
        rate_limited: bool = Query(
            False,
            description="If true, additionally apply the paper's token-bucket rate "
            "limit (~1 surfaced suggestion per minute) on top of the mixed-initiative "
            "filter. Implies surfaced_only. State is shared across requests.",
        ),
        limit: int = Query(10, ge=1, le=50),
    ) -> dict[str, Any]:
        if rate_limited:
            # surface() already filters to should_surface and applies the bucket.
            results = await gumbo.surface(focus=focus.strip() or None)
        else:
            results = await gumbo.generate(focus=focus.strip() or None)
            if surfaced_only:
                results = [s for s in results if s.should_surface]
        results = results[:limit]
        return {
            "focus": focus,
            "suggestions": [await _serialize_suggestion(s, sanitizer) for s in results],
        }

    @app.post("/suggestions/feedback")
    async def suggestions_feedback(body: SuggestionFeedbackIn) -> dict[str, Any]:
        # Thumbs up/down on a proactive suggestion (paper §4.3). The reaction is
        # fed back into the GUM as an observation so future suggestions reflect
        # what the user actually finds useful — closing the mixed-initiative loop.
        if body.vote not in ("up", "down"):
            return {"ok": False, "error": "vote must be 'up' or 'down'"}
        ok = await gum_instance.add_suggestion_feedback(
            title=body.title,
            vote=body.vote,
            description=body.description,
            focus=body.focus,
        )
        return {"ok": ok}

    @app.post("/suggestions/chat")
    async def suggestions_chat(body: SuggestionChatIn) -> dict[str, Any]:
        # "Start Chat" (paper §4.3.3): talk to GUMBO in more detail about a
        # surfaced suggestion. The conversation is grounded in the user's
        # high-confidence propositions via the local text model. The reply is
        # model-written prose over raw propositions, so it is pseudonymized under
        # --sanitize just like the suggestion text.
        turns = [{"role": m.role, "content": m.content} for m in body.messages]
        if not any(t["role"] == "user" and t["content"].strip() for t in turns):
            return {"ok": False, "error": "at least one user message is required"}
        try:
            reply = await gumbo.chat(
                turns, suggestion=body.suggestion, focus=body.focus
            )
        except Exception as exc:  # local model unavailable / transport error
            return {"ok": False, "error": f"chat failed: {exc}"}
        return {"ok": True, "reply": await _scrub(reply, sanitizer)}

    # ── commitment & deadline radar ───────────────────────────────────────
    @app.get("/agenda")
    async def agenda(
        limit: int = Query(10, ge=1, le=50),
        window_days: int | None = Query(
            None,
            ge=0,
            description="Optional horizon: only keep commitments due within this many "
            "days. Overdue and undated commitments are always kept.",
        ),
    ) -> dict[str, Any]:
        # The Commitment & Deadline Radar (spec #1): the same ranked list the CLI
        # `gum agenda` and MCP `agenda` surfaces build, exposed over HTTP for any
        # local app. Like /suggestions it runs the local text model to extract
        # commitments from the user's propositions, then pseudonymizes the
        # model-written text fields when the server is sanitized.
        from datetime import datetime

        from .agenda import (
            agenda_item_to_commitment,
            apply_overrides,
            build_agenda,
            _sort_key_commitment,
            _today_anchor,
        )

        # Anchor day math on the user's LOCAL "now" — the same frame as the
        # `today` field below and the deadlines the observers recorded — so a
        # commitment's `days_until_due` ("Due today"/"tomorrow") agrees with the
        # calendar day the UI buckets it under. (Using UTC here would drift a day
        # for evening/negative-offset timezones.)
        now = datetime.now().astimezone()
        # Build the full radar first (limit=None) so the user's overrides can
        # reorder/reinstate items before the final `limit` is applied; overrides
        # are overlaid on top so an edit shows even before re-inference catches up.
        commitments = await build_agenda(
            gum_instance, limit=None, window_days=window_days, now=now
        )
        overrides = await gum_instance.list_agenda_overrides()
        commitments = apply_overrides(commitments, overrides, now=now)
        # Fold in explicitly-added items (assistant/user), then rank the whole set
        # together so a pinned item lands where its urgency puts it.
        items = await gum_instance.list_agenda_items()
        commitments += [agenda_item_to_commitment(it, now) for it in items]
        commitments.sort(key=_sort_key_commitment)
        commitments = commitments[:limit]
        return {
            "count": len(commitments),
            "window_days": window_days,
            "today": _today_anchor(),
            "commitments": [
                await _serialize_commitment(c, sanitizer) for c in commitments
            ],
            "sanitized": sanitizer is not None,
        }

    @app.patch("/agenda/{proposition_id}")
    async def agenda_edit(
        proposition_id: int, body: AgendaEditIn
    ) -> dict[str, Any]:
        # Correct a generated agenda item (GUMBO Agenda page). The edit persists as
        # an override and propagates back into the GUM (proposition rewrite when
        # the date maps cleanly + a correction observation). A malformed due date
        # is rejected rather than silently dropped.
        from .agenda import _parse_due

        due = body.due_date.strip() if body.due_date else None
        if due and _parse_due(due) is None:
            return {"ok": False, "error": "due_date must be YYYY-MM-DD"}
        ok = await gum_instance.apply_agenda_override(
            proposition_id,
            title=body.title,
            due_date=due,
            status=body.status,
            clear_due_date=body.clear_due_date,
        )
        if not ok:
            return {"ok": False, "error": "not found"}
        return {"ok": True}

    @app.post("/agenda/{proposition_id}/dismiss")
    async def agenda_dismiss(proposition_id: int) -> dict[str, Any]:
        # Remove an item from the radar. Deliberately not a DELETE on the
        # proposition — dismissing "this isn't a commitment" must not erase a fact
        # that may still be true (see gum.dismiss_agenda_item).
        ok = await gum_instance.dismiss_agenda_item(proposition_id)
        return {"ok": ok}

    @app.post("/agenda/{proposition_id}/undo")
    async def agenda_undo(proposition_id: int) -> dict[str, Any]:
        # Revert a persisted edit/dismissal so the item shows the model's raw
        # output again. Cannot retract an already-pushed correction observation.
        ok = await gum_instance.clear_agenda_override(proposition_id)
        return {"ok": ok}

    # ── explicitly-added agenda items (assistant/user) ────────────────────────
    # Separate id space from proposition-backed commitments: these edit the
    # AgendaItem row directly (no proposition, no correction observation).
    @app.patch("/agenda/item/{item_id}")
    async def agenda_item_edit(
        item_id: int, body: AgendaEditIn
    ) -> dict[str, Any]:
        from .agenda import _parse_due

        due = body.due_date.strip() if body.due_date else None
        if due and _parse_due(due) is None:
            return {"ok": False, "error": "due_date must be YYYY-MM-DD"}
        ok = await gum_instance.update_agenda_item(
            item_id,
            title=body.title,
            due_date=due,
            status=body.status,
            clear_due_date=body.clear_due_date,
        )
        return {"ok": ok, **({} if ok else {"error": "not found"})}

    @app.post("/agenda/item/{item_id}/dismiss")
    async def agenda_item_dismiss(item_id: int) -> dict[str, Any]:
        ok = await gum_instance.set_agenda_item_dismissed(item_id, True)
        return {"ok": ok}

    @app.post("/agenda/item/{item_id}/undo")
    async def agenda_item_undo(item_id: int) -> dict[str, Any]:
        ok = await gum_instance.set_agenda_item_dismissed(item_id, False)
        return {"ok": ok}

    # ── GUMBO assistant desktop UI ────────────────────────────────────────
    @app.get("/gumbo", response_class=HTMLResponse)
    async def gumbo_page() -> str:
        # A single-page desktop-style front-end (project tabs + suggestion
        # cards) over the /suggestions endpoint above. The page is static; it
        # inherits the server's sanitize posture through that endpoint.
        return (_STATIC_DIR / "gumbo.html").read_text()

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
