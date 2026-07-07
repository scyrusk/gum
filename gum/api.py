# api.py
#
# A small, localhost-only REST API so *any* local application (in any language)
# can build on what the GUM has learned. It is served inside the listening
# daemon and shares the same live `gum` instance, so reads always reflect the
# current model. It is strictly read-only and binds to 127.0.0.1.

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query

from .gum import gum
from .models import Observation, Proposition


def _serialize_proposition(prop: Proposition, score: float | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": prop.id,
        "text": prop.text,
        "reasoning": prop.reasoning,
        "confidence": prop.confidence,
        "decay": prop.decay,
        "created_at": prop.created_at.isoformat() if prop.created_at else None,
        "updated_at": prop.updated_at.isoformat() if prop.updated_at else None,
    }
    if score is not None:
        data["score"] = score
    return data


def _serialize_observation(obs: Observation) -> dict[str, Any]:
    return {
        "id": obs.id,
        "observer_name": obs.observer_name,
        "content": obs.content,
        "content_type": obs.content_type,
        "created_at": obs.created_at.isoformat() if obs.created_at else None,
    }


def create_app(gum_instance: gum) -> FastAPI:
    """Build the FastAPI app backed by a live `gum` instance."""
    app = FastAPI(
        title="GUM Local API",
        description="Read-only local interface to your General User Model.",
        version="1.0.0",
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "user": gum_instance.user_name}

    @app.get("/query")
    async def query(
        q: str = Query("", description="Free-text query (BM25). Empty = recent."),
        limit: int = Query(10, ge=1, le=100),
        mode: str = Query("OR", description="OR | AND | PHRASE"),
    ) -> dict[str, Any]:
        results = await gum_instance.query(q, limit=limit, mode=mode)
        return {
            "query": q,
            "results": [_serialize_proposition(p, score) for p, score in results],
        }

    @app.get("/recent")
    async def recent(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
        props = await gum_instance.recent(limit=limit)
        return {"results": [_serialize_proposition(p) for p in props]}

    @app.get("/observations")
    async def observations(limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
        obs = await gum_instance.recent_observations(limit=limit)
        return {"results": [_serialize_observation(o) for o in obs]}

    return app


def build_server(gum_instance: gum, host: str = "127.0.0.1", port: int = 8422):
    """Build a uvicorn Server for the API (caller awaits ``server.serve()``).

    Host defaults to 127.0.0.1 so the API is never exposed off-machine.
    """
    import uvicorn

    config = uvicorn.Config(
        create_app(gum_instance),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(config)
