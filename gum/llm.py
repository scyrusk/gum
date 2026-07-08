# llm.py
#
# Central inference layer for the GUM.
#
# Everything the GUM sends to a language/vision model goes through here so that:
#   1. Inference defaults to a *local* Ollama server (OpenAI-compatible API).
#   2. A privacy guard refuses to talk to any non-local endpoint unless the user
#      explicitly opts in with GUM_ALLOW_REMOTE=1. This is what makes "no data
#      leaves the machine by default" a guarantee rather than a convention.
#   3. Structured (JSON) output is parsed and repaired robustly, since local
#      models are less strict than hosted ones about schema adherence.

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Type, TypeVar
from urllib.parse import urlparse

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .schemas import get_schema

# Default local Ollama OpenAI-compatible endpoint.
DEFAULT_API_BASE = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"  # Ollama ignores the key, but the client requires one.

# Context-window caps (tokens) for the two roles. Ollama otherwise loads models
# at their full trained context (e.g. 128K for qwen2.5-vl), which reserves a huge
# KV cache — a 7B vision model balloons to ~52GB and can no longer share memory
# with the text model, causing constant evict/reload thrashing. We bake these
# caps into lean derived models (see ensure_capped_model) so both stay resident.
# Override via env (GUM_VISION_NUM_CTX / GUM_TEXT_NUM_CTX).
DEFAULT_VISION_NUM_CTX = 16384
DEFAULT_TEXT_NUM_CTX = 32768

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", ""}

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("gum.llm")


# --------------------------------------------------------------------------- #
# Global inference slot                                                       #
# --------------------------------------------------------------------------- #
# A single local GPU running large models (e.g. a 32B text model + a 7B vision
# model, both resident) gets no benefit from overlapping requests: with Ollama's
# default OLLAMA_NUM_PARALLEL=1 same-model requests simply queue, and two
# *different* models fired at once just split the one GPU's compute — each runs
# proportionally slower, for no net throughput gain, while doubling the in-flight
# KV cache and risking eviction of a co-resident model (the very thrash the
# context caps above exist to prevent). Yet the GUM naturally overlaps work: the
# screen observer emits vision calls while the batch loop is mid-way through its
# text-model calls, and bursts of user activity stack multiple observations.
#
# This semaphore funnels every model call (vision + text) through a single global
# slot so requests serialize cleanly at full speed instead of colliding. It is
# released between calls, so a long proposition batch never blocks a fresh
# observation for more than one call (and vice versa). Widen it only if your
# Ollama is explicitly configured for real parallelism (OLLAMA_NUM_PARALLEL) with
# the VRAM headroom to match, via GUM_MAX_CONCURRENT_INFERENCE.
_INFERENCE_SEMAPHORE: asyncio.Semaphore | None = None


def _max_concurrent_inference() -> int:
    try:
        return max(1, int(os.getenv("GUM_MAX_CONCURRENT_INFERENCE", "1")))
    except ValueError:
        return 1


def inference_semaphore() -> asyncio.Semaphore:
    """Return the process-wide inference slot, created lazily on first use.

    Lazy creation ties the semaphore to the running event loop the first time a
    coroutine acquires it, which keeps a single shared instance across the
    screen observer and the batch loop (they run in the same loop/process).
    """
    global _INFERENCE_SEMAPHORE
    if _INFERENCE_SEMAPHORE is None:
        _INFERENCE_SEMAPHORE = asyncio.Semaphore(_max_concurrent_inference())
    return _INFERENCE_SEMAPHORE


# --------------------------------------------------------------------------- #
# Endpoint resolution + privacy guard                                         #
# --------------------------------------------------------------------------- #
def resolve_api_base(role: str, override: str | None = None) -> str:
    """Resolve the API base URL for a given role ("gum" or "screen").

    Precedence: explicit override -> role-specific env -> shared GUM env ->
    local Ollama default. Never falls back to a hosted provider.
    """
    role_env = "SCREEN_LM_API_BASE" if role == "screen" else "GUM_LM_API_BASE"
    return (
        override
        or os.getenv(role_env)
        or os.getenv("GUM_LM_API_BASE")
        or DEFAULT_API_BASE
    )


def resolve_api_key(role: str, override: str | None = None) -> str:
    """Resolve the API key. Deliberately does NOT fall back to OPENAI_API_KEY:
    a stray OpenAI key in the environment must not silently enable egress."""
    role_env = "SCREEN_LM_API_KEY" if role == "screen" else "GUM_LM_API_KEY"
    return override or os.getenv(role_env) or os.getenv("GUM_LM_API_KEY") or DEFAULT_API_KEY


def resolve_num_ctx(role: str, override: int | None = None) -> int:
    """Resolve the context-window cap (tokens) for a role ("gum" or "screen")."""
    if override is not None:
        return override
    if role == "screen":
        env = os.getenv("GUM_VISION_NUM_CTX")
        return int(env) if env else DEFAULT_VISION_NUM_CTX
    env = os.getenv("GUM_TEXT_NUM_CTX")
    return int(env) if env else DEFAULT_TEXT_NUM_CTX


def capped_model_name(base_model: str, num_ctx: int) -> str:
    """Deterministic name for the context-capped derived model of *base_model*."""
    safe = re.sub(r"[^a-z0-9._-]", "-", base_model.lower())
    return f"gum-{safe}-ctx{num_ctx}"


# Common install locations for the ollama CLI, checked when it isn't on PATH.
_OLLAMA_BIN_CANDIDATES = (
    "/usr/local/bin/ollama",
    "/opt/homebrew/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)


def _ollama_bin() -> str:
    """Resolve the ollama CLI path robustly.

    GUM is often launched from a context with a minimal PATH — a macOS
    LaunchAgent / menu-bar app, launchd, or cron — that omits /usr/local/bin and
    Homebrew's bin. A bare ``ollama`` then isn't found, and model provisioning
    silently falls back to *uncapped* base models (full 128K context, huge KV
    cache, models evicting each other). So look past PATH: honor an explicit
    GUM_OLLAMA_BIN override, then PATH, then the usual install locations.
    """
    override = os.getenv("GUM_OLLAMA_BIN")
    if override:
        return override
    found = shutil.which("ollama")
    if found:
        return found
    for cand in _OLLAMA_BIN_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return "ollama"  # last resort; raises FileNotFoundError as before if truly absent


def ensure_capped_model(base_model: str, num_ctx: int, *, logger: logging.Logger | None = None) -> str:
    """Ensure a lean, context-capped derived model of *base_model* exists in Ollama.

    Ollama's OpenAI-compatible ``/v1`` endpoint ignores per-request context
    options, so we bake ``num_ctx`` into a derived model via a tiny Modelfile
    (``FROM <base>`` + ``PARAMETER num_ctx``). A ``/v1`` request against the
    derived model then loads at the capped context, keeping the KV cache small
    enough that the vision and text models can stay resident together.

    Idempotent: returns the derived name, creating it only if missing. If the
    ``ollama`` CLI isn't available, falls back to the base model unchanged.
    """
    log = logger or logging.getLogger("gum.llm")

    # Already a derived model — don't double-wrap.
    if base_model.startswith("gum-") and "-ctx" in base_model:
        return base_model

    derived = capped_model_name(base_model, num_ctx)
    try:
        ollama = _ollama_bin()
        listed = subprocess.run(
            [ollama, "list"], capture_output=True, text=True, check=True
        ).stdout
        if derived in listed:
            return derived

        log.info("Creating context-capped model %s (num_ctx=%d) from %s", derived, num_ctx, base_model)
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as fh:
            fh.write(f"FROM {base_model}\nPARAMETER num_ctx {num_ctx}\n")
            modelfile = fh.name
        try:
            subprocess.run(
                [ollama, "create", derived, "-f", modelfile],
                capture_output=True, text=True, check=True,
            )
        finally:
            os.unlink(modelfile)
        return derived
    except FileNotFoundError:
        log.warning("`ollama` CLI not found; using %s without a context cap.", base_model)
        return base_model
    except subprocess.CalledProcessError as exc:
        log.warning("Could not create capped model %s (%s); using %s.", derived, exc.stderr.strip() if exc.stderr else exc, base_model)
        return base_model


# --------------------------------------------------------------------------- #
# Keeping models resident (Ollama keep-alive)                                 #
# --------------------------------------------------------------------------- #
# The GUM talks to Ollama through its OpenAI-compatible ``/v1`` endpoint, which
# silently *ignores* the ``keep_alive`` option. So models fall back to Ollama's
# default 5-minute idle window and unload from VRAM once no request arrives for
# that long — after which the next observation/proposition pays a full cold
# reload (tens of GB from disk into VRAM, i.e. many seconds of stall). The
# earlier efficiency work makes this worse, not better: the unchanged-frame gate
# and the capture-loop idle backoff both deliberately stop issuing inference
# while the user is reading, so a normal reading pause reliably crosses the
# 5-minute line and the next click is slow — exactly the "models are loaded yet
# still slow" symptom.
#
# Ollama's *native* ``/api/generate`` endpoint DOES honor ``keep_alive``. A
# periodic empty-prompt ping there (``done_reason:"load"``, no generation, ~0.1s)
# resets each model's residency timer. We ping on a cadence shorter than the
# keep_alive window so the text and vision models stay pinned while the GUM runs,
# using a *finite* keep_alive (refreshed every tick) rather than ``-1`` so the
# models unload on their own once the GUM stops — no VRAM left pinned forever
# after shutdown.


def ollama_native_base(api_base: str) -> str | None:
    """Map an Ollama OpenAI-compat base (``…/v1``) to its native ``…/api`` base.

    Returns ``None`` when the URL doesn't end in ``/v1`` (i.e. it doesn't look
    like an Ollama endpoint), so keep-alive pinning — an Ollama-specific
    feature — is simply skipped for custom/non-Ollama servers.
    """
    base = (api_base or "").rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")].rstrip("/") + "/api"
    return None


def keep_warm_enabled() -> bool:
    """Whether the periodic keep-alive pinger should run (GUM_KEEP_WARM)."""
    return os.getenv("GUM_KEEP_WARM", "1") != "0"


def resolve_keep_alive() -> int | str:
    """Ollama ``keep_alive`` value (GUM_KEEP_ALIVE, default '15m').

    Integer-like values (e.g. ``-1`` for "never unload", or a second count) are
    passed as ints; duration strings like ``15m`` are passed through verbatim.
    """
    v = os.getenv("GUM_KEEP_ALIVE", "15m")
    try:
        return int(v)
    except ValueError:
        return v


def resolve_keep_warm_interval() -> float:
    """Seconds between keep-alive pings (GUM_KEEP_WARM_INTERVAL, default 240).

    Floored at 30s so a misconfiguration can't turn this into a busy loop. Keep
    it comfortably below the keep_alive window so a single missed ping never lets
    a model lapse.
    """
    try:
        return max(30.0, float(os.getenv("GUM_KEEP_WARM_INTERVAL", "240")))
    except ValueError:
        return 240.0


def resolve_text_idle_unload() -> float:
    """Seconds of no observations before the text model is released from memory.

    The text (proposition) model is the biggest resident cost and is only used
    in bursts when a batch runs. Keeping it warm through short pauses avoids a
    cold reload per batch, but through a long lull it just heats the machine for
    nothing. After this idle window with no new observations, the keep-warm
    pinger explicitly releases it (freeing memory); it reloads on the next batch.
    Set GUM_TEXT_IDLE_UNLOAD=0 to keep the old always-warm behaviour.
    """
    try:
        return max(0.0, float(os.getenv("GUM_TEXT_IDLE_UNLOAD", "600")))
    except ValueError:
        return 600.0


def _native_generate_ping(url: str, model: str, keep_alive: int | str) -> None:
    """Fire a single empty-prompt native ``/api/generate`` load request.

    An empty prompt makes Ollama load (or refresh the residency timer of) the
    model and return immediately without generating any tokens. Uses stdlib
    urllib so no HTTP dependency beyond what's already installed; called from a
    worker thread via ``to_thread``.
    """
    import urllib.request

    payload = json.dumps(
        {"model": model, "prompt": "", "stream": False, "keep_alive": keep_alive}
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def resolve_warm_targets(targets: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Resolve ``(openai_api_base, model)`` pairs to unique ``(native_url, model)``.

    Drops any target whose base isn't an Ollama ``/v1`` endpoint and de-dupes,
    so the two roles that share one local Ollama don't get pinged twice.
    """
    seen: set[tuple[str, str]] = set()
    resolved: list[tuple[str, str]] = []
    for base, model in targets:
        native = ollama_native_base(base)
        if native is None:
            continue
        entry = (native + "/generate", model)
        if entry in seen:
            continue
        seen.add(entry)
        resolved.append(entry)
    return resolved


async def keep_models_warm(
    targets: list[tuple[str, str]],
    *,
    gated_targets: list[tuple[str, str]] | None = None,
    is_active: "Callable[[], bool] | None" = None,
    keep_alive: int | str = "15m",
    interval: float = 240.0,
    logger: logging.Logger | None = None,
) -> None:
    """Periodically ping each ``(api_base, model)`` so Ollama keeps it resident.

    ``targets`` are pinned unconditionally (e.g. the frequently-used vision
    model). ``gated_targets`` are pinned *only while ``is_active()`` is True*;
    when activity stops they are released with an immediate-unload ping so the
    memory is reclaimed promptly instead of lingering for a full keep_alive
    window. This lets the big text model unload during quiet periods and reload
    on the next batch.

    Runs until cancelled. Each ping is routed through the global inference slot
    so it serializes cleanly with real work; a ping that fails (e.g. the endpoint
    isn't Ollama after all) is logged at debug and retried next tick rather than
    killing the loop.
    """
    log = logger or logging.getLogger("gum.llm")
    resolved = resolve_warm_targets(targets)
    resolved_gated = resolve_warm_targets(gated_targets or [])
    if not resolved and not resolved_gated:
        log.debug("keep-warm: no Ollama targets to pin; pinger idle")
        return

    async def _ping(url: str, model: str, ka: int | str) -> None:
        try:
            async with inference_semaphore():
                await asyncio.to_thread(_native_generate_ping, url, model, ka)
        except Exception as exc:  # pragma: no cover - network/transport
            log.debug("keep-warm ping failed for %s (%s)", model, exc)

    log.info(
        "keep-warm: pinning %d model(s) every %.0fs (keep_alive=%s); %d activity-gated",
        len(resolved),
        interval,
        keep_alive,
        len(resolved_gated),
    )

    gated_loaded = True  # gated models are resident at startup (just provisioned)
    while True:
        for url, model in resolved:
            await _ping(url, model, keep_alive)

        if resolved_gated:
            active = is_active is None or is_active()
            if active:
                for url, model in resolved_gated:
                    await _ping(url, model, keep_alive)
                gated_loaded = True
            elif gated_loaded:
                # Just went idle: release the gated (text) models now.
                log.info("keep-warm: idle — releasing %d gated model(s) to free memory", len(resolved_gated))
                for url, model in resolved_gated:
                    await _ping(url, model, 0)
                gated_loaded = False

        await asyncio.sleep(interval)


def _allow_remote() -> bool:
    return os.getenv("GUM_ALLOW_REMOTE", "").strip().lower() in {"1", "true", "yes", "on"}


def assert_local(base_url: str) -> None:
    """Raise unless *base_url* points at the local machine.

    Override with GUM_ALLOW_REMOTE=1 when you deliberately want to use a remote
    endpoint (e.g. a GPU box on your LAN). By default the GUM refuses, so no
    screenshots or propositions can leave the machine by accident.
    """
    host = (urlparse(base_url).hostname or "").lower()
    if host in _LOCAL_HOSTS or _allow_remote():
        return
    raise RuntimeError(
        f"Refusing to send data to non-local inference endpoint '{base_url}' "
        f"(host '{host}'). The GUM keeps all screenshots and propositions on "
        f"your machine by default. If you really intend to use a remote model, "
        f"set GUM_ALLOW_REMOTE=1."
    )


def make_client(
    role: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
) -> AsyncOpenAI:
    """Build an AsyncOpenAI client for *role*, enforcing the local-only guard."""
    base_url = resolve_api_base(role, api_base)
    assert_local(base_url)
    logger.debug("LLM client for role=%s -> %s", role, base_url)
    return AsyncOpenAI(base_url=base_url, api_key=resolve_api_key(role, api_key))


# --------------------------------------------------------------------------- #
# Robust structured (JSON) output                                             #
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(content: str) -> str:
    """Best-effort extraction of a JSON object from a model response.

    Handles ```json fenced blocks and stray prose/reasoning around the object
    by falling back to the outermost brace-delimited span.
    """
    if content is None:
        return ""
    text = content.strip()

    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    # If there's leading/trailing prose, keep the outermost {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


async def structured_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    schema_model: Type[T],
    *,
    retries: int = 3,
    temperature: float | None = None,
    logger: logging.Logger | None = None,
) -> T:
    """Call the model and return an instance of *schema_model*, validated.

    Uses the OpenAI ``response_format`` json_schema shape (which Ollama maps to
    its native structured-output ``format``). On any parse/validation failure it
    re-prompts the model with the specific error and retries, then finally falls
    back to json_object mode. Raises the last error only if every attempt fails.
    """
    log = logger or logging.getLogger("gum.llm")
    json_schema = schema_model.model_json_schema()
    convo = list(messages)
    last_err: Exception | None = None

    for attempt in range(retries):
        # Last attempt: relax to json_object mode in case the server/model
        # struggles with a strict json_schema.
        use_schema = attempt < retries - 1
        response_format = (
            get_schema(json_schema) if use_schema else {"type": "json_object"}
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": convo,
            "response_format": response_format,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            async with inference_semaphore():
                rsp = await client.chat.completions.create(**kwargs)
            content = rsp.choices[0].message.content or ""
            return schema_model.model_validate_json(extract_json(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_err = exc
            log.warning(
                "Structured output invalid (attempt %d/%d): %s",
                attempt + 1,
                retries,
                exc,
            )
            # Feed the bad output back so the model can correct itself.
            convo = list(messages) + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not match the required JSON "
                        f"schema. Error: {exc}. Respond again with ONLY valid JSON "
                        "matching the schema, no prose or code fences."
                    ),
                },
            ]
        except Exception as exc:  # transport / API errors
            last_err = exc
            log.warning(
                "Inference call failed (attempt %d/%d): %s", attempt + 1, retries, exc
            )

    raise RuntimeError(
        f"structured_completion failed after {retries} attempts for model "
        f"'{model}': {last_err}"
    )
