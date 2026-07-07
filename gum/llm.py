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

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Type, TypeVar
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
        listed = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        ).stdout
        if derived in listed:
            return derived

        log.info("Creating context-capped model %s (num_ctx=%d) from %s", derived, num_ctx, base_model)
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as fh:
            fh.write(f"FROM {base_model}\nPARAMETER num_ctx {num_ctx}\n")
            modelfile = fh.name
        try:
            subprocess.run(
                ["ollama", "create", derived, "-f", modelfile],
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
