# Spec #1 — Commitment & Deadline Radar (`gum agenda`)

Build a "commitment radar" for the GUM: a ranked, dated list of the user's open
commitments/deadlines derived from their propositions, surfaced as a CLI command,
an MCP resource, and a JSON API.

## Why

The GUM already infers things like "has a major impending deadline" and tracks
grants/papers/reviews, but that intelligence is latent. The paper's §4.2 vision is an
OS that surfaces the truly time-critical item and suppresses the rest. Make that a
standing artifact. Read `paper/gum.pdf` §4.2 for framing.

## Build

1. `gum/agenda.py` — a Commitment extraction + ranking module:
   - Pull candidate propositions via the existing `gum.query()`/`gum.recent()` APIs
     (`gum/gum.py`). Do NOT add new observers.
   - Use the text LLM (`gum/llm.py`) to classify propositions that imply an open
     commitment and extract `{title, due_date|null, owner/source, status_guess}`.
     Follow the prompt-construction style in `gum/prompts/gum.py`; add a new prompt
     constant there.
   - Rank by urgency = f(due proximity, confidence, decay). Reuse the decay score
     already on `Proposition` (see `gum/models.py`). Items with no date rank by
     confidence*recency.
   - Return a typed list (dataclass, mirror the `Suggestion` dataclass in `gum/gumbo.py`).
2. CLI: add a `gum agenda` subcommand in `gum/cli.py` following the existing
   `add_parser`/`cmd_*` pattern (see `cmd_recent`). Flags: `--limit`, `--json`,
   `--window DAYS`. Pretty terminal output by default; `--json` for machines.
3. MCP: expose it as an `@mcp.resource` (or tool) named `agenda` in `gum/mcp_server.py`,
   RESPECTING the existing sanitize path — reuse the same sanitization wrapper the
   other tools use. Never bypass fail-closed sanitization.

## Constraints

- Reuse existing query/LLM/sanitize plumbing; no schema migrations, no new DB tables.
- Keep everything on the user's existing local models (Llama/Ollama via `gum/llm.py`).
- Match surrounding code style, type hints, and async patterns.

## Done when

- `gum agenda` prints a ranked commitment list against the live GUM db.
- `gum agenda --json` emits valid JSON.
- The MCP `agenda` resource returns sanitized output.
- Add unit tests under `tests/` mirroring existing test style; full suite passes; lint clean.
- Update `docs/` and README with the new command.
