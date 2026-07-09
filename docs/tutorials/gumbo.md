# GUMBO: A Proactive Assistant

GUMBO is the proactive assistant from the paper (§4.3), built on top of your GUM.
Where the GUM quietly *learns* about you, GUMBO *acts*: it reads your most
relevant, high-confidence propositions and asks the **local** text model — the
same Ollama-backed model the GUM already uses, so nothing leaves your machine —
for concrete things it could do for you right now. Each suggestion is scored with
the paper's mixed-initiative expected-utility decision (§4.3.2) so only the ones
genuinely worth an interruption surface.

## Open the assistant

While the GUM daemon is running (`gum start`) it serves a single-page desktop-style
assistant at [`http://127.0.0.1:8422/gumbo`](http://127.0.0.1:8422/gumbo). Open it
directly in a browser, or from the macOS menu-bar app (`gum tray`) via
**Open GUMBO Assistant**.

The page has two sections (paper Fig 3):

- **Suggestions** — proactive suggestion cards, ranked by expected utility.
- **Memory** — a browsable, searchable table of the raw propositions in your GUM.

### Suggestions & project tabs

The suggestions view opens on an **All** tab. Add **project tabs** (the `+` button)
to focus GUMBO on a topic: each tab passes its name as a `focus`, so GUMBO retrieves
the propositions related to that project (paper §4.3.1) and keeps its suggestions
on-topic. Tabs are stored in your browser's local storage, so they persist across
sessions.

Each suggestion card carries a title, what GUMBO proposes to do, and a rationale
grounded in your propositions, plus buttons that close the mixed-initiative loop:

- **👍 / 👎** — thumbs up/down is fed back into the GUM as an observation, so future
  propositions (and thus suggestions) reflect what you actually found useful.
- **💬 Start Chat** — talk to GUMBO in detail about a suggestion (paper §4.3.3). The
  conversation is grounded in the same high-confidence propositions the suggestion
  came from, via the local text model, so GUMBO answers from what it actually knows.

### Memory (curating your model)

The **Memory** page (paper Fig 3B) lists the raw propositions, each annotated with
its **support** — the number of observations backing it — and a search box (BM25
over your propositions). From here you can curate the model directly:

- **Edit** a proposition that is close-but-wrong (its statement, reasoning, or the
  1–10 confidence pill) instead of throwing it away.
- **Forget** a proposition you judge wrong or don't want the model to hold.

Curation is responsive even while a batch is being processed — edits and deletes do
not block on in-flight inference.

## Build on it: the REST API

Everything the assistant page does goes through the same localhost-only REST API the
rest of the GUM serves (default `http://127.0.0.1:8422`), so any local app can drive
GUMBO too. When the daemon is started with `--sanitize`, model-written text in these
responses is pseudonymized just like the rest of the API.

```bash
# Scored, ranked proactive suggestions. Optional focus (a project/topic).
curl "http://127.0.0.1:8422/suggestions?focus=wedding%20planning&limit=5"

# surfaced_only=true keeps only suggestions the mixed-initiative filter would surface
# (expected utility of interrupting > staying quiet). rate_limited=true additionally
# applies the paper's token-bucket limit (~1 surfaced suggestion per minute).
curl "http://127.0.0.1:8422/suggestions?surfaced_only=true"
curl "http://127.0.0.1:8422/suggestions?rate_limited=true"

# Thumbs up/down feedback (fed back into the GUM as an observation).
curl -X POST "http://127.0.0.1:8422/suggestions/feedback" \
  -H 'Content-Type: application/json' \
  -d '{"title": "Draft the wedding-travel budget", "vote": "up"}'

# Start Chat: converse about a suggestion. `messages` is the running turn list;
# at least one user message is required. `suggestion` and `focus` are optional.
curl -X POST "http://127.0.0.1:8422/suggestions/chat" \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Help me start that budget."}],
       "suggestion": {"title": "Draft the wedding-travel budget"}}'

# Memory: browse/search raw propositions (each with a "support" count).
curl "http://127.0.0.1:8422/memory?q=email&limit=20"

# Curate: edit (any subset of text/reasoning/confidence) or delete a proposition.
curl -X PATCH "http://127.0.0.1:8422/memory/42" \
  -H 'Content-Type: application/json' -d '{"confidence": 4}'
curl -X DELETE "http://127.0.0.1:8422/memory/42"
```

Or use the engine directly from Python:

```python
from gum import Gumbo

# `g` is a live gum instance (same one the daemon runs). The engine is cheap to
# construct and does no I/O until you call it.
gumbo = Gumbo(g)
suggestions = await gumbo.generate(focus="wedding planning")   # ranked, de-duplicated
surfaced = await gumbo.surface()                               # rate-limited, worth interrupting
reply = await gumbo.chat([{"role": "user", "content": "Help me plan."}])
```

## Configuration

The engine reads a few environment variables (see `.env.example`), all optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GUMBO_MIN_CONFIDENCE` | `7` | Minimum proposition confidence (1–10) that may seed a suggestion. |
| `GUMBO_MAX_PROPOSITIONS` | `20` | How many propositions to feed the model per generation. |
| `GUMBO_NUM_SUGGESTIONS` | `5` | How many suggestions to ask the model for. |
| `GUMBO_DEDUP_THRESHOLD` | `0.6` | Lexical (Jaccard) overlap above which two suggestions are treated as duplicates (paper §4.3.2). `0` disables. |
| `GUMBO_SURFACE_INTERVAL` | `60` | Seconds per token in the surfacing rate limit (~1 surfaced suggestion per interval). `0` disables. |

## Tutorial

(coming soon: a walkthrough on how this was built!)
