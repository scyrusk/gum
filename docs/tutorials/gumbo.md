# GUMBO: A Proactive Assistant

GUMBO turns your GUM into a proactive assistant: it reads your high-confidence
propositions, generates candidate suggestions, and uses the paper's
mixed-initiative decision (§4.3.2) to decide which are actually worth
interrupting you with. The full engine, its `/suggestions` API, and the web UI
are documented elsewhere; this page covers the **execution bridge** (spec #4) —
the opt-in path that lets GUMBO not just *propose* an action but hand it to a
sandboxed agent and bring back a draft for your approval.

## Why an execution bridge?

The paper's loudest negative finding (§4.3.3, §8.4) is that GUMBO produces good
ideas but cannot act on them — *"Ideas are cheap. Execution is everything."* The
execution bridge closes that loop: a high-confidence, low-risk suggestion is
dispatched to a local agent that already receives grounded GUM context, and the
agent's output comes back as a **reviewable draft** — never a committed action.

## The safety model

The bridge is built so that it *cannot* take an irreversible action on your
behalf. Four independent guardrails enforce this:

1. **Default-OFF, explicit opt-in.** Nothing executes unless you turn it on. The
   Python engine is gated by `execution_enabled` (or the `GUMBO_EXECUTION_ENABLED`
   environment variable); running `gum execute` is itself the explicit opt-in on
   the CLI. With execution off, `Gumbo.execute()` is a no-op and GUMBO only ever
   proposes.
2. **A risk gate.** Before anything is dispatched, the local text model classifies
   the suggestion's implied action for **reversibility** and **risk**. A suggestion
   is auto-dispatched *only* when all of: it was already worth surfacing, its
   `P(useful)` clears a high bar (`GUM_EXECUTOR_MIN_PROBABILITY`, default 8), the
   action is read-only/reversible, and its risk is low (`GUM_EXECUTOR_MAX_RISK`,
   default 3). The classifier is biased toward the *less-safe* reading under
   uncertainty, so an ambiguous action stays proposal-only. Anything that misses
   the gate is held for review and no agent runs. Every score the gate reads is
   contractually bounded to 1–10 — the assessed `risk`, and the suggestion's own
   `P(useful)` and the `benefit`/`cost` scores behind its worth-surfacing decision
   — so a local-model reply with an out-of-range score is rejected as malformed
   (driving a retry) instead of a bogus value like `P(useful)=100` sailing through
   the confidence bar. If the classifier itself cannot complete (a failed or
   malformed local-model call, including a persistently out-of-range score), that
   suggestion fails *closed* to proposal-only rather than dispatching — and one
   un-assessable suggestion never aborts the rest of an `execute()` batch.
3. **A sandboxed agent.** The dispatched agent (the shipped backend shells out to
   the local `claude` CLI) runs in a **restricted scratch workspace** under the
   GUM data directory — never your real project tree. Each dispatch gets its own
   fresh, ephemeral subdirectory that is deleted when the run finishes, so no run's
   scratch files or state leak into the next. It runs with a hard wall-clock
   **timeout** that tears down its whole process tree on overrun. The CLI runs in
   a read/research-only **permission mode** (`plan` by default), so the tool layer
   *itself* refuses file edits, `Bash`, and outward-facing actions — the
   "produce a draft, never act" contract is enforced by the CLI, not merely
   requested in the prompt (which also instructs the agent to draft, not act).
4. **Human-in-the-loop approval.** Every result lands in a *pending-approval*
   state. Nothing the agent produced is used until you approve it. A dispatch that
   errors or times out — including a misbehaving backend that raises mid-run rather
   than reporting the failure — lands in a *failed* state (nothing to approve)
   instead of aborting the rest of an `execute()` batch.

!!! note "Grounding is pseudonymized and fail-closed"
    The agent is grounded on the **same** context assembly the `gum mcp` server
    hands local agents — retrieval on the substantive terms, then PII
    pseudonymization on egress, fail-closed. The execution bridge does not fork a
    second grounding path, so raw identities never reach an off-device model even
    when the backend relays context to a frontier model. The **whole** dispatched
    prompt is held to this bar, not just the grounding block: the GUM-generated
    suggestion text (which may embed real names or projects) and your own name —
    which the agent instruction would otherwise stamp in verbatim — are run
    through the *same* sanitizer, so they reach the backend only as the same
    stable pseudo-IDs (`[PERSON_1]`, `[ORG_1]`) the context uses. If that grounding
    cannot be built for a suggestion — a transient retrieval error, or the
    fail-closed sanitizer refusing to load its PII model — that suggestion is held
    *proposal-only* rather than dispatched un-grounded or un-sanitized, and (like a
    failed risk assessment) it never aborts the rest of an `execute()` batch.

    On return, the direction reverses only for the local review artifact: the
    executor runs the successful draft through `Sanitizer.rehydrate()` using
    that same persistent entity map, so the CLI and local REST review surface
    show a usable draft with real names restored. This is an on-device database
    lookup performed after the agent exits; the restored text is never sent back
    to the agent or any other cloud model.

    The read-only backend also explicitly requires the finished deliverable in
    its final stdout response. Even in Claude's `plan` permission mode, it must
    not call `ExitPlanMode`, leave the draft in a transient plan file, or return
    commentary about where the file was written; only stdout survives sandbox
    cleanup and becomes the artifact shown for review.

## Turning it on

```bash
pip install 'gum-ai[sanitize]'   # egress sanitization is fail-closed by default
```

The CLI review path is the simplest way to try it:

```bash
gum execute --review              # generate → gate → dispatch → approve/reject
gum execute --review "grant writing"   # steer generation with a project focus
```

`gum execute` runs the same rate-limited surfacing pipeline as the assistant (the
token bucket still caps how many suggestions can act per interval), risk-gates
each surface-worthy suggestion, and dispatches the ones that clear the gate. With
`--review` you'll see each outcome and, for every draft awaiting approval, be
asked to **approve / reject / skip**:

```
================================================================================
[1/2] Draft a reply to the reviewer thread   (DRAFT — awaiting your approval)
--------------------------------------------------------------------------------
Write a short reply thanking the reviewers and addressing the timeline question.
risk 2/10 · reversible · P(useful) 9/10

agent draft:
Hi all — thanks for the thorough review. On the timeline: …

approve / reject / skip? [a/r/s]
```

Approving keeps the draft; rejecting discards it. Either way the decision is
recorded through GUMBO's existing suggestion-feedback plumbing
(`add_suggestion_feedback`) — the same accept/reject signal a thumbs-up/down on a
suggestion uses — so your judgment on an *executed* draft feeds back into the
model and shapes future propositions and suggestions (paper §4.3). A suggestion
that failed the risk gate is shown as *proposal only* and is never prompted for
approval, because no agent ran.

Without `--review`, `gum execute` just lists the outcomes (useful for a dry run
to see what *would* be dispatched).

### From the local REST API

The same bridge is exposed over the localhost API (`gum/api.py`) so the web
suggestion cards — or any local app — can offer an "execute" action alongside the
existing thumbs up/down. It is the **same** default-OFF opt-in: the route only
runs when the server is built with `execute=True` (or `GUMBO_EXECUTION_ENABLED=1`);
otherwise it returns `{"ok": false, "enabled": false}` and touches no agent.

```bash
# enabled server:
curl -s -X POST localhost:8422/suggestions/execute \
     -H 'content-type: application/json' -d '{"focus": "grant writing"}'
```

```json
{
  "ok": true, "enabled": true, "focus": "grant writing", "dispatched": 1,
  "outcomes": [
    {
      "status": "pending_approval",
      "suggestion": {"title": "Draft a reply to the reviewer thread", "...": "..."},
      "assessment": {"reversibility": "reversible", "risk": 2, "rationale": "…"},
      "result": {"ok": true, "output": "Hi all — thanks for the thorough review. …"}
    }
  ]
}
```

Each `pending_approval` outcome is a draft awaiting the user's decision; a
gate-rejected one comes back `proposal_only` with no `result`. Approve/reject a
draft by POSTing the suggestion to the existing `/suggestions/feedback` route
(`vote: "up" | "down"`) — the same feedback plumbing the CLI review path uses, so
the accept/reject signal flows back into the GUM either way. Under
`gum start --sanitize`, suggestion and diagnostic metadata remains pseudonymized
like the rest of the API, but a successful draft is the local review artifact:
its entities stay rehydrated so the user can review the usable deliverable. The
feedback route receives suggestion metadata and the vote, never that restored
draft text.

## Configuration

| Knob | Default | What it does |
| --- | --- | --- |
| `GUMBO_EXECUTION_ENABLED` | `false` | Master opt-in for the `execute()` path — gates both the Python engine and the REST `POST /suggestions/execute` route |
| `GUM_EXECUTOR_MIN_PROBABILITY` | `8` | Minimum `P(useful)` (1–10) a suggestion needs to auto-dispatch |
| `GUM_EXECUTOR_MAX_RISK` | `3` | Maximum assessed risk (1–10) allowed to auto-dispatch |
| `GUM_EXECUTOR_CONTEXT_LIMIT` | `10` | How many propositions ground the dispatched task |
| `GUM_EXECUTOR_TIMEOUT` | `120` | Hard wall-clock cap (seconds) on an agent run |
| `GUM_EXECUTOR_WORKSPACE` | scratch dir under the GUM data dir | Root for the agent's sandbox; each dispatch gets a fresh, auto-deleted subdirectory here |
| `GUM_EXECUTOR_CLAUDE_ARGS` | — | Extra args appended to the `claude` CLI invocation |
| `GUM_EXECUTOR_PERMISSION_MODE` | `plan` | The CLI permission mode the agent runs in; set empty to disable (only for a fully-trusted backend) |

!!! warning "Review the code before enabling this"
    The bridge is default-OFF and proposal-only by design, and it cannot take an
    irreversible action before you approve a draft. Even so, execution is the
    highest-stakes part of GUMBO — read `gum/executor.py` and satisfy yourself
    about the gate and the sandbox before turning it on in your own environment.
