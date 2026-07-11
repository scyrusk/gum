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
   the gate is held for review and no agent runs.
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
   state. Nothing the agent produced is used until you approve it.

!!! note "Grounding is pseudonymized and fail-closed"
    The agent is grounded on the **same** context assembly the `gum mcp` server
    hands local agents — retrieval on the substantive terms, then PII
    pseudonymization on egress, fail-closed. The execution bridge does not fork a
    second grounding path, so raw identities never reach an off-device model even
    when the backend relays context to a frontier model.

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

## Configuration

| Knob | Default | What it does |
| --- | --- | --- |
| `GUMBO_EXECUTION_ENABLED` | `false` | Master opt-in for the Python engine's `execute()` path |
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
