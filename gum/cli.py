from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import os
import sys
import signal
import argparse
import asyncio
import shutil
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from gum import gum
from gum.observers import Screen
from gum import daemon
from gum.llm import ensure_capped_model, resolve_num_ctx

# ── local-first defaults ──────────────────────────────────────────────────────
DEFAULT_TEXT_MODEL = "qwen2.5-coder:32b"
DEFAULT_VISION_MODEL = "qwen2.5vl:7b"
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8422

# Observations are stored in UTC; the CLI reports/filters days in Eastern time.
EASTERN = ZoneInfo("America/New_York")


def _parse_eastern_day(date_str: str) -> tuple[datetime, datetime]:
    """Parse a date string as a calendar day in Eastern time.

    Returns (start_utc, end_utc): the UTC bounds of that Eastern-time day,
    i.e. [midnight ET, next midnight ET). Accepts M/D/YYYY and YYYY-MM-DD.
    """
    parsed = None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%m-%d-%Y", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError(
            f"Could not parse date '{date_str}'. Use M/D/YYYY (e.g. 7/7/2026) or YYYY-MM-DD."
        )
    start = datetime(parsed.year, parsed.month, parsed.day, tzinfo=EASTERN)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


# ── config resolution helpers ─────────────────────────────────────────────────
def _text_model(args) -> str:
    return getattr(args, "text_model", None) or os.getenv("MODEL_NAME") or DEFAULT_TEXT_MODEL


def _vision_model(args) -> str:
    return getattr(args, "vision_model", None) or os.getenv("SCREEN_MODEL_NAME") or DEFAULT_VISION_MODEL


def _user_name(args) -> str | None:
    return getattr(args, "user_name", None) or os.getenv("USER_NAME")


def _api_host(args) -> str:
    return getattr(args, "host", None) or os.getenv("GUM_API_HOST") or DEFAULT_API_HOST


def _api_port(args) -> int:
    return getattr(args, "port", None) or int(os.getenv("GUM_API_PORT", str(DEFAULT_API_PORT)))


def _batch_sizes(args) -> tuple[int, int]:
    mn = getattr(args, "min_batch_size", None) or int(os.getenv("MIN_BATCH_SIZE", "5"))
    mx = getattr(args, "max_batch_size", None) or int(os.getenv("MAX_BATCH_SIZE", "15"))
    return mn, mx


def _history_k(args) -> int:
    hk = getattr(args, "history_k", None)
    if hk is not None:
        return hk
    return int(os.getenv("HISTORY_K", "2"))


def _sanitize_enabled(args) -> bool:
    """Whether to pseudonymize PII on output: CLI flag → GUM_SANITIZE env → off."""
    if getattr(args, "sanitize", False):
        return True
    return os.getenv("GUM_SANITIZE", "").strip().lower() in ("1", "true", "yes", "on")


# ── argument parsing ──────────────────────────────────────────────────────────
def _add_run_args(p: argparse.ArgumentParser) -> None:
    """Runtime arguments shared by `start` and the internal `_run` command."""
    p.add_argument("--user-name", "-u", type=str, help="Your full name (or set USER_NAME)")
    p.add_argument("--text-model", "-m", type=str, help="Ollama text model for propositions")
    p.add_argument("--vision-model", type=str, help="Ollama vision model for screen transcription")
    p.add_argument("--port", type=int, help="Local REST API port (default 8422)")
    p.add_argument("--host", type=str, help="Local REST API host (default 127.0.0.1)")
    p.add_argument("--min-batch-size", type=int, help="Min observations to trigger a batch")
    p.add_argument("--max-batch-size", type=int, help="Max observations per batch")
    p.add_argument("--history-k", type=int, help="Screenshots of history kept per summary (default 2)")
    p.add_argument("--sanitize", "-s", action="store_true",
                   help="Serve only PII-sanitized data over the local REST API (fail-closed)")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="gum",
        description="GUM - a local General User Model powered by Ollama",
    )
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="Start the GUM in the background")
    _add_run_args(p_start)

    sub.add_parser("stop", help="Stop the background GUM")
    sub.add_parser("status", help="Show whether the GUM is running")

    # Internal foreground runner (used by `start`); hidden from help.
    p_run = sub.add_parser("_run")
    _add_run_args(p_run)

    p_query = sub.add_parser("query", help="Search the GUM for propositions")
    p_query.add_argument("query", nargs="?", default="", help="Query string (empty = recent)")
    p_query.add_argument("--limit", "-l", type=int, default=10)
    p_query.add_argument("--mode", type=str, default="OR", help="OR | AND | PHRASE")
    p_query.add_argument("--user-name", "-u", type=str)
    p_query.add_argument("--text-model", "-m", type=str)
    p_query.add_argument("--sanitize", "-s", action="store_true",
                         help="Replace PII with consistent pseudo-IDs before output")

    p_recent = sub.add_parser("recent", help="List the most recent propositions")
    p_recent.add_argument("--limit", "-l", type=int, default=10)
    p_recent.add_argument("--user-name", "-u", type=str)
    p_recent.add_argument("--text-model", "-m", type=str)
    p_recent.add_argument("--sanitize", "-s", action="store_true",
                          help="Replace PII with consistent pseudo-IDs before output")

    p_agenda = sub.add_parser(
        "agenda",
        help="Show a ranked radar of your open commitments and deadlines",
    )
    p_agenda.add_argument("--limit", "-l", type=int, default=10,
                          help="Max commitments to show (default 10)")
    p_agenda.add_argument("--window", "-w", type=int, default=None, metavar="DAYS",
                          help="Only show commitments due within this many days "
                          "(overdue and undated ones are always kept)")
    p_agenda.add_argument("--json", action="store_true",
                          help="Emit the radar as JSON for machines")
    p_agenda.add_argument("--user-name", "-u", type=str)
    p_agenda.add_argument("--text-model", "-m", type=str)
    p_agenda.add_argument("--sanitize", "-s", action="store_true",
                          help="Replace PII with consistent pseudo-IDs before output")

    p_obs = sub.add_parser("observations", help="List raw observations (all of a day with --date)")
    p_obs.add_argument("--limit", "-l", type=int, default=None,
                       help="Max results (default 10; all-day when --date is set)")
    p_obs.add_argument("--date", "-d", type=str,
                       help="Only observations from this Eastern-time day, e.g. 7/7/2026 or 2026-07-07")
    p_obs.add_argument("--full", action="store_true", help="Print full content instead of a preview")
    p_obs.add_argument("--output", "-o", type=str, help="Write results to this file instead of stdout")
    p_obs.add_argument("--user-name", "-u", type=str)
    p_obs.add_argument("--text-model", "-m", type=str)
    p_obs.add_argument("--sanitize", "-s", action="store_true",
                       help="Replace PII with consistent pseudo-IDs before output")

    p_review = sub.add_parser("review", help="Open the proposition review GUI in your browser")
    p_review.add_argument("--port", type=int, help="Review server port (default 8423)")
    p_review.add_argument("--host", type=str, help="Review server host (default 127.0.0.1)")
    p_review.add_argument("--user-name", "-u", type=str)
    p_review.add_argument("--text-model", "-m", type=str)

    p_execute = sub.add_parser(
        "execute",
        help="Run the GUMBO execution bridge: dispatch high-confidence, reversible "
        "suggestions to a sandboxed agent and review the drafts for approval (spec #4)",
    )
    p_execute.add_argument(
        "focus",
        nargs="?",
        default=None,
        help="Optional project focus to steer which suggestions are generated",
    )
    p_execute.add_argument(
        "--review",
        action="store_true",
        help="Interactively approve or reject each agent-produced draft; the "
        "decision is recorded as GUMBO feedback (default: just list the outcomes)",
    )
    p_execute.add_argument("--user-name", "-u", type=str)
    p_execute.add_argument("--text-model", "-m", type=str)

    p_mcp = sub.add_parser(
        "mcp",
        help="Serve the GUM over MCP (stdio) so a local agent can gather sanitized context",
    )
    p_mcp.add_argument("--user-name", "-u", type=str)
    p_mcp.add_argument("--text-model", "-m", type=str)
    p_mcp.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Serve RAW propositions (default is fail-closed PII sanitization). "
        "Only for a fully-local, trusted agent.",
    )

    p_rehydrate = sub.add_parser(
        "rehydrate",
        help="Restore real values for pseudo-IDs in a file the agent produced "
        "from sanitized GUM context (the inverse of egress sanitization)",
    )
    p_rehydrate.add_argument(
        "input",
        nargs="?",
        help="File to rehydrate. Reads stdin when omitted or '-'.",
    )
    p_rehydrate.add_argument(
        "-o",
        "--output",
        type=str,
        help="Write result here. Defaults to overwriting INPUT in place; "
        "with stdin input, writes to stdout.",
    )

    sub.add_parser("tray", help="Launch the macOS menu-bar app (needs the [tray] extra)")

    sub.add_parser("reset-cache", help="Delete the GUM cache (~/.cache/gum) and exit")

    return parser, parser.parse_args()


# ── command handlers ──────────────────────────────────────────────────────────
def _provision_models(args) -> tuple[str, str]:
    """Ensure lean, context-capped derived models exist; return (vision, text).

    Baking a small num_ctx into derived models keeps the vision and text models
    small enough to stay co-resident in Ollama instead of thrashing.
    """
    print("Preparing local models (first run creates lean context-capped copies)…")
    vision = ensure_capped_model(_vision_model(args), resolve_num_ctx("screen"))
    text = ensure_capped_model(_text_model(args), resolve_num_ctx("gum"))
    return vision, text


def _run_command_from(args, vision_model: str, text_model: str) -> list[str]:
    """Build the argv for the detached `_run` process from `start` args."""
    cmd = [sys.executable, "-m", "gum.cli", "_run"]
    cmd += ["--user-name", _user_name(args)]
    cmd += ["--text-model", text_model]
    cmd += ["--vision-model", vision_model]
    cmd += ["--host", _api_host(args)]
    cmd += ["--port", str(_api_port(args))]
    mn, mx = _batch_sizes(args)
    cmd += ["--min-batch-size", str(mn), "--max-batch-size", str(mx)]
    cmd += ["--history-k", str(_history_k(args))]
    if _sanitize_enabled(args):
        cmd += ["--sanitize"]
    return cmd


def cmd_start(args) -> None:
    user = _user_name(args)
    if not user:
        print("Please provide a user name with -u or set USER_NAME.")
        return
    if daemon.is_running():
        print(f"GUM is already running (pid {daemon.read_pid()}). Use `gum stop` first.")
        return
    vision, text = _provision_models(args)
    try:
        pid = daemon.start(_run_command_from(args, vision, text))
    except RuntimeError as exc:
        print(str(exc))
        return
    print(f"GUM started (pid {pid}).")
    print(f"  Text model:   {text}")
    print(f"  Vision model: {vision}")
    print(f"  Local API:    http://{_api_host(args)}:{_api_port(args)}")
    print(f"  Logs:         {daemon.log_path()}")
    print("Use `gum status` to check, `gum stop` to stop.")


def cmd_stop(args) -> None:
    if daemon.stop():
        print("GUM stopped.")
    else:
        print("GUM is not running.")


async def cmd_status(args) -> None:
    pid = daemon.read_pid()
    if pid is None:
        print("GUM is not running.")
        return
    print(f"GUM is running (pid {pid}).")
    print(f"  Logs: {daemon.log_path()}")
    try:
        g = gum(_user_name(args) or "default", _text_model(args))
        await g.connect_db()
        latest = await g.recent(limit=1)
        if latest:
            p = latest[0]
            print(f"  Latest proposition ({p.created_at}): {p.text[:100]}")
        else:
            print("  No propositions yet — interact with your computer to build the model.")
    except Exception as exc:
        print(f"  (could not read database: {exc})")


async def cmd_run(args) -> None:
    """Foreground listener + local REST API. Launched detached by `gum start`."""
    from gum.api import build_server

    user = _user_name(args)
    if not user:
        print("USER_NAME is required to run the GUM.")
        return

    # Idempotent: names passed by `gum start` are already capped; this covers a
    # direct `gum _run` invocation with base model names.
    text_model = ensure_capped_model(_text_model(args), resolve_num_ctx("gum"))
    vision_model = ensure_capped_model(_vision_model(args), resolve_num_ctx("screen"))
    host, port = _api_host(args), _api_port(args)
    min_batch, max_batch = _batch_sizes(args)
    sanitize = _sanitize_enabled(args)

    api_note = "http://{}:{} (sanitized)".format(host, port) if sanitize else f"http://{host}:{port}"
    print(f"Listening as {user} — text={text_model}, vision={vision_model}, api={api_note}")

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    async with gum(
        user,
        text_model,
        Screen(vision_model, history_k=_history_k(args)),
        min_batch_size=min_batch,
        max_batch_size=max_batch,
    ) as gum_instance:
        server = build_server(gum_instance, host=host, port=port, sanitize=sanitize)
        api_task = asyncio.create_task(server.serve())
        await stop_event.wait()
        print("Shutting down GUM…")
        server.should_exit = True
        await api_task


async def _scrub(text: str, enabled: bool) -> str:
    """Pseudonymize *text* when sanitization is enabled, else return it unchanged."""
    if not enabled or not text:
        return text
    from gum.sanitize import get_sanitizer
    return await asyncio.to_thread(get_sanitizer().sanitize, text)


async def _scrub_fragment(text: str, enabled: bool) -> str:
    """Pseudonymize a short, context-free field (a bare name, a terse title).

    Routes through the sanitizer's carrier-context path (see
    :meth:`gum.sanitize.Sanitizer.sanitize_fragment`), which the NER model needs
    to reliably tag a lone name that a plain :func:`_scrub` would leak.
    """
    if not enabled or not text:
        return text
    from gum.sanitize import get_sanitizer
    return await asyncio.to_thread(get_sanitizer().sanitize_fragment, text)


async def cmd_query(args) -> None:
    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()
    result = await g.query(args.query, limit=args.limit, mode=args.mode)
    sanitize = _sanitize_enabled(args)
    print(f"\nFound {len(result)} results:")
    for prop, score in result:
        print(f"\nProposition: {await _scrub(prop.text, sanitize)}")
        if prop.reasoning:
            print(f"Reasoning: {await _scrub(prop.reasoning, sanitize)}")
        if prop.confidence is not None:
            print(f"Confidence: {prop.confidence:.2f}")
        print(f"Relevance Score: {score:.2f}")
        print("-" * 80)


async def cmd_recent(args) -> None:
    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()
    props = await g.recent(limit=args.limit)
    sanitize = _sanitize_enabled(args)
    print(f"\nRecent {len(props)} propositions:")
    for p in props:
        print(f"\nProposition: {await _scrub(p.text, sanitize)}")
        if p.reasoning:
            print(f"Reasoning: {await _scrub(p.reasoning, sanitize)}")
        if p.confidence is not None:
            print(f"Confidence: {p.confidence:.2f}")
        print(f"Created At: {p.created_at}")
        print("-" * 80)


def _due_label(days: int | None) -> str:
    """Human-readable deadline tag for a commitment, e.g. 'overdue 3d'."""
    if days is None:
        return "no date"
    if days < 0:
        return f"overdue {abs(days)}d"
    if days == 0:
        return "due today"
    return f"in {days}d"


async def cmd_agenda(args) -> None:
    """Print a ranked radar of the user's open commitments and deadlines."""
    from gum.agenda import build_agenda

    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()
    commitments = await build_agenda(
        g, limit=args.limit, window_days=args.window, now=datetime.now().astimezone()
    )

    # The model-written text fields (title/source/proposition_text) carry PII;
    # scrub them before they leave the process when sanitization is on. The
    # numeric/date fields never do, so they pass through untouched. title/source
    # are terse, context-free fragments (often a bare name) that the NER model
    # under-detects, so they use the carrier-context _scrub_fragment; the
    # full-sentence proposition_text uses the plain _scrub.
    sanitize = _sanitize_enabled(args)
    for c in commitments:
        c.title = await _scrub_fragment(c.title, sanitize)
        c.source = await _scrub_fragment(c.source, sanitize)
        c.proposition_text = await _scrub(c.proposition_text, sanitize)

    if args.json:
        import json
        print(json.dumps([c.to_dict() for c in commitments], indent=2))
        return

    if not commitments:
        print("\nNothing on the radar — no open commitments detected.")
        return

    print(f"\nCommitment radar — {len(commitments)} open item(s), most urgent first:")
    for c in commitments:
        conf = f"{c.confidence}/10" if c.confidence is not None else "?/10"
        print(f"\n  [{_due_label(c.days_until_due)}] {c.title}")
        due = c.due_date or "unscheduled"
        print(f"      due {due} · {c.source} · {c.status_guess} · "
              f"confidence {conf} · urgency {c.urgency:.2f}")
        print(f"      from: {c.proposition_text}")
    print("-" * 80)


async def cmd_observations(args) -> None:
    start_utc = end_utc = None
    if args.date:
        try:
            start_utc, end_utc = _parse_eastern_day(args.date)
        except ValueError as exc:
            print(exc)
            return

    # A date view returns the whole day unless the user set an explicit --limit.
    limit = args.limit if args.limit is not None else (1_000_000 if args.date else 10)

    # Exports (--output) default to full content; previews only make sense on-screen.
    full = args.full or bool(args.output)

    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()
    obs = await g.recent_observations(limit=limit, start_time=start_utc, end_time=end_utc)
    sanitize = _sanitize_enabled(args)

    lines: list[str] = []
    if args.date:
        obs = list(reversed(obs))  # chronological for a day view
        day = start_utc.astimezone(EASTERN).strftime("%m/%d/%Y")
        lines.append(f"{len(obs)} observations on {day} (Eastern):")
    else:
        lines.append(f"Recent {len(obs)} observations:")

    for o in obs:
        ts = o.created_at.replace(tzinfo=timezone.utc).astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z")
        # Pseudonymize the full content before previewing so truncation never
        # slices through a pseudo-ID token.
        raw = await _scrub(o.content, sanitize)
        content = raw if full else raw.replace("\n", " ")[:300]
        lines.append("")
        lines.append(f"[{o.observer_name}] {ts}  (id={o.id})")
        lines.append(content)
        if not full and len(raw) > 300:
            lines.append("… (use --full to see everything)")
        lines.append("-" * 80)

    text = "\n".join(lines)
    if args.output:
        path = os.path.expanduser(args.output)
        try:
            with open(path, "w") as fh:
                fh.write(text + "\n")
        except OSError as exc:
            print(f"Could not write to {path}: {exc}")
            return
        print(f"Wrote {len(obs)} observations to {path}")
    else:
        print("\n" + text)


async def cmd_review(args) -> None:
    """Serve the proposition review GUI locally and open it in the browser."""
    import webbrowser
    from gum.api import build_server

    host = getattr(args, "host", None) or DEFAULT_API_HOST
    port = getattr(args, "port", None) or int(os.getenv("GUM_REVIEW_PORT", "8423"))

    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()  # also creates the feedback table if it doesn't exist yet

    url = f"http://{host}:{port}/"
    server = build_server(g, host=host, port=port)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    api_task = asyncio.create_task(server.serve())
    print(f"GUM review UI running at {url}")
    print("Rate each proposition Accurate / Somewhat / Inaccurate; your feedback trains the model. Press Ctrl-C to stop.")
    await asyncio.sleep(0.6)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    await stop_event.wait()
    print("\nClosing review server…")
    server.should_exit = True
    await api_task


# The three approve/reject decisions the review surface accepts, and the aliases
# a user might type for each. "approve"/"reject" map onto the existing suggestion
# thumbs-up/down feedback; "skip" records nothing.
_REVIEW_APPROVE = frozenset({"a", "approve", "approved", "y", "yes", "keep"})
_REVIEW_REJECT = frozenset({"r", "reject", "rejected", "n", "no", "discard"})
_REVIEW_SKIP = frozenset({"s", "skip", ""})


def _render_outcome(outcome, index: int, total: int) -> str:
    """Format a single ExecutionOutcome for the review surface (no side effects)."""
    from gum.executor import (
        STATUS_FAILED,
        STATUS_PENDING_APPROVAL,
        STATUS_PROPOSAL_ONLY,
    )

    s = outcome.suggestion
    label = {
        STATUS_PENDING_APPROVAL: "DRAFT — awaiting your approval",
        STATUS_PROPOSAL_ONLY: "PROPOSAL ONLY — held, no agent ran",
        STATUS_FAILED: "FAILED — agent errored, nothing to approve",
    }.get(outcome.status, outcome.status)

    lines = [
        "",
        "=" * 80,
        f"[{index}/{total}] {s.title}   ({label})",
        "-" * 80,
        s.description.strip(),
    ]
    if outcome.assessment is not None:
        a = outcome.assessment
        lines.append(
            f"risk {a.risk}/10 · {a.reversibility} · P(useful) {s.probability_useful}/10"
        )
    if outcome.reason:
        lines.append(f"note: {outcome.reason}")
    if outcome.result is not None and outcome.result.output:
        lines.append("")
        lines.append("agent draft:")
        lines.append(outcome.result.output.strip())
    return "\n".join(lines)


async def review_outcomes(
    gum_instance,
    outcomes,
    *,
    interactive: bool = True,
    prompt=input,
    out=print,
) -> list[dict]:
    """Present each execution outcome and record approve/reject as GUMBO feedback.

    Renders every :class:`~gum.executor.ExecutionOutcome`; when *interactive*, each
    draft awaiting approval prompts the user to approve, reject, or skip. Approve
    and reject are fed back through the GUM's existing
    :meth:`~gum.gum.gum.add_suggestion_feedback` plumbing — the same accept/reject
    signal path a suggestion thumbs-up/down uses — so a decision on an *executed*
    draft trains future propositions just like any other reaction. Nothing is
    committed; the executor already produced only reviewable drafts. Returns the
    list of recorded ``{"title", "vote"}`` decisions. *prompt*/*out* are injectable
    so the loop is drivable in tests without real stdin/stdout.
    """
    from gum.executor import STATUS_PENDING_APPROVAL

    total = len(outcomes)
    recorded: list[dict] = []
    for i, outcome in enumerate(outcomes, 1):
        out(_render_outcome(outcome, i, total))
        if not interactive or outcome.status != STATUS_PENDING_APPROVAL:
            continue
        while True:
            choice = (prompt("approve / reject / skip? [a/r/s] ") or "").strip().lower()
            if choice in _REVIEW_APPROVE:
                vote = "up"
            elif choice in _REVIEW_REJECT:
                vote = "down"
            elif choice in _REVIEW_SKIP:
                vote = None
            else:
                out("Please answer a (approve), r (reject), or s (skip).")
                continue
            break
        if vote is None:
            continue
        await gum_instance.add_suggestion_feedback(
            title=outcome.suggestion.title,
            vote=vote,
            description=outcome.suggestion.description,
        )
        recorded.append({"title": outcome.suggestion.title, "vote": vote})
        out("Recorded ✓ approved" if vote == "up" else "Recorded ✗ rejected")
    return recorded


async def cmd_execute(args) -> None:
    """Run the execution bridge and (optionally) review its drafts for approval.

    Invoking this command *is* the explicit opt-in the bridge requires, so it
    builds a Gumbo with execution enabled even though it is default-OFF everywhere
    else. Each surface-worthy suggestion is risk-gated and, if it clears the gate,
    dispatched to the sandboxed agent; the results are drafts held for approval.
    With ``--review`` the user approves/rejects each draft and the decision flows
    back through the existing suggestion-feedback plumbing.
    """
    from gum.gumbo import Gumbo

    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()  # also ensures the feedback table exists

    gumbo = Gumbo(g, execution_enabled=True)
    outcomes = await gumbo.execute(getattr(args, "focus", None))
    if not outcomes:
        print("No surface-worthy suggestions to execute right now.")
        return

    dispatched = sum(1 for o in outcomes if o.dispatched)
    print(
        f"\n{len(outcomes)} suggestion(s) considered; {dispatched} dispatched to the agent."
    )
    await review_outcomes(g, outcomes, interactive=getattr(args, "review", False))


def cmd_mcp(args) -> None:
    """Serve the GUM over MCP (stdio) for a local executing agent.

    stdio is the MCP transport: the client (Claude Desktop, Codex, …) launches
    this process and speaks JSON-RPC over stdin/stdout, so NOTHING may be printed
    to stdout here — the DB is connected lazily inside the server's own loop and
    ``run()`` owns the event loop.
    """
    from gum.mcp_server import run_stdio

    g = gum(_user_name(args) or "default", _text_model(args))
    sanitize = not getattr(args, "no_sanitize", False)
    run_stdio(g, sanitize=sanitize)


def cmd_rehydrate(args) -> None:
    """Restore real values for pseudo-IDs in an agent-produced artifact.

    This is the local, trusted tail of the sanitized-context workflow: the MCP
    server hands out pseudonymized context ([PERSON_1], [ORG_1], …), a frontier
    model drafts something that still carries those placeholders, and this swaps
    them back for real names so the *user* gets a usable document. It is pure
    lookup against the local entity map — no model loads — so it is fast and works
    without the [sanitize] extra.

    By design it writes to a file (or stdout for stdin input), never leaking the
    restored PII back through a channel a frontier model reads: only a count of
    substitutions is reported.
    """
    from gum.sanitize import find_pseudo_ids, get_sanitizer

    src = args.input
    if src and src != "-":
        with open(os.path.expanduser(src), "r", encoding="utf-8") as fh:
            text = fh.read()
        dest = args.output or src
    else:
        text = sys.stdin.read()
        dest = args.output  # None → stdout

    restored, n = get_sanitizer().rehydrate(text)

    if dest:
        with open(os.path.expanduser(dest), "w", encoding="utf-8") as fh:
            fh.write(restored)
        # Status only — never the restored PII — so this is safe to run from an
        # agent shell without re-exposing what sanitization protected.
        print(f"Rehydrated {n} pseudo-ID(s) → {dest}", file=sys.stderr)
    else:
        sys.stdout.write(restored)

    # Any pseudo-ID still present after rehydration had no entry in the local
    # entity map — typically one the frontier model invented — so it stays as a
    # placeholder in the user's artifact. Flag it so the leftover isn't shipped
    # silently. Pseudo-IDs are not PII, so naming them on stderr is safe.
    leftover = find_pseudo_ids(restored)
    if leftover:
        print(
            f"Warning: {len(leftover)} pseudo-ID(s) could not be restored and "
            f"remain in the output: {', '.join(leftover)}. They have no entry in "
            "the local entity map (the model may have invented them); check them "
            "by hand.",
            file=sys.stderr,
        )


def cmd_reset_cache(args) -> None:
    cache_dir = os.path.expanduser("~/.cache/gum/")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Deleted cache directory: {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")


# ── entry point ───────────────────────────────────────────────────────────────
def cli() -> None:
    parser, args = parse_args()
    command = args.command

    if command in (None,):
        parser.print_help()
        return

    if command == "start":
        cmd_start(args)
    elif command == "stop":
        cmd_stop(args)
    elif command == "reset-cache":
        cmd_reset_cache(args)
    elif command == "status":
        asyncio.run(cmd_status(args))
    elif command == "_run":
        asyncio.run(cmd_run(args))
    elif command == "query":
        asyncio.run(cmd_query(args))
    elif command == "recent":
        asyncio.run(cmd_recent(args))
    elif command == "agenda":
        asyncio.run(cmd_agenda(args))
    elif command == "observations":
        asyncio.run(cmd_observations(args))
    elif command == "review":
        asyncio.run(cmd_review(args))
    elif command == "execute":
        asyncio.run(cmd_execute(args))
    elif command == "mcp":
        cmd_mcp(args)
    elif command == "rehydrate":
        cmd_rehydrate(args)
    elif command == "tray":
        from gum.tray import run as run_tray
        run_tray()  # runs the AppKit event loop on the main thread
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
