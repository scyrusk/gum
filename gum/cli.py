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

    p_recent = sub.add_parser("recent", help="List the most recent propositions")
    p_recent.add_argument("--limit", "-l", type=int, default=10)
    p_recent.add_argument("--user-name", "-u", type=str)
    p_recent.add_argument("--text-model", "-m", type=str)

    p_obs = sub.add_parser("observations", help="List raw observations (all of a day with --date)")
    p_obs.add_argument("--limit", "-l", type=int, default=None,
                       help="Max results (default 10; all-day when --date is set)")
    p_obs.add_argument("--date", "-d", type=str,
                       help="Only observations from this Eastern-time day, e.g. 7/7/2026 or 2026-07-07")
    p_obs.add_argument("--full", action="store_true", help="Print full content instead of a preview")
    p_obs.add_argument("--user-name", "-u", type=str)
    p_obs.add_argument("--text-model", "-m", type=str)

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

    print(f"Listening as {user} — text={text_model}, vision={vision_model}, api=http://{host}:{port}")

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
        server = build_server(gum_instance, host=host, port=port)
        api_task = asyncio.create_task(server.serve())
        await stop_event.wait()
        print("Shutting down GUM…")
        server.should_exit = True
        await api_task


async def cmd_query(args) -> None:
    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()
    result = await g.query(args.query, limit=args.limit, mode=args.mode)
    print(f"\nFound {len(result)} results:")
    for prop, score in result:
        print(f"\nProposition: {prop.text}")
        if prop.reasoning:
            print(f"Reasoning: {prop.reasoning}")
        if prop.confidence is not None:
            print(f"Confidence: {prop.confidence:.2f}")
        print(f"Relevance Score: {score:.2f}")
        print("-" * 80)


async def cmd_recent(args) -> None:
    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()
    props = await g.recent(limit=args.limit)
    print(f"\nRecent {len(props)} propositions:")
    for p in props:
        print(f"\nProposition: {p.text}")
        if p.reasoning:
            print(f"Reasoning: {p.reasoning}")
        if p.confidence is not None:
            print(f"Confidence: {p.confidence:.2f}")
        print(f"Created At: {p.created_at}")
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

    g = gum(_user_name(args) or "default", _text_model(args))
    await g.connect_db()
    obs = await g.recent_observations(limit=limit, start_time=start_utc, end_time=end_utc)

    if args.date:
        obs = list(reversed(obs))  # chronological for a day view
        day = start_utc.astimezone(EASTERN).strftime("%m/%d/%Y")
        print(f"\n{len(obs)} observations on {day} (Eastern):")
    else:
        print(f"\nRecent {len(obs)} observations:")

    for o in obs:
        ts = o.created_at.replace(tzinfo=timezone.utc).astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z")
        content = o.content if args.full else o.content.replace("\n", " ")[:300]
        print(f"\n[{o.observer_name}] {ts}  (id={o.id})")
        print(content)
        if not args.full and len(o.content) > 300:
            print("… (use --full to see everything)")
        print("-" * 80)


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
    elif command == "observations":
        asyncio.run(cmd_observations(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
