#!/usr/bin/env python
"""Summarize what the user did and what was learned about them the previous day.

Samples the observations and high-confidence propositions GUM recorded for the prior
calendar day (Eastern time), feeds them *raw* to the local text model to write a short
markdown reflection, then pseudonymizes that OUTPUT before writing it. Because all
inference is local (Ollama), raw PII never leaves the machine; sanitizing the written
file — which may sync off-device — is the actual privacy boundary. The result is
written to ``tmp/daily-summary-YYYY-MM-DD.md`` in the repo.

Designed to be run unattended from cron at 08:00 daily. All inference stays local
(Ollama), matching GUM's local-first guarantee.

Run:
    /Users/sauvik/Code/gum/.venv/bin/python scripts/daily_summary.py
    # optional: summarize a specific day instead of yesterday
    ... scripts/daily_summary.py --date 2026-07-06
"""

from __future__ import annotations

# Load the repo's .env (USER_NAME, MODEL_NAME, GUM_SANITIZE_*, HF_TOKEN, …). Load
# by absolute path off this file's location, not the cwd, so the script behaves
# identically whether launched from the repo, from $HOME under cron, or elsewhere.
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import argparse
import asyncio
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from gum.gum import gum
from gum.llm import ensure_capped_model, resolve_num_ctx
from gum.sanitize import get_sanitizer

# Observations are stored in UTC; days are reckoned in Eastern time to match the
# `gum observations --date` CLI view.
EASTERN = ZoneInfo("America/New_York")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tmp"

# Records the first day this job became responsible for producing summaries, so
# backfill never reaches back before the feature was installed (which would mint a
# huge historical backlog on first run). Created on the first run, then only used
# to bound the catch-up window.
ANCHOR_PATH = OUT_DIR / ".daily-summary-anchor"

logger = logging.getLogger("gum.daily_summary")

# Chars per map chunk = one model call. The text model runs at a 32K-token context
# (~130k chars at ~4 chars/token); 80k leaves room for prompt scaffolding and output.
# Bigger chunks = fewer (costly) forward passes.
MAP_CHUNK_CHARS = 80_000

# --- Sampling budget --------------------------------------------------------- #
# The text model is input-bound (~200-650 prompt tokens/sec), so total runtime scales
# with how much observation text we feed it. To keep the whole summary within a time
# budget (~8-10 min run off-hours), the day is split into chronological buckets and we
# randomly sub-sample observations + high-confidence propositions within each bucket,
# bounding the total input. Bucketing keeps morning/midday/evening each represented;
# random sampling within a bucket avoids systematic bias; screen frames are highly
# redundant so this costs little fidelity. All tunable via env.
SUMMARY_TIME_BUCKETS = int(os.getenv("GUM_SUMMARY_TIME_BUCKETS", "8"))
# ~80k chars ≈ one 32B map call ≈ ~2 min on this box uncontended; the reduce adds
# ~1.5 min. 320k → ~4 map chunks → ~8 min map + reduce ≈ within a 10-min budget. Raise
# for more fidelity (slower), lower for a tighter cap.
MAX_OBS_CHARS = int(os.getenv("GUM_SUMMARY_MAX_OBS_CHARS", "320000"))
MAX_PROPS = int(os.getenv("GUM_SUMMARY_MAX_PROPS", "150"))
MIN_CONFIDENCE = int(os.getenv("GUM_SUMMARY_MIN_CONFIDENCE", "7"))
# Cap on rows pulled per bucket to sample from (bounds peak memory on a huge bucket).
BUCKET_FETCH_CAP = int(os.getenv("GUM_SUMMARY_BUCKET_FETCH", "4000"))


# --------------------------------------------------------------------------- #
# Time range                                                                  #
# --------------------------------------------------------------------------- #
def eastern_day_bounds(day) -> tuple[datetime, datetime]:
    """Return the [start, end) UTC bounds of the Eastern-time calendar *day*."""
    start = datetime(day.year, day.month, day.day, tzinfo=EASTERN)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def parse_day(date_str: str | None):
    """Return the target date: an explicit --date, else yesterday (Eastern)."""
    if not date_str:
        return datetime.now(EASTERN).date() - timedelta(days=1)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise SystemExit(f"Could not parse date '{date_str}'. Use YYYY-MM-DD or M/D/YYYY.")


# --------------------------------------------------------------------------- #
# Sanitization                                                                #
# --------------------------------------------------------------------------- #
async def scrub(text: str) -> str:
    """Pseudonymize PII in *text* off the event loop (inference is blocking)."""
    if not text:
        return text
    return await asyncio.to_thread(get_sanitizer().sanitize, text)


async def scrub_output(text: str, user: str) -> str:
    """Pseudonymize PII in the model's OUTPUT before it is written to disk.

    Inputs to the local summarizer are raw (nothing leaves the machine), so only the
    written summary — which may sync off-device — is sanitized. This is one small
    sanitizer call on the final text instead of hundreds on every input. The user's
    own name is preserved (it is their summary); everyone else's PII is pseudonymized.
    """
    if not text or not user or user == "the user":
        return await scrub(text)
    # Shield the user's own name with a private-use sentinel the NER won't flag, then
    # restore it after sanitizing everyone else.
    sentinel = "\ue000"  # private-use char; not a name, never appears in model output
    scrubbed = await scrub(text.replace(user, sentinel))
    return scrubbed.replace(sentinel, user)


# --------------------------------------------------------------------------- #
# Inference helpers                                                           #
# --------------------------------------------------------------------------- #
async def complete(client, model: str, prompt: str) -> str:
    """One plain free-text chat completion against the local text model."""
    rsp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return (rsp.choices[0].message.content or "").strip()


def _activity_prompt(user: str, day_label: str, chunk_no: int, chunk: str) -> str:
    """The 'map' prompt: distill one context-sized window of observations.

    Observations are raw here — PII is redacted from the final written summary, not
    the local model's inputs — so the model sees real names/context for grounding.
    """
    return (
        f"Below is a chronological stream of on-screen activity observations for "
        f"{user} on {day_label}. Each entry is a timestamped transcription of what "
        f"was on screen.\n\n"
        f"Write a tight bulleted list of the concrete things {user} was doing in "
        f"this window: tasks, projects, apps/websites, topics read or written, "
        f"people communicated with, and any notable decisions or accomplishments. "
        f"Be factual and specific; do not invent details; omit idle/ambiguous "
        f"screens.\n\n"
        f"--- OBSERVATIONS (chunk {chunk_no}) ---\n{chunk}"
    )


def _day_buckets(start_utc, end_utc, n: int) -> list[tuple]:
    """Split ``[start, end)`` into *n* equal chronological sub-windows."""
    n = max(1, n)
    span = (end_utc - start_utc) / n
    return [(start_utc + span * i, start_utc + span * (i + 1)) for i in range(n)]


async def sample_observations(g, start_utc, end_utc, rng) -> list[tuple]:
    """Randomly sub-sample observations across the day within a char budget.

    The day is split into chronological buckets so morning/midday/evening are each
    represented; within a bucket observations are shuffled and taken until that
    bucket's share of :data:`MAX_OBS_CHARS` is filled. Exact-duplicate frames are
    dropped. Returns ``(created_at, content)`` pairs in chronological order.
    """
    buckets = _day_buckets(start_utc, end_utc, SUMMARY_TIME_BUCKETS)
    per_bucket = MAX_OBS_CHARS / len(buckets)
    seen: set[int] = set()
    kept: list[tuple] = []
    for b0, b1 in buckets:
        rows = await g.recent_observations(
            limit=BUCKET_FETCH_CAP, ascending=True, start_time=b0, end_time=b1
        )
        rng.shuffle(rows)
        used = 0
        for o in rows:
            h = hash(o.content)
            if h in seen:  # drop repeated screen frames (day-wide)
                continue
            if used and used + len(o.content) > per_bucket:
                break
            seen.add(h)
            kept.append((o.created_at, o.content))
            used += len(o.content)
    kept.sort(key=lambda x: x[0])
    return kept


async def sample_propositions(g, start_utc, end_utc, rng) -> list:
    """Randomly sub-sample high-confidence propositions across the day.

    Same chronological bucketing as observations; within a bucket only propositions
    at or above :data:`MIN_CONFIDENCE` are kept, shuffled, and capped to that bucket's
    share of :data:`MAX_PROPS`. Returns propositions in chronological order.
    """
    buckets = _day_buckets(start_utc, end_utc, SUMMARY_TIME_BUCKETS)
    per_bucket = max(1, MAX_PROPS // len(buckets))
    kept = []
    for b0, b1 in buckets:
        props = await g.recent(limit=BUCKET_FETCH_CAP, start_time=b0, end_time=b1)
        hi = [p for p in props if (p.confidence or 0) >= MIN_CONFIDENCE]
        rng.shuffle(hi)
        kept.extend(hi[:per_bucket])
    kept.sort(key=lambda p: p.created_at)
    return kept


async def map_activity_notes(g, model, user, day_label, obs_sampled) -> str:
    """Distill the sampled observations into activity notes, chunk by chunk (the
    'map'). Input is already bounded by sampling, so this is a handful of model calls.
    """
    if not obs_sampled:
        return "(no screen observations were recorded)"
    notes: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    chunk_no = 0

    async def flush() -> None:
        nonlocal buf, buf_chars, chunk_no
        if not buf:
            return
        chunk_no += 1
        logger.info("Summarizing observation chunk %d (%d chars)", chunk_no, buf_chars)
        notes.append(await complete(g.client, model, _activity_prompt(
            user, day_label, chunk_no, "\n\n".join(buf))))
        buf, buf_chars = [], 0

    for created_at, content in obs_sampled:  # already chronological
        ts = created_at.replace(tzinfo=timezone.utc).astimezone(EASTERN).strftime("%H:%M")
        entry = f"[{ts}] {content}"  # raw; the final summary is sanitized, not the input
        # Hard-cap one giant observation so it can't overflow the model context.
        if len(entry) > MAP_CHUNK_CHARS:
            entry = entry[:MAP_CHUNK_CHARS] + "\n…(truncated)"
        if buf and buf_chars + len(entry) > MAP_CHUNK_CHARS:
            await flush()
        buf.append(entry)
        buf_chars += len(entry) + 2
    await flush()
    logger.info("Summarized %d sampled observations in %d chunk(s)", len(obs_sampled), chunk_no)
    return "\n".join(notes)


# --------------------------------------------------------------------------- #
# One day                                                                     #
# --------------------------------------------------------------------------- #
def summary_path(day) -> Path:
    """Output file for *day*'s summary."""
    return OUT_DIR / f"daily-summary-{day.isoformat()}.md"


async def summarize_day(g, model: str, user: str, day, *, force: bool) -> bool:
    """Generate and write the summary for a single *day*.

    Returns True if a file was written, False if it was skipped because one
    already exists (and *force* is off). Writing is idempotent — a day's file is
    overwritten wholesale — so re-running is always safe.
    """
    out_path = summary_path(day)
    if out_path.exists() and not force:
        logger.info("Summary for %s already exists; skipping (%s)", day, out_path)
        return False

    start_utc, end_utc = eastern_day_bounds(day)
    day_label = day.strftime("%A, %B %-d, %Y")

    # Deterministic per-day RNG so re-running a day reproduces the same sample.
    rng = random.Random(day.toordinal())

    obs_total, _ = await g.count_observations(start_time=start_utc, end_time=end_utc)
    obs_sampled = await sample_observations(g, start_utc, end_utc, rng)
    props_sampled = await sample_propositions(g, start_utc, end_utc, rng)
    sampled = len(obs_sampled) < obs_total
    logger.info(
        "Day %s: sampled %d of %d observations and %d high-confidence proposition(s)",
        day, len(obs_sampled), obs_total, len(props_sampled),
    )

    prop_lines: list[str] = []
    for p in props_sampled:  # raw; sanitized in the final output, not here
        line = f"- {p.text}"
        if p.confidence is not None:
            line += f" (confidence {p.confidence}/10)"
        if p.reasoning:
            line += f"\n  reasoning: {p.reasoning}"
        prop_lines.append(line)
    props_block = "\n".join(prop_lines) if prop_lines else "(no high-confidence propositions)"

    # Map: distill the sampled observations into activity notes.
    activity_notes = await map_activity_notes(g, model, user, day_label, obs_sampled)

    sample_note = (
        f" Sampled {len(obs_sampled)} of {obs_total} observations and "
        f"{len(props_sampled)} high-confidence propositions to stay within the time "
        f"budget." if sampled else ""
    )
    header = (
        f"# Daily Summary — {user}\n"
        f"### {day_label}\n\n"
        f"_Generated {datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M %Z')} from "
        f"{obs_total} observation(s) and {len(props_sampled)} proposition(s)."
        f"{sample_note} Other people's PII is pseudonymized in this summary._\n"
    )

    if obs_total == 0 and not props_sampled:
        out_path.write_text(header + "\n_No activity was recorded on this day._\n")
        logger.info("No activity for %s; wrote stub to %s", day, out_path)
        return True

    # Reduce: write the final reflection from the activity notes + propositions.
    final_prompt = (
        f"You are writing a concise end-of-day reflection about {user} for "
        f"{day_label}. You are given (A) activity notes distilled from their screen "
        f"activity, and (B) propositions the system inferred about them.\n\n"
        f"Write clean GitHub-flavored markdown with EXACTLY these two sections and no "
        f"top-level title:\n\n"
        f"## What {user} did\n"
        f"A short narrative followed by grouped bullet points of the concrete things "
        f"they worked on and accomplished, roughly in the order of the day.\n\n"
        f"## What we learned about {user}\n"
        f"The most important, well-supported insights about their interests, habits, "
        f"working style, relationships, and goals — grounded in the propositions and "
        f"activity. Prefer a few high-confidence insights over many shallow ones.\n\n"
        f"Be specific and factual; do not speculate beyond the evidence.\n\n"
        f"=== (A) ACTIVITY NOTES ===\n{activity_notes}\n\n"
        f"=== (B) INFERRED PROPOSITIONS ===\n{props_block}\n"
    )
    summary = await complete(g.client, model, final_prompt)
    # Sanitize the OUTPUT only: raw PII went solely to the local model; the written
    # file (which may sync off-device) is pseudonymized. One small call, not hundreds.
    summary = await scrub_output(summary, user)

    out_path.write_text(header + "\n" + summary + "\n")
    logger.info("Wrote daily summary to %s", out_path)
    print(out_path)
    return True


# --------------------------------------------------------------------------- #
# Day selection (yesterday + backfill of missed days)                         #
# --------------------------------------------------------------------------- #
def read_anchor(default):
    """Return the backfill floor date, creating the anchor file on first run.

    On the very first run the anchor is set to *default* (yesterday), so a fresh
    install only ever produces "yesterday" — never a backlog of pre-install days.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return datetime.strptime(ANCHOR_PATH.read_text().strip(), "%Y-%m-%d").date()
    except (FileNotFoundError, ValueError):
        ANCHOR_PATH.write_text(default.isoformat() + "\n")
        logger.info("Initialized backfill anchor at %s", default)
        return default


def missing_days(yesterday, anchor, backfill_days: int, exists) -> list:
    """Pure window logic: days in [max(anchor, yesterday-(N-1)), yesterday] for
    which ``exists(day)`` is False, oldest → newest.

    The window is capped at *backfill_days* so a long shutdown can't trigger an
    unbounded catch-up, and floored at *anchor* so it never predates install.
    """
    oldest = max(anchor, yesterday - timedelta(days=max(1, backfill_days) - 1))
    out, d = [], oldest
    while d <= yesterday:
        if not exists(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def days_to_generate(backfill_days: int) -> list:
    """Days needing a summary: yesterday plus any missed day in the backfill
    window, skipping days already produced (or covered while merely asleep)."""
    yesterday = datetime.now(EASTERN).date() - timedelta(days=1)
    anchor = read_anchor(yesterday)
    return missing_days(yesterday, anchor, backfill_days, lambda d: summary_path(d).exists())


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
async def run(date_str: str | None, backfill_days: int) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # An explicit --date targets exactly that day and always (re)generates it;
    # otherwise do yesterday plus any missed days in the backfill window.
    if date_str:
        days, force = [parse_day(date_str)], True
    else:
        days, force = days_to_generate(backfill_days), False

    if not days:
        logger.info("Nothing to do — all recent days already summarized.")
        return 0

    user = os.getenv("USER_NAME") or "the user"
    base_model = os.getenv("MODEL_NAME") or "qwen2.5-coder:32b"
    # Reuse the resident context-capped model GUM runs (small KV cache) instead of
    # loading the base model at its full 128K context. Falls back to base if the
    # ollama CLI isn't reachable. See gum/llm.py.
    model = ensure_capped_model(base_model, resolve_num_ctx("gum"), logger=logger)

    # One DB connection + one sanitizer (consistent pseudo-IDs across all days).
    g = gum(user, base_model)
    await g.connect_db()

    if len(days) > 1:
        logger.info("Backfilling %d day(s): %s", len(days), ", ".join(map(str, days)))

    # Each day is independent; one failure shouldn't sink the rest of the backfill.
    failures = 0
    for day in days:
        try:
            await summarize_day(g, model, user, day, force=force)
        except Exception:
            failures += 1
            logger.exception("Failed to summarize %s", day)
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize the user's prior day(s) from GUM.")
    ap.add_argument(
        "--date",
        help="Summarize exactly this day (YYYY-MM-DD or M/D/YYYY), regenerating it. "
        "Omit to do yesterday plus any missed days in the backfill window.",
    )
    ap.add_argument(
        "--backfill-days",
        type=int,
        default=int(os.getenv("GUM_SUMMARY_BACKFILL_DAYS", "7")),
        help="How far back to catch up missed summaries (default 7). Ignored with --date.",
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args.date, args.backfill_days)))


if __name__ == "__main__":
    main()
