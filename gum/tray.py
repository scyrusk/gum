# tray.py
#
# A tiny macOS menu-bar (status-bar) companion for the GUM. It is a thin,
# native front-end over the pieces that already exist:
#
#   • start / stop / status  → the same `gum start` / `gum stop` daemon manager
#     used from the terminal (so provisioning, .env handling and the pidfile all
#     behave identically to the CLI).
#   • search & recent props  → the localhost REST API served inside the running
#     daemon (gum/api.py). The tray is exactly the "any local application" that
#     API was built for, so reads always reflect the live model.
#
# Nothing here re-implements GUM behaviour; it only orchestrates the CLI and the
# REST API. Requires `rumps` (install with `pip install "gum-ai[tray]"`).

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

from dotenv import find_dotenv, load_dotenv

from gum import daemon

# Load .env so USER_NAME / GUM_API_* resolve exactly as they do for the CLI,
# regardless of which directory the tray was launched from.
load_dotenv(find_dotenv(usecwd=True))

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8422

# How many propositions to surface in the "Recent" submenu, and how long a menu
# label may get before we truncate it (full text is shown on click).
RECENT_LIMIT = 8
LABEL_MAX = 72

REFRESH_SECONDS = 5


def _api_base() -> str:
    host = os.getenv("GUM_API_HOST") or DEFAULT_API_HOST
    port = os.getenv("GUM_API_PORT") or str(DEFAULT_API_PORT)
    return f"http://{host}:{port}"


def _user_name() -> str | None:
    return os.getenv("USER_NAME")


def _truncate(text: str, limit: int = LABEL_MAX) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _api_get(path: str, params: dict | None = None, timeout: float = 8.0) -> dict:
    """GET a JSON payload from the local GUM API. Raises urllib.error on failure."""
    url = _api_base() + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class GumTray:
    """Menu-bar controller. Imports rumps lazily so `import gum.tray` stays cheap."""

    def __init__(self) -> None:
        import rumps

        self._rumps = rumps
        self.app = rumps.App("GUM", title="🧠", quit_button="Quit GUM")

        # Persistent menu items (rebuilt contents, stable identities).
        self._status_item = rumps.MenuItem("GUM: …")
        self._start_item = rumps.MenuItem("Start GUM", callback=self._on_start)
        self._stop_item = rumps.MenuItem("Stop GUM", callback=self._on_stop)
        self._recent_menu = rumps.MenuItem("Recent Propositions")

        self.app.menu = [
            self._status_item,
            None,
            self._start_item,
            self._stop_item,
            None,
            rumps.MenuItem("Search GUM…", callback=self._on_search),
            self._recent_menu,
            None,
            rumps.MenuItem("Open Review UI", callback=self._on_review),
            rumps.MenuItem("Open Logs", callback=self._on_logs),
        ]

        # Signature of the last-rendered recents, so we only rebuild on change.
        self._recent_sig: tuple | None = None

        # Refresh once immediately, then on a timer.
        self._timer = rumps.Timer(self._refresh, REFRESH_SECONDS)
        self._timer.start()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def run(self) -> None:
        self.app.run()

    # ── periodic state refresh ───────────────────────────────────────────────
    def _refresh(self, _timer=None) -> None:
        running = daemon.is_running()
        pid = daemon.read_pid() if running else None

        self.app.title = "🧠" if running else "💤"
        self._start_item.set_callback(None if running else self._on_start)
        self._stop_item.set_callback(self._on_stop if running else None)

        if running:
            try:
                health = _api_get("/health", timeout=2.0)
                mode = " · sanitized" if health.get("sanitized") else ""
                self._status_item.title = f"GUM running (pid {pid}){mode}"
            except Exception:
                # Daemon is up but API not ready yet (still booting observers).
                self._status_item.title = f"GUM starting (pid {pid})…"
            self._refresh_recent()
        else:
            self._status_item.title = "GUM stopped"
            self._clear_recent("Start GUM to see propositions")

    def _reset_recent(self) -> None:
        # A rumps submenu only allocates its NSMenu on first .add(); clearing an
        # empty one would dereference None, so guard on the private handle.
        if self._recent_menu._menu is not None:
            self._recent_menu.clear()

    def _clear_recent(self, placeholder: str) -> None:
        if self._recent_sig == ("placeholder", placeholder):
            return
        self._reset_recent()
        item = self._rumps.MenuItem(placeholder)
        item.set_callback(None)  # disabled/greyed
        self._recent_menu.add(item)
        self._recent_sig = ("placeholder", placeholder)

    def _refresh_recent(self) -> None:
        try:
            data = _api_get("/recent", {"limit": RECENT_LIMIT}, timeout=3.0)
            props = data.get("results", [])
        except Exception:
            self._clear_recent("(could not reach GUM API)")
            return

        sig = tuple(p.get("id") for p in props)
        if sig == self._recent_sig:
            return  # unchanged; avoid churning the menu
        self._recent_sig = sig

        self._reset_recent()
        if not props:
            self._clear_recent("No propositions yet")
            return
        for p in props:
            label = _truncate(p.get("text", ""))
            item = self._rumps.MenuItem(label, callback=lambda _s, prop=p: self._show_prop(prop))
            self._recent_menu.add(item)

    # ── actions ──────────────────────────────────────────────────────────────
    def _on_start(self, _sender=None) -> None:
        user = _user_name()
        if not user:
            self._rumps.alert(
                "USER_NAME not set",
                "Set USER_NAME in your .env (or launch the tray from a directory "
                "with one) before starting GUM.",
            )
            return
        # Mirror the CLI exactly: `gum start -u <user>` handles model provisioning
        # and detaches the daemon itself. First run may take a while, so fire and
        # forget — the status timer reflects the new state when it comes up.
        self._status_item.title = "GUM starting…"
        subprocess.Popen(
            [sys.executable, "-m", "gum.cli", "start", "-u", user],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _on_stop(self, _sender=None) -> None:
        self._status_item.title = "GUM stopping…"
        subprocess.Popen(
            [sys.executable, "-m", "gum.cli", "stop"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _on_search(self, _sender=None) -> None:
        if not daemon.is_running():
            self._rumps.alert("GUM is not running", "Start GUM first, then search.")
            return
        window = self._rumps.Window(
            title="Search GUM",
            message="Keyword search over what GUM has learned about you:",
            default_text="",
            ok="Search",
            cancel="Cancel",
            dimensions=(340, 24),
        )
        response = window.run()
        if not response.clicked:
            return
        query = response.text.strip()
        try:
            data = _api_get("/query", {"q": query, "limit": 10}, timeout=15.0)
            results = data.get("results", [])
        except Exception as exc:
            self._rumps.alert("Search failed", str(exc))
            return

        if not results:
            self._rumps.alert("No results", f"Nothing matched “{query}”.")
            return

        lines = []
        for i, p in enumerate(results, 1):
            conf = p.get("confidence")
            conf_s = f"  (confidence {conf}/10)" if conf is not None else ""
            lines.append(f"{i}. {p.get('text', '').strip()}{conf_s}")
            if p.get("reasoning"):
                lines.append(f"    ↳ {p['reasoning'].strip()}")
            lines.append("")
        body = "\n".join(lines).rstrip()

        # A read-only-ish viewer: pre-fill a sizeable text box with the results.
        self._rumps.Window(
            title=f"GUM · {len(results)} result(s) for “{_truncate(query, 40)}”",
            message="",
            default_text=body,
            ok="Close",
            cancel=None,
            dimensions=(560, 360),
        ).run()

    def _show_prop(self, prop: dict) -> None:
        parts = [prop.get("text", "").strip()]
        conf = prop.get("confidence")
        if conf is not None:
            parts.append(f"\nConfidence: {conf}/10")
        if prop.get("created_at"):
            parts.append(f"Created: {prop['created_at']}")
        if prop.get("reasoning"):
            parts.append(f"\nReasoning:\n{prop['reasoning'].strip()}")
        self._rumps.alert(title="Proposition", message="\n".join(parts))

    def _on_review(self, _sender=None) -> None:
        if not daemon.is_running():
            self._rumps.alert("GUM is not running", "Start GUM first to open the review UI.")
            return
        # The daemon's API already serves the review page at "/".
        webbrowser.open(_api_base() + "/")

    def _on_logs(self, _sender=None) -> None:
        subprocess.Popen(["open", str(daemon.log_path())])


def run() -> None:
    """Entry point for `gum tray`."""
    try:
        import rumps  # noqa: F401
    except ImportError:
        print(
            "The menu-bar app needs `rumps`. Install it with:\n"
            '    pip install "gum-ai[tray]"\n'
            "  (or: pip install rumps)"
        )
        sys.exit(1)
    GumTray().run()
