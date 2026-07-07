# daemon.py
#
# Lightweight process manager so the GUM can be spun up and shut down on
# demand (`gum start` / `gum stop` / `gum status`). The listening process is
# launched detached, with its PID and logs recorded under the GUM data
# directory. No third-party dependency — just a pidfile + POSIX signals.

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

DEFAULT_DATA_DIR = "~/.cache/gum"


def _data_dir(data_directory: str = DEFAULT_DATA_DIR) -> Path:
    path = Path(os.path.expanduser(data_directory))
    path.mkdir(parents=True, exist_ok=True)
    return path


def pid_path(data_directory: str = DEFAULT_DATA_DIR) -> Path:
    return _data_dir(data_directory) / "gum.pid"


def log_path(data_directory: str = DEFAULT_DATA_DIR) -> Path:
    return _data_dir(data_directory) / "gum.log"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(data_directory: str = DEFAULT_DATA_DIR) -> int | None:
    """Return the PID of the running daemon, or None. Cleans up stale pidfiles."""
    p = pid_path(data_directory)
    if not p.exists():
        return None
    try:
        pid = int(p.read_text().strip())
    except (ValueError, OSError):
        p.unlink(missing_ok=True)
        return None
    if _pid_alive(pid):
        return pid
    p.unlink(missing_ok=True)
    return None


def is_running(data_directory: str = DEFAULT_DATA_DIR) -> bool:
    return read_pid(data_directory) is not None


def start(cmd: list[str], data_directory: str = DEFAULT_DATA_DIR) -> int:
    """Launch *cmd* as a detached background process, recording its PID.

    Returns the PID. Raises RuntimeError if a daemon is already running.
    """
    existing = read_pid(data_directory)
    if existing is not None:
        raise RuntimeError(f"GUM is already running (pid {existing}).")

    logf = open(log_path(data_directory), "a", buffering=1)
    logf.write(f"\n=== GUM daemon starting: {' '.join(cmd)} ===\n")
    logf.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # detach from this shell / session
        close_fds=True,
    )
    pid_path(data_directory).write_text(str(proc.pid))
    return proc.pid


def stop(data_directory: str = DEFAULT_DATA_DIR, timeout: float = 15.0) -> bool:
    """Stop the running daemon. Returns True if a process was stopped."""
    pid = read_pid(data_directory)
    if pid is None:
        return False

    os.kill(pid, signal.SIGTERM)

    # Wait for graceful shutdown (the listener flushes observers via __aexit__).
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.25)
    else:
        # Escalate if it did not exit in time.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    pid_path(data_directory).unlink(missing_ok=True)
    return True
