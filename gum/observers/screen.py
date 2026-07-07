from __future__ import annotations
###############################################################################
# Imports                                                                     #
###############################################################################

# — Standard library —
import base64
import io
import logging
import os
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional

import asyncio

# — Third-party —
import mss
import Quartz
from PIL import Image, ImageChops
from pynput import mouse           # still synchronous
from shapely.geometry import box
from shapely.ops import unary_union

# — Local —
from .observer import Observer
from ..schemas import Update
from ..llm import make_client, inference_semaphore

# — Local —
from gum.prompts.screen import TRANSCRIPTION_PROMPT, SUMMARY_PROMPT, COMBINED_PROMPT

###############################################################################
# Window‑geometry helpers                                                     #
###############################################################################


def _get_global_bounds() -> tuple[float, float, float, float]:
    """Return a bounding box enclosing **all** physical displays.

    Returns
    -------
    (min_x, min_y, max_x, max_y) tuple in Quartz global coordinates.
    """
    err, ids, cnt = Quartz.CGGetActiveDisplayList(16, None, None)
    if err != Quartz.kCGErrorSuccess:  # pragma: no cover (defensive)
        raise OSError(f"CGGetActiveDisplayList failed: {err}")

    min_x = min_y = float("inf")
    max_x = max_y = -float("inf")
    for did in ids[:cnt]:
        r = Quartz.CGDisplayBounds(did)
        x0, y0 = r.origin.x, r.origin.y
        x1, y1 = x0 + r.size.width, y0 + r.size.height
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x1), max(max_y, y1)
    return min_x, min_y, max_x, max_y


def _get_visible_windows() -> List[tuple[dict, float]]:
    """List *onscreen* windows with their visible‑area ratio.

    Each tuple is ``(window_info_dict, visible_ratio)`` where *visible_ratio*
    is in ``[0.0, 1.0]``.  Internal system windows (Dock, WindowServer, …) are
    ignored.
    """
    _, _, _, gmax_y = _get_global_bounds()

    opts = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListOptionIncludingWindow
    )
    wins = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID)

    occupied = None  # running union of opaque regions above the current window
    result: list[tuple[dict, float]] = []

    for info in wins:
        owner = info.get("kCGWindowOwnerName", "")
        if owner in ("Dock", "WindowServer", "Window Server"):
            continue

        bounds = info.get("kCGWindowBounds", {})
        x, y, w, h = (
            bounds.get("X", 0),
            bounds.get("Y", 0),
            bounds.get("Width", 0),
            bounds.get("Height", 0),
        )
        if w <= 0 or h <= 0:
            continue  # hidden or minimised

        inv_y = gmax_y - y - h  # Quartz→Shapely Y‑flip
        poly = box(x, inv_y, x + w, inv_y + h)
        if poly.is_empty:
            continue

        visible = poly if occupied is None else poly.difference(occupied)
        if not visible.is_empty:
            ratio = visible.area / poly.area
            result.append((info, ratio))
            occupied = poly if occupied is None else unary_union([occupied, poly])

    return result


def _is_app_visible(names: Iterable[str]) -> bool:
    """Return *True* if **any** window from *names* is at least partially visible."""
    targets = set(names)
    return any(
        info.get("kCGWindowOwnerName", "") in targets and ratio > 0
        for info, ratio in _get_visible_windows()
    )

###############################################################################
# Screen observer                                                             #
###############################################################################

class Screen(Observer):
    """Observer that captures and analyzes screen content around user interactions.

    This observer captures screenshots before and after user interactions (mouse movements,
    clicks, and scrolls) and uses GPT-4 Vision to analyze the content. It can also take
    periodic screenshots and skip captures when certain applications are visible.

    Args:
        screenshots_dir (str, optional): Directory to store screenshots. Defaults to "~/.cache/gum/screenshots".
        skip_when_visible (Optional[str | list[str]], optional): Application names to skip when visible.
            Defaults to None.
        transcription_prompt (Optional[str], optional): Custom prompt for transcribing screenshots.
            Defaults to None.
        summary_prompt (Optional[str], optional): Custom prompt for summarizing screenshots.
            Defaults to None.
        model_name (str, optional): GPT model to use for vision analysis. Defaults to "gpt-4o-mini".
        history_k (int, optional): Number of recent screenshots to keep in history. Defaults to 10.
        debug (bool, optional): Enable debug logging. Defaults to False.

    Attributes:
        _CAPTURE_FPS (int): Frames per second for screen capture.
        _DEBOUNCE_SEC (int): Seconds to wait before processing an interaction.
        _MON_START (int): Index of first real display in mss.
    """

    # The capture loop only maintains a recent "before" frame per monitor so
    # that, when a mouse event fires, there is a snapshot of the pre-interaction
    # state. That frame just needs to be sub-second fresh — grabbing every
    # display's full framebuffer 10x/second is wasteful and, on a machine that
    # is simultaneously running local inference, steals CPU and (on Apple
    # Silicon's unified memory) memory bandwidth from the GPU. 2 FPS keeps the
    # before-frame <0.5s old while cutting this idle background load ~5x.
    # Override with GUM_CAPTURE_FPS.
    _CAPTURE_FPS: int = 2
    _DEBOUNCE_SEC: int = 2
    _MON_START: int = 1     # first real display in mss

    # ─────────────────────────────── construction
    def __init__(
        self,
        model_name: str = "qwen2.5vl:7b",
        screenshots_dir: str = "~/.cache/gum/screenshots",
        skip_when_visible: Optional[str | list[str]] = None,
        transcription_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        history_k: int = 2,
        debug: bool = False,
        api_key: str | None = None,
        api_base: str | None = None,
        max_image_dim: int = 1280,
        max_summary_images: int = 4,
    ) -> None:
        """Initialize the Screen observer.
        
        Args:
            screenshots_dir (str, optional): Directory to store screenshots. Defaults to "~/.cache/gum/screenshots".
            skip_when_visible (Optional[str | list[str]], optional): Application names to skip when visible.
                Defaults to None.
            transcription_prompt (Optional[str], optional): Custom prompt for transcribing screenshots.
                Defaults to None.
            summary_prompt (Optional[str], optional): Custom prompt for summarizing screenshots.
                Defaults to None.
            model_name (str, optional): GPT model to use for vision analysis. Defaults to "gpt-4o-mini".
            history_k (int, optional): Number of recent screenshots to keep in history. Defaults to 10.
            debug (bool, optional): Enable debug logging. Defaults to False.
        """
        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

        self._guard = {skip_when_visible} if isinstance(skip_when_visible, str) else set(skip_when_visible or [])

        self.transcription_prompt = transcription_prompt or TRANSCRIPTION_PROMPT
        self.summary_prompt = summary_prompt or SUMMARY_PROMPT
        self.combined_prompt = COMBINED_PROMPT
        self.model_name = model_name

        # Each observation normally costs two sequential vision calls
        # (transcription + summary). On a single local GPU those run back to
        # back, so an observation only surfaces after both complete. Collapsing
        # them into one call ~halves that latency — the dominant, most frequent
        # inference the GUM performs. Default on; set GUM_COMBINE_VISION=0 to
        # fall back to the legacy two-call path if combined-call quality regresses.
        self.combine_vision = os.getenv("GUM_COMBINE_VISION", "1") != "0"

        # Local-VLM cost controls: cap image resolution and the number of
        # images sent per summary call so local inference stays responsive.
        self.max_image_dim = max_image_dim
        self.max_summary_images = max_summary_images

        # Context-frame downscaling. In the combined vision call the LAST image
        # is the current screen (transcription is scoped to it and needs full
        # OCR-grade resolution); every earlier image is temporal context that
        # only feeds the coarse action summary ("opened a menu", "scrolled").
        # VLM prefill cost scales with image pixel *area*, so sending those
        # context frames at a smaller long-edge (e.g. 768 vs 1280) cuts roughly
        # 64% of each context frame's image tokens — a large reduction on the
        # GUM's most frequent inference — with no effect on the transcription
        # and negligible effect on the gross-action summary. Only applied in the
        # combined path (the legacy transcription call reads every frame). Set
        # GUM_CONTEXT_IMAGE_DIM=0 (or >= max_image_dim) to send context frames
        # at full resolution.
        try:
            self.context_image_dim = int(os.getenv("GUM_CONTEXT_IMAGE_DIM", "768"))
        except ValueError:
            self.context_image_dim = 768
        if self.context_image_dim and self.context_image_dim >= self.max_image_dim:
            self.context_image_dim = 0

        # Skip the whole observation (VLM call + emitted update) when an
        # interaction left the screen visually unchanged — the common case for
        # mouse moves/clicks over static content (reading, hovering). The GUM's
        # most frequent inference is this per-interaction vision call, so
        # transcribing an identical frame both burns that inference and injects
        # a no-signal observation that dilutes the downstream proposition batch.
        # A cheap downscaled-grayscale diff gates it; default on, disable with
        # GUM_SKIP_UNCHANGED_FRAMES=0. The gate counts how many thumbnail pixels
        # changed appreciably (see _frames_differ) rather than averaging the
        # whole frame, so a small-but-real edit (a toggled checkbox, a new
        # tooltip) still registers while a static screen stays at zero.
        # GUM_FRAME_DIFF_THRESHOLD tunes how many changed pixels are tolerated
        # before an interaction counts as unchanged (default 3, conservative).
        self.skip_unchanged = os.getenv("GUM_SKIP_UNCHANGED_FRAMES", "1") != "0"
        try:
            self.changed_pixel_tolerance = int(os.getenv("GUM_FRAME_DIFF_THRESHOLD", "3"))
        except ValueError:
            self.changed_pixel_tolerance = 3

        self.debug = debug

        # state shared with worker
        self._frames: Dict[int, Any] = {}
        self._frame_lock = asyncio.Lock()

        self._history: deque[str] = deque(maxlen=max(0, history_k))
        self._pending_event: Optional[dict] = None
        self._debounce_handle: Optional[asyncio.TimerHandle] = None
        # Monotonic timestamp of the most recent mouse event. Drives the
        # capture loop's adaptive backoff (see _capture_interval): the loop only
        # needs to keep a fresh "before" frame while the mouse is active, since
        # observations are only ever produced from mouse events. Seed it as
        # "just active" so startup captures at full rate.
        self._last_input: float = time.monotonic()
        # Local-first inference client (defaults to Ollama; refuses non-local
        # endpoints unless GUM_ALLOW_REMOTE=1). See gum/llm.py.
        self.client = make_client("screen", api_base=api_base, api_key=api_key)

        # call parent
        super().__init__()

    # ─────────────────────────────── tiny sync helpers
    @staticmethod
    def _capture_interval(
        now: float,
        last_input: float,
        active_fps: float,
        idle_fps: float,
        idle_after: float,
    ) -> float:
        """Seconds to wait before the next before-frame grab.

        The capture loop's sole job is to keep a sub-second-fresh "before" frame
        ready for the next mouse event — and the screen observer only ever
        produces an observation from a mouse event. So while the mouse has been
        active within the last ``idle_after`` seconds, grab at the full
        ``active_fps`` rate. Once input has been idle beyond that grace period,
        back off to ``idle_fps``: with no mouse events there is no interaction to
        transcribe, a stale before-frame costs nothing (the change-gate drops it
        if the screen really moved), and the continuous full-framebuffer grab
        otherwise just steals CPU/memory bandwidth from co-resident local
        inference. Passing ``idle_fps <= 0`` or ``idle_after <= 0`` disables the
        backoff (always full rate).
        """
        active = idle_fps <= 0 or idle_after <= 0 or (now - last_input) < idle_after
        fps = active_fps if active else idle_fps
        return 1.0 / fps if fps > 0 else 1.0 / active_fps

    @staticmethod
    def _mon_for(x: float, y: float, mons: list[dict]) -> Optional[int]:
        """Find which monitor contains the given coordinates.
        
        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            mons (list[dict]): List of monitor information dictionaries.
            
        Returns:
            Optional[int]: Monitor index if found, None otherwise.
        """
        for idx, m in enumerate(mons, 1):
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return idx
        return None

    @staticmethod
    def _encode_image(img_path: str, max_dim: int | None = None) -> str:
        """Encode an image file as base64, optionally downscaling it first.

        Args:
            img_path (str): Path to the image file.
            max_dim (int | None): If set, downscale the image so its long edge
                is at most this many pixels before encoding (used to send
                low-cost context frames to the VLM). When ``None`` the file is
                encoded verbatim from disk, avoiding a re-encode.

        Returns:
            str: Base64 encoded JPEG data.
        """
        if not max_dim:
            with open(img_path, "rb") as fh:
                return base64.b64encode(fh.read()).decode()

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode()

    # ─────────────────────────────── OpenAI Vision (async)
    @staticmethod
    def _per_image_dims(n: int, context_dim: int | None) -> list[int | None]:
        """Per-image downscale caps for a vision call of *n* images.

        The last image is the current screen and is always sent at full
        resolution (``None``); every earlier image is temporal context and is
        capped at ``context_dim`` when that is set. With ``context_dim`` unset
        (or a single image) all images are sent verbatim.
        """
        dims: list[int | None] = [None] * n
        if context_dim and n > 1:
            for i in range(n - 1):
                dims[i] = context_dim
        return dims

    async def _call_gpt_vision(
        self, prompt: str, img_paths: list[str], *, context_dim: int | None = None
    ) -> str:
        """Call GPT Vision API to analyze images.

        Args:
            prompt (str): Prompt to guide the analysis.
            img_paths (list[str]): List of image paths to analyze.
            context_dim (int | None): If set, downscale every image except the
                last (the current screen) to this long-edge before encoding, to
                cut prefill cost on context frames the transcription doesn't read.

        Returns:
            str: GPT's analysis of the images.
        """
        dims = self._per_image_dims(len(img_paths), context_dim)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
            for encoded in (await asyncio.gather(
                *[asyncio.to_thread(self._encode_image, p, d) for p, d in zip(img_paths, dims)]
            ))
        ]
        content.append({"type": "text", "text": prompt})

        # Serialize through the shared inference slot so vision calls don't
        # collide with the text model's proposition calls on the local server.
        async with inference_semaphore():
            rsp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "text"},
            )
        return rsp.choices[0].message.content

    # ─────────────────────────────── I/O helpers
    async def _save_frame(self, frame, tag: str) -> str:
        """Save a frame as a JPEG image.
        
        Args:
            frame: Frame data to save.
            tag (str): Tag to include in the filename.
            
        Returns:
            str: Path to the saved image.
        """
        ts   = f"{time.time():.5f}"
        path = os.path.join(self.screens_dir, f"{ts}_{tag}.jpg")
        await asyncio.to_thread(self._encode_frame, frame, path)
        return path

    def _encode_frame(self, frame, path: str) -> None:
        """Downscale (Retina screenshots are huge) and save a frame as JPEG.

        Capping the long edge keeps local VLM inference fast without hurting
        OCR quality. Runs in a worker thread via ``_save_frame``.
        """
        img = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)
        if self.max_image_dim and max(img.size) > self.max_image_dim:
            img.thumbnail((self.max_image_dim, self.max_image_dim), Image.LANCZOS)
        img.save(path, "JPEG", quality=70)

    # ─────────────────────────────── change detection
    _DIFF_THUMB = (128, 128)
    # A thumbnail pixel counts as "changed" only when its grayscale intensity
    # moves by more than this (0-255). Absorbs sub-threshold resample shimmer
    # while still tripping on any genuine content change.
    _DIFF_PIXEL_DELTA = 16

    def _frames_differ(self, before, after) -> bool:
        """Return True if two raw framebuffers differ meaningfully.

        Downscales both grabs to a small grayscale thumbnail, takes the
        per-pixel absolute difference, and counts how many thumbnail pixels
        changed by more than ``_DIFF_PIXEL_DELTA``. Frames "differ" once that
        count exceeds ``changed_pixel_tolerance``. A *count* (not a whole-frame
        average) is used deliberately: averaging dilutes a small localized edit
        into noise, whereas a toggled checkbox or a popped-up tooltip lights up
        a handful of thumbnail pixels at full delta and is preserved. Raw grabs
        are uncompressed, so a static screen yields an exact match (count 0).
        Runs off the event loop via ``to_thread``; any failure falls back to
        "differ" so a comparison error never silently drops an observation.
        """
        try:
            if before is None or after is None:
                return True
            if before.width != after.width or before.height != after.height:
                return True
            ba = (
                Image.frombytes("RGB", (before.width, before.height), before.rgb)
                .convert("L")
                .resize(self._DIFF_THUMB, Image.BILINEAR)
            )
            aa = (
                Image.frombytes("RGB", (after.width, after.height), after.rgb)
                .convert("L")
                .resize(self._DIFF_THUMB, Image.BILINEAR)
            )
            # histogram bins above the per-pixel delta = count of changed pixels
            hist = ImageChops.difference(ba, aa).histogram()
            changed = sum(hist[self._DIFF_PIXEL_DELTA + 1:])
            return changed > self.changed_pixel_tolerance
        except Exception:                                               # pragma: no cover
            return True

    async def _process_and_emit(self, before_path: str, after_path: str) -> None:
        """Process screenshots and emit an update.
        
        Args:
            before_path (str): Path to the "before" screenshot.
            after_path (str | None): Path to the "after" screenshot, if any.
        """
        # chronology: append 'before' first (history order == real order).
        # After this append, self._history already ends with before_path, so
        # list(self._history) yields [..recent befores.., before_path].
        self._history.append(before_path)
        prev_paths = list(self._history)

        # Chronological image set (recent history + this interaction's after),
        # capped so local VLMs stay responsive. The last image is the current
        # screen state. Do NOT re-append before_path here: it is already the
        # last element of prev_paths from history above, and sending the same
        # frame twice just doubles that image's encode + VLM prefill tokens
        # (the dominant cost of every combined vision call) for no new signal.
        prev_paths.append(after_path)
        prev_paths = prev_paths[-self.max_summary_images:]

        if self.combine_vision:
            # One vision call yields both the transcription and the action
            # summary, halving per-observation inference latency. Context frames
            # (all but the current/last) go at reduced resolution since the
            # transcription only reads the current frame.
            try:
                txt = (await self._call_gpt_vision(
                    self.combined_prompt, prev_paths, context_dim=self.context_image_dim
                )).strip()
            except Exception as exc:                                    # pragma: no cover
                txt = f"[analysis failed: {exc}]"
        else:
            # Legacy two-call path (GUM_COMBINE_VISION=0).
            try:
                transcription = await self._call_gpt_vision(self.transcription_prompt, [before_path, after_path])
            except Exception as exc:                                    # pragma: no cover
                transcription = f"[transcription failed: {exc}]"

            try:
                summary = await self._call_gpt_vision(self.summary_prompt, prev_paths)
            except Exception as exc:                                    # pragma: no cover
                summary = f"[summary failed: {exc}]"

            txt = (transcription + summary).strip()

        await self.update_queue.put(Update(content=txt, content_type="input_text"))

    # ─────────────────────────────── skip guard
    def _skip(self) -> bool:
        """Check if capture should be skipped based on visible applications.
        
        Returns:
            bool: True if capture should be skipped, False otherwise.
        """
        return _is_app_visible(self._guard) if self._guard else False

    # ─────────────────────────────── main async worker
    async def _worker(self) -> None:          # overrides base class
        """Main worker method that captures and processes screenshots.
        
        This method runs in a background task and handles:
        - Mouse event monitoring
        - Screen capture
        - Periodic screenshots
        - Image processing and analysis
        """
        log = logging.getLogger("Screen")
        if self.debug:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [Screen] %(message)s", datefmt="%H:%M:%S")
        else:
            log.addHandler(logging.NullHandler())
            log.propagate = False

        try:
            CAP_FPS = max(1, int(os.getenv("GUM_CAPTURE_FPS", str(self._CAPTURE_FPS))))
        except ValueError:
            CAP_FPS = self._CAPTURE_FPS
        # Adaptive idle backoff for the capture loop. While the mouse is active
        # we grab at CAP_FPS; after IDLE_AFTER seconds of no mouse input we drop
        # to IDLE_FPS, since a fresh before-frame is only needed to serve a mouse
        # event. Set GUM_CAPTURE_IDLE_FPS=0 to disable the backoff.
        try:
            IDLE_FPS = float(os.getenv("GUM_CAPTURE_IDLE_FPS", "0.2"))
        except ValueError:
            IDLE_FPS = 0.2
        try:
            IDLE_AFTER = float(os.getenv("GUM_CAPTURE_IDLE_AFTER", "3.0"))
        except ValueError:
            IDLE_AFTER = 3.0
        DEBOUNCE = self._DEBOUNCE_SEC

        loop = asyncio.get_running_loop()

        # ------------------------------------------------------------------
        # All calls to mss / Quartz are wrapped in `to_thread`
        # ------------------------------------------------------------------
        with mss.mss() as sct:
            mons = sct.monitors[self._MON_START:]

            # ---- mouse callbacks (pynput is sync → schedule into loop) ----
            def schedule_event(x: float, y: float, typ: str):
                asyncio.run_coroutine_threadsafe(mouse_event(x, y, typ), loop)

            listener = mouse.Listener(
                on_move=lambda x, y: schedule_event(x, y, "move"),
                on_click=lambda x, y, btn, prs: schedule_event(x, y, "click") if prs else None,
                on_scroll=lambda x, y, dx, dy: schedule_event(x, y, "scroll"),
            )
            listener.start()

            # ---- nested helper inside the async context ----
            async def flush():
                """Process pending event and emit update."""
                if self._pending_event is None:
                    return
                if self._skip():
                    self._pending_event = None
                    return

                ev = self._pending_event
                aft = await asyncio.to_thread(sct.grab, mons[ev["mon"] - 1])

                # Gate on visible change: if the screen is static, skip the VLM
                # call and the no-signal observation it would produce entirely.
                if self.skip_unchanged and not await asyncio.to_thread(
                    self._frames_differ, ev["before"], aft
                ):
                    log.info(f"{ev['type']} skipped on monitor {ev['mon']} (screen unchanged)")
                    self._pending_event = None
                    return

                bef_path = await self._save_frame(ev["before"], "before")
                aft_path = await self._save_frame(aft, "after")
                await self._process_and_emit(bef_path, aft_path)

                log.info(f"{ev['type']} captured on monitor {ev['mon']}")
                self._pending_event = None

            def debounce_flush():
                """Schedule flush as a task."""
                asyncio.create_task(flush())

            # ---- mouse event reception ----
            async def mouse_event(x: float, y: float, typ: str):
                """Handle mouse events.
                
                Args:
                    x (float): X coordinate.
                    y (float): Y coordinate.
                    typ (str): Event type ("move", "click", or "scroll").
                """
                # Record input activity so the capture loop stays at full rate
                # while the mouse is in use (see _capture_interval). Stamped for
                # every mouse event, even guarded/off-screen ones, so the loop is
                # already warmed up by the time a real interaction lands.
                self._last_input = time.monotonic()

                idx = self._mon_for(x, y, mons)
                log.info(
                    f"{typ:<6} @({x:7.1f},{y:7.1f}) → mon={idx}   {'(guarded)' if self._skip() else ''}"
                )
                if self._skip() or idx is None:
                    return

                # lazily grab before-frame
                if self._pending_event is None:
                    async with self._frame_lock:
                        bf = self._frames.get(idx)
                    if bf is None:
                        return
                    self._pending_event = {"type": typ, "mon": idx, "before": bf}

                # reset debounce timer
                if self._debounce_handle:
                    self._debounce_handle.cancel()
                self._debounce_handle = loop.call_later(DEBOUNCE, debounce_flush)

            # ---- main capture loop ----
            log.info(f"Screen observer started — guarding {self._guard or '∅'}")

            # Cap on how long the loop may sleep between wake-ups. Keeps renewed
            # mouse activity (which pulls the interval back to full rate) and
            # shutdown responsive even when the idle interval is several seconds.
            _POLL_MAX = 0.5
            last_grab = 0.0

            while self._running:                         # flag from base class
                now = time.monotonic()
                interval = self._capture_interval(
                    now, self._last_input, CAP_FPS, IDLE_FPS, IDLE_AFTER
                )

                # refresh 'before' buffers when the next grab is due — full rate
                # while the mouse is active, backing off to the idle rate once
                # input has been quiet (see _capture_interval) so idle background
                # grabbing stops competing with local inference for CPU/memory
                # bandwidth.
                if (now - last_grab) >= interval:
                    last_grab = now
                    for idx, m in enumerate(mons, 1):
                        frame = await asyncio.to_thread(sct.grab, m)
                        async with self._frame_lock:
                            self._frames[idx] = frame

                # Sleep until the next grab is due, capped at _POLL_MAX so
                # renewed activity resumes full-rate capture within a fraction of
                # a second instead of sleeping through a multi-second idle gap.
                remaining = interval - (time.monotonic() - last_grab)
                await asyncio.sleep(max(0.0, min(remaining, _POLL_MAX)))

            # shutdown
            listener.stop()
            if self._debounce_handle:
                self._debounce_handle.cancel()
