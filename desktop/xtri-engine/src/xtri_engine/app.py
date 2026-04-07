"""XTRI Engine — Toga shell wrapping the FastAPI backend in a WKWebView.

Tier 1 architecture (single-user native macOS .app):
  1. Run uvicorn programmatically in a worker THREAD (not a subprocess).
     The Briefcase GUI-Stub binary is NOT a generic Python interpreter — it
     ignores `-c` and `-m`, so the dev-mode `[sys.executable, "-m", "uvicorn"]`
     pattern silently hangs in a packaged .app. A worker thread sidesteps the
     stub binary entirely.
  2. Poll GET /health until the backend is ready (or timeout).
  3. Open a Toga MainWindow whose content is a WebView pointed at the backend.
  4. On exit (Toga's on_exit hook + main window on_close → request_exit),
     signal uvicorn to stop and join the worker thread.

The xtri-service package (installed via file:// URL — see desktop/xtri-engine/
pyproject.toml) ships `app.main:app` plus its referencias/ and static/ resources.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Load .env BEFORE uvicorn imports app.main. main.py has its own dotenv loader,
# but it uses __file__-relative paths that don't resolve correctly in a packaged
# .app (where __file__ is inside Contents/Resources/app_packages/, not the repo).
# We try a stable user-managed location first, then fall back to the hardcoded
# repo root (which matches the file:// pin in desktop/xtri-engine/pyproject.toml).
try:
    from dotenv import load_dotenv

    for _env_candidate in (
        Path.home() / "Library" / "Application Support" / "XTRI Engine" / ".env",
        Path("/Volumes/KINGSTON/apps/apps/MIRT/.env"),
    ):
        if _env_candidate.exists():
            load_dotenv(_env_candidate)
            break
except ImportError:
    pass

import httpx
import toga
import uvicorn
from toga.style.pack import Pack

HOST = "127.0.0.1"
PORT = 5003
APP_URL = f"http://{HOST}:{PORT}/"
HEALTH_URL = f"http://{HOST}:{PORT}/health"

STARTUP_TIMEOUT_S = 60.0  # cold scipy/pandas/supabase imports + JSON load → ~20-30s
POLL_INTERVAL_S = 0.25
SHUTDOWN_TIMEOUT_S = 5.0


class _XTRIServer(uvicorn.Server):
    """uvicorn.Server with signal handler installation disabled.

    `signal.signal()` only works on the main thread. We're running uvicorn in a
    worker thread, so calling install_signal_handlers would raise. Toga owns
    the main thread / Cocoa runloop and manages app shutdown via on_exit.
    Accept *args/**kwargs because the signature changed across uvicorn versions.
    """

    def install_signal_handlers(self, *args: object, **kwargs: object) -> None:
        return None


class XTRIEngine(toga.App):
    """Toga App that boots uvicorn in a thread and renders the frontend in a WebView."""

    _server: Optional[_XTRIServer]
    _server_thread: Optional[threading.Thread]

    def startup(self) -> None:
        self._server = None
        self._server_thread = None
        # Toga runs startup() inside a CFTimer callback. If we let an exception
        # propagate, rubicon-objc swallows it as "Exception ignored", the app
        # becomes a zombie, and uvicorn keeps running. Catch everything, force
        # cleanup, then hard-exit.
        try:
            self._start_backend()
            self._wait_for_ready()
        except BaseException as exc:
            sys.stderr.write(f"[xtri-engine] startup failed: {exc}\n")
            sys.stderr.flush()
            self._stop_backend()
            sys.exit(1)

        self.main_window = toga.MainWindow(
            title=self.formal_name,
            size=(1400, 900),
        )
        self.main_window.content = toga.WebView(
            url=APP_URL,
            style=Pack(flex=1),
        )
        # Window close button → request_exit (which fires on_exit override).
        self.main_window.on_close = self._on_main_window_close
        self.main_window.show()

    # ----- Toga lifecycle hooks --------------------------------------------

    def on_exit(self) -> bool:
        """Override of toga.App.on_exit (NOT an attribute assignment).

        Toga's __init__ wraps `self.on_exit` via `simple_handler` which respects
        MRO, so this override is picked up correctly. This is the ONLY place we
        can reliably tear down the uvicorn server while Python is still in
        control of the interpreter — Cocoa's NSApp.terminate_() bypasses
        Python's atexit hooks.

        Returning True allows the app to exit; returning False would cancel.
        """
        self._stop_backend()
        return True

    def _on_main_window_close(self, window: toga.Window, **kwargs: object) -> bool:
        """Forward window-close to app exit so on_exit fires.

        Calling self.exit() directly is WRONG — it bypasses on_exit (see
        toga.App.exit docstring). Use request_exit() which calls on_exit and
        then app.exit() if on_exit returns True.
        """
        self.request_exit()
        return True

    # ----- backend lifecycle ------------------------------------------------

    def _start_backend(self) -> None:
        """Spin up a uvicorn.Server in a daemon worker thread."""
        config = uvicorn.Config(
            "app.main:app",
            host=HOST,
            port=PORT,
            log_level="warning",
            workers=1,
            access_log=False,
        )
        self._server = _XTRIServer(config=config)
        self._server_thread = threading.Thread(
            target=self._server.run,
            name="uvicorn-server",
            daemon=True,
        )
        self._server_thread.start()

    def _wait_for_ready(self) -> None:
        """Block until GET /health returns 200, or raise on timeout / thread death."""
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            # If the server thread died (e.g., port already bound, import error)
            # surface immediately instead of waiting for the timeout.
            if self._server_thread is None or not self._server_thread.is_alive():
                raise RuntimeError("uvicorn server thread exited before becoming ready")
            try:
                response = httpx.get(HEALTH_URL, timeout=1.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass  # backend not up yet — keep polling
            time.sleep(POLL_INTERVAL_S)

        raise TimeoutError(
            f"Backend did not become ready at {HEALTH_URL} "
            f"within {STARTUP_TIMEOUT_S}s"
        )

    def _stop_backend(self) -> None:
        """Signal uvicorn to exit gracefully and join the worker thread."""
        if self._server is None:
            return
        self._server.should_exit = True
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=SHUTDOWN_TIMEOUT_S)


def main() -> XTRIEngine:
    return XTRIEngine()
