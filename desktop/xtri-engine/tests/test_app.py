import sys
import threading
import time
import types
from pathlib import Path
from unittest import mock

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent.parent

for extra_path in (
    PROJECT_ROOT / "src",
    REPO_ROOT / "python_tri_v3_service",
):
    sys.path.insert(0, str(extra_path))

if "toga" not in sys.modules:
    toga_module = types.ModuleType("toga")

    class _DummyApp:
        pass

    class _DummyWindow:
        pass

    class _DummyMainWindow:
        pass

    class _DummyWebView:
        pass

    toga_module.App = _DummyApp
    toga_module.Window = _DummyWindow
    toga_module.MainWindow = _DummyMainWindow
    toga_module.WebView = _DummyWebView

    style_module = types.ModuleType("toga.style")
    pack_module = types.ModuleType("toga.style.pack")

    class _DummyPack:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    pack_module.Pack = _DummyPack

    sys.modules["toga"] = toga_module
    sys.modules["toga.style"] = style_module
    sys.modules["toga.style.pack"] = pack_module

from xtri_engine import app as desktop_app


def test_prepare_local_runtime_dirs_delega_para_layout():
    fake_layout = {"app_support": Path("/tmp/xtri")}

    with mock.patch.object(
        desktop_app,
        "ensure_app_support_layout",
        return_value=fake_layout,
    ) as mocked:
        result = desktop_app.prepare_local_runtime_dirs()

    mocked.assert_called_once_with()
    assert result == fake_layout


def test_app_url_abre_com_metodo_theta_ref_fixo_e_client_toga():
    assert desktop_app.APP_URL.endswith("/?anchor_method=theta_ref_fixo&client=toga")


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle tests — guard against the iter-4 subprocess regression.
# Scope: the thread-based uvicorn boot, /health poll, request_exit → on_exit
# → _stop_backend path. All with mocked uvicorn so the tests never bind a port.
# ─────────────────────────────────────────────────────────────────────────────


def _fresh_app() -> desktop_app.XTRIEngine:
    """Instantiate XTRIEngine without going through toga.App.__init__.

    The ``toga.App`` dummy in this file has no __init__, so bare construction
    is fine, but the real Toga App runs ``startup()`` from the Cocoa runloop.
    We skip that here and exercise the lifecycle helpers directly.
    """
    return desktop_app.XTRIEngine()


class _FakeServer:
    """Minimal stand-in for ``_XTRIServer`` — counts start/stop, blocks until told.

    CAUTION: ``run_duration > 0`` means ``should_exit`` is NOT checked during the
    initial ``time.sleep`` — anyone extending this with a non-zero duration AND
    calling ``_stop_backend`` in that window will block for the full duration.
    For robust early-exit, replace the sleep with an Event.wait(timeout=...).
    """

    def __init__(self, *, run_duration: float = 0.0) -> None:
        self.should_exit = False
        self.run_called = False
        self._run_duration = run_duration

    def run(self) -> None:
        self.run_called = True
        if self._run_duration > 0:
            time.sleep(self._run_duration)
        while not self.should_exit:
            time.sleep(0.01)


class TestBackendLifecycle:
    def test_start_backend_launches_daemon_thread(self, monkeypatch):
        fake = _FakeServer()
        monkeypatch.setattr(desktop_app, "_XTRIServer", lambda config: fake)

        app = _fresh_app()
        app._server = None
        app._server_thread = None
        try:
            app._start_backend()

            assert app._server is fake
            assert isinstance(app._server_thread, threading.Thread)
            assert app._server_thread.name == "uvicorn-server"
            assert app._server_thread.daemon is True
            # Give the thread a moment to enter its run loop.
            for _ in range(50):
                if fake.run_called:
                    break
                time.sleep(0.01)
            assert fake.run_called
        finally:
            app._stop_backend()

    def test_stop_backend_sets_should_exit_and_joins(self, monkeypatch):
        fake = _FakeServer()
        monkeypatch.setattr(desktop_app, "_XTRIServer", lambda config: fake)

        app = _fresh_app()
        app._server = None
        app._server_thread = None
        app._start_backend()
        # Wait for the thread to actually be running.
        for _ in range(50):
            if fake.run_called:
                break
            time.sleep(0.01)

        app._stop_backend()

        assert fake.should_exit is True
        assert app._server_thread is not None
        assert not app._server_thread.is_alive(), "worker thread did not join in time"

    def test_stop_backend_is_noop_when_no_server(self):
        app = _fresh_app()
        app._server = None
        app._server_thread = None
        # Must not raise — shutdown on a never-started app is a real path
        # (startup() exception before _start_backend returns).
        app._stop_backend()


class TestReadinessProbe:
    def test_wait_for_ready_returns_when_health_is_200(self, monkeypatch):
        """Happy path: /health starts returning 200 mid-poll and the wait exits."""
        fake = _FakeServer()
        monkeypatch.setattr(desktop_app, "_XTRIServer", lambda config: fake)

        call_counter = {"n": 0}

        def fake_get(url, timeout):
            call_counter["n"] += 1
            if call_counter["n"] < 3:
                raise desktop_app.httpx.ConnectError("backend not up yet")
            return types.SimpleNamespace(status_code=200)

        monkeypatch.setattr(desktop_app.httpx, "get", fake_get)
        monkeypatch.setattr(desktop_app, "POLL_INTERVAL_S", 0.01)

        app = _fresh_app()
        app._server = None
        app._server_thread = None
        try:
            app._start_backend()
            app._wait_for_ready()  # should return without raising
            assert call_counter["n"] >= 3
        finally:
            app._stop_backend()

    def test_wait_for_ready_raises_when_thread_dies(self, monkeypatch):
        """If the worker thread exits before /health responds, fail fast."""

        class _ImmediateExitServer:
            should_exit = False

            def run(self):
                # Simulate a uvicorn that can't bind the port — thread returns instantly.
                return None

        monkeypatch.setattr(
            desktop_app, "_XTRIServer", lambda config: _ImmediateExitServer(),
        )
        monkeypatch.setattr(
            desktop_app.httpx,
            "get",
            mock.Mock(side_effect=desktop_app.httpx.ConnectError("nobody listening")),
        )
        monkeypatch.setattr(desktop_app, "POLL_INTERVAL_S", 0.01)

        app = _fresh_app()
        app._server = None
        app._server_thread = None
        app._start_backend()
        # Let the "immediate exit" thread finish.
        app._server_thread.join(timeout=1.0)

        with pytest.raises(RuntimeError, match="exited before becoming ready"):
            app._wait_for_ready()

    def test_wait_for_ready_raises_timeout_when_health_never_responds(self, monkeypatch):
        fake = _FakeServer()
        monkeypatch.setattr(desktop_app, "_XTRIServer", lambda config: fake)
        monkeypatch.setattr(
            desktop_app.httpx,
            "get",
            mock.Mock(side_effect=desktop_app.httpx.ConnectError("never responds")),
        )
        monkeypatch.setattr(desktop_app, "STARTUP_TIMEOUT_S", 0.15)
        monkeypatch.setattr(desktop_app, "POLL_INTERVAL_S", 0.01)

        app = _fresh_app()
        app._server = None
        app._server_thread = None
        app._start_backend()
        try:
            with pytest.raises(TimeoutError, match="did not become ready"):
                app._wait_for_ready()
        finally:
            app._stop_backend()


class TestExitHooks:
    def test_on_exit_stops_backend_and_returns_true(self, monkeypatch):
        fake = _FakeServer()
        monkeypatch.setattr(desktop_app, "_XTRIServer", lambda config: fake)

        app = _fresh_app()
        app._server = None
        app._server_thread = None
        app._start_backend()

        result = app.on_exit()

        assert result is True
        assert fake.should_exit is True

    def test_main_window_close_calls_request_exit(self):
        """Window-close must route through request_exit so on_exit actually fires.

        Calling self.exit() directly would bypass on_exit and leave uvicorn running.

        Note: this only verifies the dispatch path (window-close → request_exit).
        It does NOT verify that toga.App actually has a ``request_exit`` method in
        production — the dummy toga.App in this file has no methods at all, so we
        monkey-patch the attribute onto the instance. A rename here from
        ``request_exit`` to something else would still be caught because the
        assert_called_once_with would fail; but we rely on Toga's real surface to
        keep request_exit available, which is verified by ``briefcase dev``.
        """
        app = _fresh_app()
        app.request_exit = mock.Mock()
        result = app._on_main_window_close(window=None)
        assert result is True
        app.request_exit.assert_called_once_with()
