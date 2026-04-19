import sys
import types
from pathlib import Path
from unittest import mock

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
