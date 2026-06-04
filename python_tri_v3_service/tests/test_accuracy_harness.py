import pytest
from pathlib import Path


@pytest.mark.unit
def test_microdata_dir_has_no_double_apps():
    from app.microdata.loader import MICRODATA_DIR, default_csv_path

    assert "apps/apps" not in str(MICRODATA_DIR), (
        f"MICRODATA_DIR ainda tem o bug de caminho: {MICRODATA_DIR}"
    )
    assert MICRODATA_DIR == Path("/Volumes/KINGSTON/apps/MIRT/microdata")
    p = default_csv_path(2024)
    assert p.name == "MICRODADOS_ENEM_2024.csv"
    assert p.parent == MICRODATA_DIR
