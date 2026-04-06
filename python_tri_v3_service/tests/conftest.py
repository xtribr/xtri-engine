"""
Shared fixtures for XTRI Engine API tests.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tabela_referencia import TabelaReferenciaTRI


@pytest.fixture(scope="session")
def tabela() -> TabelaReferenciaTRI:
    """Official ENEM reference table (loaded once per session)."""
    t = TabelaReferenciaTRI()
    t.validar()
    return t


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI test client (shared across all API tests)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

@pytest.fixture()
def gabarito_45() -> dict[str, str]:
    """Gabarito with 45 questions, answers cycling A-E."""
    opcoes = ["A", "B", "C", "D", "E"]
    return {str(i): opcoes[(i - 1) % 5] for i in range(1, 46)}


@pytest.fixture()
def areas_config_padrao() -> dict[str, list[int]]:
    """Standard ENEM 4-area config (2 cadernos of 45 each)."""
    return {
        "LC": [1, 45],
        "CH": [46, 90],
        "CN": [1, 45],
        "MT": [46, 90],
    }


@pytest.fixture()
def areas_config_simples() -> dict[str, tuple[int, int]]:
    """Single-area config for focused tests."""
    return {"MT": (1, 45)}


def _gerar_alunos_v2(
    n: int,
    gabarito: dict[str, str],
    *,
    seed: int = 42,
    acerto_medio: float = 0.5,
) -> list[dict]:
    """Generate N synthetic students in V2 format (q1, q2, ..., qN)."""
    rng = np.random.default_rng(seed)
    opcoes = ["A", "B", "C", "D", "E"]
    alunos = []
    for idx in range(n):
        aluno: dict = {"id": str(idx + 1), "nome": f"Aluno {idx + 1}"}
        for q_num_str, gabarito_resp in gabarito.items():
            if rng.random() < acerto_medio:
                aluno[f"q{q_num_str}"] = gabarito_resp
            else:
                erradas = [o for o in opcoes if o != gabarito_resp]
                aluno[f"q{q_num_str}"] = rng.choice(erradas)
        alunos.append(aluno)
    return alunos


@pytest.fixture()
def alunos_v2_30(gabarito_45):
    """30 synthetic students in V2 format."""
    return _gerar_alunos_v2(30, gabarito_45, seed=42, acerto_medio=0.5)


def _gerar_respostas_binarias(
    n_alunos: int,
    n_itens: int,
    *,
    seed: int = 42,
    acerto_medio: float = 0.5,
) -> list[list[int]]:
    """Generate binary response matrix for scientific engine."""
    rng = np.random.default_rng(seed)
    dificuldades = rng.uniform(0.2, 0.8, size=n_itens)
    habilidades = rng.normal(acerto_medio, 0.15, size=n_alunos)
    respostas = []
    for i in range(n_alunos):
        row = []
        for j in range(n_itens):
            prob = max(0.0, min(1.0, habilidades[i] + (0.5 - dificuldades[j]) * 0.5))
            row.append(1 if rng.random() < prob else 0)
        respostas.append(row)
    return respostas


@pytest.fixture()
def respostas_binarias_50x45() -> list[list[int]]:
    """50 students x 45 items binary responses."""
    return _gerar_respostas_binarias(50, 45, seed=42)


@pytest.fixture()
def respostas_binarias_200x45() -> list[list[int]]:
    """200 students x 45 items binary responses (for MIRT)."""
    return _gerar_respostas_binarias(200, 45, seed=123)
