# Arnês de Acurácia TRI (Fase 1) — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir um arnês de avaliação confiável que mede a acurácia do motor TRI contra a verdade-base real do INEP — recuperação de parâmetro (com linking de escala), nota ponta-a-ponta com hold-out (k-fold), e decomposição calibração-vs-ancoragem — emitindo um relatório via CLI e um baseline congelado.

**Architecture:** Novo pacote `app/evaluation/` (arquivos pequenos e focados) que orquestra peças já existentes (`calibrar_itens_fast`, `eap_theta_batch`, ancoragem em `transform`, `microdata.loader`, params oficiais em `scripts_local/data/params_oficiais.json`). Mede, não reescreve. Execução pesada via `python -m app.evaluation`; guard rápido via `tests/test_accuracy_harness.py`.

**Tech Stack:** Python 3.14, NumPy, SciPy (`scipy.optimize.minimize`, `scipy.stats.norm`), pandas, pytest (marcadores `unit`/`integration`).

**Spec:** `docs/superpowers/specs/2026-06-04-tri-accuracy-harness-design.md`

---

## Mapa de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `app/microdata/loader.py` (modificar) | Corrigir bug de caminho `apps/apps` → `apps` |
| `app/evaluation/__init__.py` (criar) | Marcador de pacote |
| `app/evaluation/metrics.py` (criar) | MAE/RMSE/r, IC bootstrap, diferença pareada, stats de recuperação |
| `app/evaluation/linking.py` (criar) | Mean-sigma + Stocking-Lord; aplicar linking |
| `app/evaluation/ground_truth.py` (criar) | Carregar params oficiais; co_prova AZUL; alinhar/validar gabarito |
| `app/evaluation/kfold.py` (criar) | Matriz por grupo; OOF k-fold; agregação por área |
| `app/evaluation/attribution.py` (criar) | Pontuar com params oficiais; decompor erro |
| `app/evaluation/report.py` (criar) | Montar relatório JSON + tabela texto |
| `app/evaluation/__main__.py` (criar) | CLI: `python -m app.evaluation --year 2024` |
| `tests/test_accuracy_harness.py` (criar) | Testes unitários + floors finos; corrige `TestRealINEPBenchmark` |
| `tests/test_accuracy_benchmarks.py` (modificar) | Corrige resolução de caminho + skip barulhento |

**Convenções de tipo fixadas (usadas em todo o plano):**
- Item de parâmetro = `dict` com chaves `a, b, c` (float) e opcional `saturado` (bool) — compatível com `calibrar_itens_fast`/`eap_theta_batch`.
- Resultado de área k-fold = `dict` com chaves `area, n, pred (np.ndarray), ref (np.ndarray)`.
- `Linking` = dataclass frozen com `A: float, B: float`.

Todos os comandos de teste rodam de `python_tri_v3_service/`:
```bash
cd /Volumes/KINGSTON/apps/MIRT/python_tri_v3_service
```

---

## Task 1: Corrigir o bug de caminho que desliga o guardrail

**Files:**
- Modify: `app/microdata/loader.py:7` (docstring) e `:35` (`MICRODATA_DIR`)
- Test: `tests/test_accuracy_harness.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_accuracy_harness.py
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_accuracy_harness.py::test_microdata_dir_has_no_double_apps -v`
Expected: FAIL — `assert "apps/apps" not in "/Volumes/KINGSTON/apps/apps/MIRT/microdata"`

- [ ] **Step 3: Corrigir o caminho**

Em `app/microdata/loader.py:35`, trocar:
```python
MICRODATA_DIR = Path("/Volumes/KINGSTON/apps/apps/MIRT/microdata")
```
por:
```python
MICRODATA_DIR = Path("/Volumes/KINGSTON/apps/MIRT/microdata")
```
E na docstring do módulo (linha ~7), corrigir o mesmo caminho de exemplo `apps/apps/MIRT` → `apps/MIRT`.

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_accuracy_harness.py::test_microdata_dir_has_no_double_apps -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/microdata/loader.py tests/test_accuracy_harness.py
git commit -m "fix(microdata): corrige MICRODATA_DIR (apps/apps -> apps) que pulava o benchmark real"
```

---

## Task 2: `metrics.py` — métricas e IC bootstrap

**Files:**
- Create: `app/evaluation/__init__.py` (vazio)
- Create: `app/evaluation/metrics.py`
- Test: `tests/test_accuracy_harness.py`

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_accuracy_harness.py  (adicionar)
import numpy as np


@pytest.mark.unit
def test_mae_rmse_pearson_exact():
    from app.evaluation.metrics import mae, rmse, pearson_r

    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.0, 2.0, 3.0, 5.0])
    assert mae(pred, target) == pytest.approx(0.25)
    assert rmse(pred, target) == pytest.approx(0.5)
    assert pearson_r(pred, target) == pytest.approx(0.9827076, abs=1e-5)


@pytest.mark.unit
def test_bootstrap_ci_contains_true_mean():
    from app.evaluation.metrics import bootstrap_ci

    rng = np.random.default_rng(0)
    values = rng.normal(50.0, 5.0, size=2000)
    lo, hi = bootstrap_ci(values, n_boot=500, seed=42)
    assert lo < 50.0 < hi
    assert hi - lo < 2.0  # IC estreito com n grande


@pytest.mark.unit
def test_paired_diff_ci_detects_improvement():
    from app.evaluation.metrics import paired_diff_ci

    err_baseline = np.full(1000, 40.0)
    err_candidate = np.full(1000, 35.0)  # 5 pontos melhor, sempre
    mean_diff, lo, hi = paired_diff_ci(err_candidate, err_baseline, seed=42)
    assert mean_diff == pytest.approx(-5.0)
    assert hi < 0  # IC inteiramente do lado da melhora → exclui 0


@pytest.mark.unit
def test_param_recovery_stats_perfect():
    from app.evaluation.metrics import param_recovery_stats

    ours = np.array([0.5, 1.0, 1.5, 2.0])
    stats = param_recovery_stats(ours, ours)
    assert stats["rmse"] == pytest.approx(0.0)
    assert stats["r"] == pytest.approx(1.0)
    assert stats["n"] == 4
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_accuracy_harness.py -k "mae_rmse or bootstrap or paired or recovery_stats" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.evaluation'`

- [ ] **Step 3: Implementar**

```python
# app/evaluation/__init__.py
```
(arquivo vazio)

```python
# app/evaluation/metrics.py
"""Métricas de acurácia e intervalos de confiança por bootstrap."""
from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(pred) < 2 or np.std(pred) == 0 or np.std(target) == 0:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    stat: str = "mean",
) -> tuple[float, float]:
    """IC percentil por bootstrap de uma estatística (mean|median) de `values`."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    fn = {"mean": np.mean, "median": np.median}[stat]
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = fn(values[rng.integers(0, n, n)])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_diff_ci(
    err_candidate: np.ndarray,
    err_baseline: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """IC bootstrap da diferença PAREADA de erro (candidato − baseline).

    Negativo = candidato erra menos. Se `hi < 0`, a melhora é significativa
    (o IC exclui 0). Pareado por aluno: mais sensível que comparar dois ICs.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(err_candidate, dtype=float) - np.asarray(err_baseline, dtype=float)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = diff[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(diff.mean()), lo, hi


def param_recovery_stats(ours: np.ndarray, official: np.ndarray) -> dict:
    """Compara um vetor de parâmetros calibrados vs. oficiais (já linkados)."""
    ours = np.asarray(ours, dtype=float)
    official = np.asarray(official, dtype=float)
    return {
        "n": int(len(ours)),
        "r": pearson_r(ours, official),
        "rmse": rmse(ours, official),
        "mean_diff": float(np.mean(ours - official)),
    }
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_accuracy_harness.py -k "mae_rmse or bootstrap or paired or recovery_stats" -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/__init__.py app/evaluation/metrics.py tests/test_accuracy_harness.py
git commit -m "feat(evaluation): metricas de acuracia + IC bootstrap + diferenca pareada"
```

---

## Task 3: `linking.py` — equating de escala (mean-sigma + Stocking-Lord)

**Files:**
- Create: `app/evaluation/linking.py`
- Test: `tests/test_accuracy_harness.py`

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_accuracy_harness.py  (adicionar)
@pytest.mark.unit
def test_mean_sigma_recovers_known_transform():
    from app.evaluation.linking import mean_sigma_linking

    rng = np.random.default_rng(1)
    b_ours = rng.normal(0.0, 1.0, size=45)
    A_true, B_true = 1.3, -0.4
    b_official = A_true * b_ours + B_true
    link = mean_sigma_linking(b_ours, b_official)
    assert link.A == pytest.approx(A_true, abs=1e-6)
    assert link.B == pytest.approx(B_true, abs=1e-6)


@pytest.mark.unit
def test_apply_linking_transforms_a_and_b():
    from app.evaluation.linking import Linking, apply_linking

    params = [{"a": 2.0, "b": 1.0, "c": 0.2}]
    out = apply_linking(params, Linking(A=2.0, B=0.5))
    assert out[0]["b"] == pytest.approx(2.0 * 1.0 + 0.5)
    assert out[0]["a"] == pytest.approx(2.0 / 2.0)
    assert out[0]["c"] == pytest.approx(0.2)  # c não muda


@pytest.mark.unit
def test_stocking_lord_recovers_known_transform():
    from app.evaluation.linking import stocking_lord_linking

    rng = np.random.default_rng(2)
    a_f = rng.uniform(0.8, 2.2, size=45)
    b_f = rng.uniform(-2.0, 2.0, size=45)
    c_f = np.full(45, 0.2)
    official = [{"a": a_f[j], "b": b_f[j], "c": c_f[j]} for j in range(45)]
    A_true, B_true = 1.25, 0.3
    # "ours" na métrica de origem: a_ours = a_f * A, b_ours = (b_f - B)/A
    ours = [
        {"a": a_f[j] * A_true, "b": (b_f[j] - B_true) / A_true, "c": c_f[j]}
        for j in range(45)
    ]
    link = stocking_lord_linking(ours, official)
    assert link.A == pytest.approx(A_true, abs=0.05)
    assert link.B == pytest.approx(B_true, abs=0.05)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_accuracy_harness.py -k "linking or stocking" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.evaluation.linking'`

- [ ] **Step 3: Implementar**

```python
# app/evaluation/linking.py
"""Equating de escala IRT: transforma nossos params para a métrica INEP.

Convenção: θ_oficial = A·θ_nosso + B. Para itens: b' = A·b + B, a' = a/A
(c invariante). Linking é necessário porque nossa calibração usa prior N(0,1)
fresco por grupo, enquanto os params oficiais vivem na métrica histórica INEP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Linking:
    A: float
    B: float


def _p3pl(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def mean_sigma_linking(b_ours: np.ndarray, b_official: np.ndarray) -> Linking:
    """Linking pelos momentos das dificuldades dos itens comuns."""
    b_ours = np.asarray(b_ours, dtype=float)
    b_official = np.asarray(b_official, dtype=float)
    A = float(b_official.std(ddof=1) / b_ours.std(ddof=1))
    B = float(b_official.mean() - A * b_ours.mean())
    return Linking(A=A, B=B)


def stocking_lord_linking(
    params_ours: list[dict],
    params_official: list[dict],
    *,
    theta_grid: np.ndarray | None = None,
) -> Linking:
    """Linking que minimiza a diferença das curvas características do teste (TCC)."""
    if theta_grid is None:
        theta_grid = np.linspace(-4.0, 4.0, 81)

    a_o = np.array([p["a"] for p in params_ours], dtype=float)
    b_o = np.array([p["b"] for p in params_ours], dtype=float)
    c_o = np.array([p["c"] for p in params_ours], dtype=float)
    a_f = np.array([p["a"] for p in params_official], dtype=float)
    b_f = np.array([p["b"] for p in params_official], dtype=float)
    c_f = np.array([p["c"] for p in params_official], dtype=float)

    tcc_target = sum(_p3pl(theta_grid, a_f[j], b_f[j], c_f[j]) for j in range(len(a_f)))

    def loss(x: np.ndarray) -> float:
        A, B = float(x[0]), float(x[1])
        if A <= 1e-6:
            return 1e12
        tcc = np.zeros_like(theta_grid)
        for j in range(len(a_o)):
            tcc = tcc + _p3pl(theta_grid, a_o[j] / A, A * b_o[j] + B, c_o[j])
        return float(np.mean((tcc - tcc_target) ** 2))

    res = minimize(loss, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    return Linking(A=float(res.x[0]), B=float(res.x[1]))


def apply_linking(params: list[dict], link: Linking) -> list[dict]:
    """Retorna NOVOS dicts com a/b transformados para a métrica alvo."""
    return [
        {**p, "a": p["a"] / link.A, "b": link.A * p["b"] + link.B}
        for p in params
    ]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_accuracy_harness.py -k "linking or stocking" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/linking.py tests/test_accuracy_harness.py
git commit -m "feat(evaluation): linking de escala mean-sigma e Stocking-Lord"
```

---

## Task 4: `ground_truth.py` — params oficiais + alinhamento por gabarito

**Files:**
- Create: `app/evaluation/ground_truth.py`
- Test: `tests/test_accuracy_harness.py`

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_accuracy_harness.py  (adicionar)
@pytest.mark.unit
def test_official_items_and_coprova():
    from app.evaluation.ground_truth import (
        load_official_params,
        official_coprova,
        official_items,
    )

    params = load_official_params()  # caminho padrão real
    assert official_coprova(params, 2024, "MT") == 1323
    assert official_coprova(params, 2024, "CN") == 1333
    items = official_items(params, 2024, "MT")
    assert len(items) == 45
    assert set(items[0].keys()) >= {"a", "b", "c", "posicao", "gabarito"}


@pytest.mark.unit
def test_align_official_to_gabarito_matches_and_skips_annulled():
    from app.evaluation.ground_truth import align_official_to_gabarito

    official = [
        {"a": 1.0, "b": 0.0, "c": 0.2, "posicao": 91, "gabarito": "A"},
        {"a": 1.2, "b": 0.5, "c": 0.2, "posicao": 92, "gabarito": "B"},
        {"a": 0.9, "b": -0.3, "c": 0.2, "posicao": 93, "gabarito": "C"},
    ]
    # gabarito do grupo casa, mas item 2 (index 1) foi anulado ("X")
    valid_cols, aligned = align_official_to_gabarito(official, "AXC")
    assert valid_cols == [0, 2]
    assert [p["posicao"] for p in aligned] == [91, 93]


@pytest.mark.unit
def test_align_raises_on_gabarito_mismatch():
    from app.evaluation.ground_truth import align_official_to_gabarito

    official = [{"a": 1.0, "b": 0.0, "c": 0.2, "posicao": 91, "gabarito": "A"}]
    with pytest.raises(ValueError, match="gabarito"):
        align_official_to_gabarito(official, "B")  # não casa
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_accuracy_harness.py -k "official or align" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.evaluation.ground_truth'`

- [ ] **Step 3: Implementar**

```python
# app/evaluation/ground_truth.py
"""Carrega params oficiais INEP e alinha aos itens calibrados por gabarito."""
from __future__ import annotations

import json
from pathlib import Path

_ANULADO = set("X.*")

OFFICIAL_PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts_local"
    / "data"
    / "params_oficiais.json"
)


def load_official_params(path: Path | str = OFFICIAL_PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def official_coprova(params: dict, year: int, area: str) -> int:
    return int(params["_meta"]["co_provas"][f"{year}_{area}"])


def official_items(params: dict, year: int, area: str) -> list[dict]:
    """Lista de 45 itens: [{posicao, a, b, c, gabarito}, ...] em ordem canônica."""
    raw = params[str(year)][area]  # [[posicao, a, b, c, gab], ...]
    return [
        {
            "posicao": int(row[0]),
            "a": float(row[1]),
            "b": float(row[2]),
            "c": float(row[3]),
            "gabarito": str(row[4]),
        }
        for row in raw
    ]


def align_official_to_gabarito(
    official: list[dict], group_gabarito: str
) -> tuple[list[int], list[dict]]:
    """Valida que o gabarito oficial casa com o do grupo, posição a posição.

    Retorna (colunas_válidas, params_oficiais_válidos), pulando itens anulados.
    Levanta ValueError se algum item não-anulado tiver gabarito divergente —
    sinal de cor/ordem trocada (aborta em vez de produzir números errados).
    """
    if len(official) != len(group_gabarito):
        raise ValueError(
            f"tamanho do gabarito difere: oficial={len(official)} "
            f"grupo={len(group_gabarito)}"
        )
    valid_cols: list[int] = []
    aligned: list[dict] = []
    for j, item in enumerate(official):
        gab_grupo = group_gabarito[j]
        if gab_grupo in _ANULADO or item["gabarito"] in _ANULADO:
            continue  # item anulado: sem recuperação significativa
        if item["gabarito"] != gab_grupo:
            raise ValueError(
                f"gabarito divergente na posição {j}: "
                f"oficial={item['gabarito']!r} grupo={gab_grupo!r}"
            )
        valid_cols.append(j)
        aligned.append({"a": item["a"], "b": item["b"], "c": item["c"], "saturado": False})
    return valid_cols, aligned
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_accuracy_harness.py -k "official or align" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/ground_truth.py tests/test_accuracy_harness.py
git commit -m "feat(evaluation): carga de params oficiais INEP + alinhamento por gabarito"
```

---

## Task 5: `kfold.py` — matriz por grupo e predições OOF

**Files:**
- Create: `app/evaluation/kfold.py`
- Test: `tests/test_accuracy_harness.py`

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_accuracy_harness.py  (adicionar)
@pytest.mark.unit
def test_kfold_indices_partition_and_no_leak():
    from app.evaluation.kfold import _kfold_indices

    folds = _kfold_indices(100, k=5, seed=42)
    assert len(folds) == 5
    allidx = np.concatenate(folds)
    assert sorted(allidx.tolist()) == list(range(100))  # partição exata, sem overlap


@pytest.mark.unit
def test_anchor_batch_matches_individual():
    from app.evaluation.kfold import anchor_batch
    from app.irt.transform import ancorar_nota_individual
    from app.tabela_referencia import TabelaReferenciaTRI

    tabela = TabelaReferenciaTRI()
    theta = np.array([0.5, -0.2, 1.1])
    acertos = np.array([30, 20, 38])
    got = anchor_batch(theta, acertos, "MT", tabela)
    expect = np.array([
        ancorar_nota_individual(theta=0.5, area="MT", acertos=30, tabela=tabela, tolerancia_pct=0.0),
        ancorar_nota_individual(theta=-0.2, area="MT", acertos=20, tabela=tabela, tolerancia_pct=0.0),
        ancorar_nota_individual(theta=1.1, area="MT", acertos=38, tabela=tabela, tolerancia_pct=0.0),
    ])
    assert np.allclose(got, expect)


@pytest.mark.unit
def test_kfold_oof_predictions_shape_and_range():
    from app.evaluation.kfold import kfold_oof_predictions
    from app.tabela_referencia import TabelaReferenciaTRI

    rng = np.random.default_rng(3)
    # 600 alunos sintéticos, 45 itens, sinal claro (theta latente -> acertos)
    theta_true = rng.normal(0, 1, size=600)
    b = np.linspace(-2, 2, 45)
    p = 0.2 + 0.8 / (1 + np.exp(-(1.3) * (theta_true[:, None] - b[None, :])))
    resp = (rng.uniform(size=p.shape) < p).astype(int)
    tabela = TabelaReferenciaTRI()
    oof = kfold_oof_predictions(resp, "CN", tabela, k=5, seed=42)
    assert oof.shape == (600,)
    assert np.all(~np.isnan(oof))
    assert np.all(oof >= 100.0) and np.all(oof <= 1000.0)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_accuracy_harness.py -k "kfold or anchor_batch" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.evaluation.kfold'`

- [ ] **Step 3: Implementar**

```python
# app/evaluation/kfold.py
"""Hold-out k-fold sobre alunos: calibra no treino, pontua no teste.

Mede generalização (não memorização). Reusa calibração e EAP existentes;
nunca mistura provas/áreas num modelo (regra inviolável).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.irt.transform import ancorar_nota_individual
from app.microdata.calibration import (
    _effective_lc_gabarito,
    _parse_nota_series,
    _resp_to_binary,
    calibrar_itens_fast,
    eap_theta_batch,
)
from app.microdata.schema import MicrodataSchema
from app.tabela_referencia import TabelaReferenciaTRI

MIN_ITEMS_PER_STUDENT = 30


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    """k folds de teste (partição exata dos índices 0..n-1)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def anchor_batch(
    theta_arr: np.ndarray,
    acertos_arr: np.ndarray,
    area: str,
    tabela: TabelaReferenciaTRI,
) -> np.ndarray:
    return np.array(
        [
            ancorar_nota_individual(
                theta=float(t), area=area, acertos=int(ac),
                tabela=tabela, tolerancia_pct=0.0,
            )
            for t, ac in zip(theta_arr, acertos_arr)
        ],
        dtype=float,
    )


def build_group_matrix(
    df_group: pd.DataFrame, schema: MicrodataSchema, area: str
) -> tuple[np.ndarray, np.ndarray]:
    """Constrói (matriz 0/1, nota_ref) de um grupo homogêneo (mesma prova)."""
    resp_arr = df_group[schema.tx_respostas[area]].astype(str).to_numpy()
    gab_arr = df_group[schema.tx_gabarito[area]].astype(str).to_numpy()
    nota_ref = _parse_nota_series(df_group[schema.nu_nota[area]]).to_numpy()
    lingua_arr = (
        df_group[schema.tp_lingua].astype(str).to_numpy() if area == "LC" else None
    )
    rows: list[list[int]] = []
    notas: list[float] = []
    for i in range(len(df_group)):
        gab = gab_arr[i]
        if area == "LC" and len(gab) == 50:
            try:
                tp = int(lingua_arr[i].strip())
            except (ValueError, AttributeError):
                continue
            sliced = _effective_lc_gabarito(gab, tp)
            if sliced is None:
                continue
            gab = sliced
        binary = _resp_to_binary(resp_arr[i], gab)
        if binary is None or len(binary) < MIN_ITEMS_PER_STUDENT:
            continue
        rows.append(binary)
        notas.append(float(nota_ref[i]))
    if not rows or len({len(r) for r in rows}) > 1:
        return np.empty((0, 0), dtype=int), np.empty((0,), dtype=float)
    return np.array(rows, dtype=int), np.array(notas, dtype=float)


def kfold_oof_predictions(
    resp_matrix: np.ndarray,
    area: str,
    tabela: TabelaReferenciaTRI,
    *,
    k: int = 5,
    seed: int = 42,
    c_strategy: str = "fixed",
) -> np.ndarray:
    """Notas out-of-fold: para cada fold, calibra no resto e pontua o fold."""
    n = len(resp_matrix)
    oof = np.full(n, np.nan, dtype=float)
    acertos = resp_matrix.sum(axis=1)
    folds = _kfold_indices(n, k, seed)
    all_idx = np.arange(n)
    for test_idx in folds:
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        params = calibrar_itens_fast(resp_matrix[train_idx], c_strategy=c_strategy)
        theta, _ = eap_theta_batch(resp_matrix[test_idx], params)
        oof[test_idx] = anchor_batch(theta, acertos[test_idx], area, tabela)
    return oof


def run_kfold_area(
    df: pd.DataFrame,
    schema: MicrodataSchema,
    area: str,
    tabela: TabelaReferenciaTRI,
    *,
    k: int = 5,
    seed: int = 42,
    c_strategy: str = "fixed",
    min_group: int = 500,
) -> dict:
    """Agrega predições OOF por (co_prova[, tp_lingua]) e empilha por área."""
    co_col = schema.co_prova[area]
    group_cols = [co_col, schema.tp_lingua] if area == "LC" else [co_col]
    preds: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    for _, g in df.groupby(group_cols, dropna=False):
        resp, nota_ref = build_group_matrix(g, schema, area)
        if len(resp) < min_group:
            continue
        oof = kfold_oof_predictions(resp, area, tabela, k=k, seed=seed, c_strategy=c_strategy)
        mask = ~np.isnan(oof) & ~np.isnan(nota_ref) & (nota_ref > 0)
        preds.append(oof[mask])
        refs.append(nota_ref[mask])
    if not preds:
        return {"area": area, "n": 0, "pred": np.array([]), "ref": np.array([])}
    return {
        "area": area,
        "n": int(sum(len(p) for p in preds)),
        "pred": np.concatenate(preds),
        "ref": np.concatenate(refs),
    }
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_accuracy_harness.py -k "kfold or anchor_batch" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/kfold.py tests/test_accuracy_harness.py
git commit -m "feat(evaluation): hold-out k-fold sobre alunos com predicoes OOF por area"
```

---

## Task 6: `attribution.py` — decomposição calibração vs. ancoragem

**Files:**
- Create: `app/evaluation/attribution.py`
- Test: `tests/test_accuracy_harness.py`

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_accuracy_harness.py  (adicionar)
@pytest.mark.unit
def test_score_with_official_params_shapes():
    from app.evaluation.attribution import score_with_params
    from app.tabela_referencia import TabelaReferenciaTRI

    rng = np.random.default_rng(4)
    resp = (rng.uniform(size=(50, 45)) < 0.6).astype(int)
    params = [{"a": 1.2, "b": 0.0, "c": 0.2, "saturado": False} for _ in range(45)]
    tabela = TabelaReferenciaTRI()
    notas = score_with_params(resp, params, "CN", tabela)
    assert notas.shape == (50,)
    assert np.all(notas >= 100.0) and np.all(notas <= 1000.0)


@pytest.mark.unit
def test_decompose_error_attributes_calibration():
    from app.evaluation.attribution import decompose_error

    ref = np.array([500.0, 600.0, 700.0])
    our_pred = np.array([540.0, 560.0, 760.0])      # erro total maior
    official_pred = np.array([510.0, 590.0, 710.0])  # ancoragem ~boa
    out = decompose_error(our_pred, official_pred, ref)
    assert out["mae_full"] == pytest.approx(np.mean([40, 40, 60]))
    assert out["mae_anchor"] == pytest.approx(np.mean([10, 10, 10]))
    assert out["calib_contrib_mean"] == pytest.approx(out["mae_full"] - out["mae_anchor"])
    assert "calib_contrib_ci" in out  # (lo, hi) por bootstrap pareado
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_accuracy_harness.py -k "score_with_official or decompose" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.evaluation.attribution'`

- [ ] **Step 3: Implementar**

```python
# app/evaluation/attribution.py
"""Decompõe o erro de nota em parcela de ANCORAGEM vs. CALIBRAÇÃO.

Pontuar com params OFICIAIS isola a ancoragem (params são "verdadeiros");
a diferença para o erro com NOSSOS params atribui a parcela de calibração.
Só CN/MT/AZUL de 2024 (único ano com NU_NOTA e params oficiais).
"""
from __future__ import annotations

import numpy as np

from app.evaluation.kfold import anchor_batch
from app.evaluation.metrics import mae, paired_diff_ci
from app.microdata.calibration import eap_theta_batch
from app.tabela_referencia import TabelaReferenciaTRI


def score_with_params(
    resp_matrix: np.ndarray,
    item_params: list[dict],
    area: str,
    tabela: TabelaReferenciaTRI,
) -> np.ndarray:
    """EAP com params FIXOS + ancoragem. `resp_matrix` deve alinhar a `item_params`."""
    theta, _ = eap_theta_batch(resp_matrix, item_params)
    acertos = resp_matrix.sum(axis=1)
    return anchor_batch(theta, acertos, area, tabela)


def decompose_error(
    our_pred: np.ndarray, official_pred: np.ndarray, nota_ref: np.ndarray
) -> dict:
    """Atribui erro de nota a ancoragem (oficial) vs. calibração (resíduo)."""
    err_full = np.abs(np.asarray(our_pred, float) - np.asarray(nota_ref, float))
    err_anchor = np.abs(np.asarray(official_pred, float) - np.asarray(nota_ref, float))
    mean_diff, lo, hi = paired_diff_ci(err_full, err_anchor)
    return {
        "n": int(len(nota_ref)),
        "mae_full": mae(our_pred, nota_ref),
        "mae_anchor": mae(official_pred, nota_ref),
        "calib_contrib_mean": mean_diff,
        "calib_contrib_ci": (lo, hi),
    }
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_accuracy_harness.py -k "score_with_official or decompose" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/attribution.py tests/test_accuracy_harness.py
git commit -m "feat(evaluation): decomposicao de erro calibracao-vs-ancoragem"
```

---

## Task 7: `report.py` — relatório JSON + tabela texto

**Files:**
- Create: `app/evaluation/report.py`
- Test: `tests/test_accuracy_harness.py`

- [ ] **Step 1: Escrever os testes que falham**

```python
# tests/test_accuracy_harness.py  (adicionar)
@pytest.mark.unit
def test_build_and_render_report(tmp_path):
    from app.evaluation.report import build_report, render_text_table, write_report

    end_to_end = {
        "LC": {"area": "LC", "n": 1000, "mae": 38.2, "rmse": 47.1, "r": 0.91,
               "mae_ci": (37.0, 39.4)},
        "MT": {"area": "MT", "n": 1000, "mae": 41.0, "rmse": 52.0, "r": 0.90,
               "mae_ci": (39.5, 42.5)},
    }
    attribution = {
        "MT": {"n": 1000, "mae_full": 41.0, "mae_anchor": 22.0,
               "calib_contrib_mean": 19.0, "calib_contrib_ci": (17.0, 21.0)},
    }
    recovery = {
        "MT": {
            "2024": {
                "mean_sigma": {"a": {"r": 0.85, "rmse": 0.3, "mean_diff": 0.0, "n": 45},
                               "b": {"r": 0.93, "rmse": 0.25, "mean_diff": 0.0, "n": 45}},
                "stocking_lord": {"a": {"r": 0.86, "rmse": 0.28, "mean_diff": 0.0, "n": 45},
                                  "b": {"r": 0.94, "rmse": 0.22, "mean_diff": 0.0, "n": 45}},
            }
        }
    }
    report = build_report(year=2024, end_to_end=end_to_end,
                          attribution=attribution, recovery=recovery)
    assert report["year"] == 2024
    assert report["end_to_end"]["MT"]["mae"] == 41.0

    table = render_text_table(report)
    assert "MT" in table and "MAE" in table

    out = tmp_path / "report.json"
    write_report(report, out)
    assert out.is_file()
    import json
    loaded = json.loads(out.read_text())
    assert loaded["year"] == 2024
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_accuracy_harness.py -k "build_and_render" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.evaluation.report'`

- [ ] **Step 3: Implementar**

```python
# app/evaluation/report.py
"""Monta e renderiza o relatório de acurácia (JSON + tabela texto)."""
from __future__ import annotations

import json
from pathlib import Path


def build_report(
    *, year: int, end_to_end: dict, attribution: dict, recovery: dict
) -> dict:
    return {
        "year": year,
        "end_to_end": end_to_end,
        "attribution": attribution,
        "recovery": recovery,
    }


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def render_text_table(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"=== ARNÊS DE ACURÁCIA TRI — ENEM {report['year']} ===")
    lines.append("")
    lines.append("[Nota ponta-a-ponta vs NU_NOTA oficial — k-fold hold-out]")
    lines.append(f"{'Área':<5} {'n':>7} {'MAE':>7} {'IC(MAE)':>16} {'RMSE':>7} {'r':>6}")
    for area in ("LC", "CH", "CN", "MT"):
        e = report["end_to_end"].get(area)
        if not e:
            continue
        lo, hi = e.get("mae_ci", (float("nan"), float("nan")))
        lines.append(
            f"{area:<5} {e['n']:>7} {_fmt(e['mae']):>7} "
            f"[{_fmt(lo)},{_fmt(hi)}]".rjust(16)
            + f" {_fmt(e['rmse']):>7} {_fmt(e['r']):>6}"
        )
    lines.append("")
    lines.append("[Atribuição de erro (CN/MT, AZUL) — quanto vem de calibração vs ancoragem]")
    lines.append(f"{'Área':<5} {'MAE_total':>10} {'MAE_anc':>10} {'calib':>8} {'IC(calib)':>16}")
    for area in ("CN", "MT"):
        a = report["attribution"].get(area)
        if not a:
            continue
        lo, hi = a["calib_contrib_ci"]
        lines.append(
            f"{area:<5} {_fmt(a['mae_full']):>10} {_fmt(a['mae_anchor']):>10} "
            f"{_fmt(a['calib_contrib_mean']):>8} "
            + f"[{_fmt(lo)},{_fmt(hi)}]".rjust(16)
        )
    lines.append("")
    lines.append("[Recuperação de parâmetro (CN/MT) — nosso a,b vs oficial INEP, por linking]")
    for area in ("CN", "MT"):
        rec = report["recovery"].get(area, {})
        for ano, by_link in rec.items():
            for link_name, by_param in by_link.items():
                ra = by_param["a"]
                rb = by_param["b"]
                lines.append(
                    f"{area} {ano} {link_name:<13} "
                    f"a: r={_fmt(ra['r'])} rmse={_fmt(ra['rmse'])} | "
                    f"b: r={_fmt(rb['r'])} rmse={_fmt(rb['rmse'])}"
                )
    return "\n".join(lines)


def write_report(report: dict, out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_accuracy_harness.py -k "build_and_render" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/report.py tests/test_accuracy_harness.py
git commit -m "feat(evaluation): relatorio JSON + tabela texto do arnes"
```

---

## Task 8: `__main__.py` — CLI orquestrando o arnês completo

**Files:**
- Create: `app/evaluation/__main__.py`
- Test: `tests/test_accuracy_harness.py`

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_accuracy_harness.py  (adicionar)
@pytest.mark.unit
def test_run_recovery_for_group_end_to_end():
    """Recuperação de parâmetro num grupo sintético com gabarito casado."""
    from app.evaluation.__main__ import run_recovery_for_group

    rng = np.random.default_rng(5)
    # 45 itens, gabarito "A" em todas as posições; official b conhecido
    gabarito = "A" * 45
    b_official = np.linspace(-2, 2, 45)
    official_items = [
        {"a": 1.2, "b": float(b_official[j]), "c": 0.2, "posicao": 91 + j, "gabarito": "A"}
        for j in range(45)
    ]
    # respostas sintéticas com sinal (theta latente)
    theta = rng.normal(0, 1, size=800)
    p = 0.2 + 0.8 / (1 + np.exp(-1.3 * (theta[:, None] - b_official[None, :])))
    resp = (rng.uniform(size=p.shape) < p).astype(int)

    rec = run_recovery_for_group(resp, gabarito, official_items)
    # ambos os linkings presentes, com r de b alto (recuperação sã)
    assert set(rec.keys()) == {"mean_sigma", "stocking_lord"}
    assert rec["stocking_lord"]["b"]["r"] > 0.9
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_accuracy_harness.py -k "run_recovery_for_group" -v`
Expected: FAIL — `ModuleNotFoundError` / `cannot import name 'run_recovery_for_group'`

- [ ] **Step 3: Implementar**

```python
# app/evaluation/__main__.py
"""CLI do arnês de acurácia: python -m app.evaluation --year 2024

Orquestra: carrega microdata real → k-fold ponta-a-ponta (4 áreas) →
recuperação de parâmetro + atribuição (CN/MT/AZUL) → relatório JSON + tabela.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from app.evaluation.attribution import decompose_error, score_with_params
from app.evaluation.ground_truth import (
    align_official_to_gabarito,
    load_official_params,
    official_coprova,
    official_items,
)
from app.evaluation.kfold import (
    build_group_matrix,
    kfold_oof_predictions,
    run_kfold_area,
)
from app.evaluation.linking import (
    apply_linking,
    mean_sigma_linking,
    stocking_lord_linking,
)
from app.evaluation.metrics import bootstrap_ci, mae, param_recovery_stats, pearson_r, rmse
from app.evaluation.report import build_report, render_text_table, write_report
from app.microdata.calibration import calibrar_itens_fast
from app.microdata.loader import default_csv_path, load_microdata, load_primeira_aplicacao_provas
from app.microdata.schema import detect_schema
from app.tabela_referencia import TabelaReferenciaTRI

ATTR_AREAS = ("CN", "MT")
ALL_AREAS = ("LC", "CH", "CN", "MT")


def _recovery_stats_pair(ours_linked: list[dict], official: list[dict]) -> dict:
    a_ours = np.array([p["a"] for p in ours_linked])
    b_ours = np.array([p["b"] for p in ours_linked])
    a_off = np.array([p["a"] for p in official])
    b_off = np.array([p["b"] for p in official])
    return {
        "a": param_recovery_stats(a_ours, a_off),
        "b": param_recovery_stats(b_ours, b_off),
    }


def run_recovery_for_group(
    resp_matrix: np.ndarray, group_gabarito: str, official: list[dict]
) -> dict:
    """Calibra o grupo, valida/aligna ao oficial, linka e mede recuperação."""
    valid_cols, official_valid = align_official_to_gabarito(official, group_gabarito)
    ours_all = calibrar_itens_fast(resp_matrix, c_strategy="fixed")
    ours = [ours_all[j] for j in valid_cols]

    b_ours = np.array([p["b"] for p in ours])
    b_off = np.array([p["b"] for p in official_valid])
    link_ms = mean_sigma_linking(b_ours, b_off)
    link_sl = stocking_lord_linking(ours, official_valid)

    return {
        "mean_sigma": _recovery_stats_pair(apply_linking(ours, link_ms), official_valid),
        "stocking_lord": _recovery_stats_pair(apply_linking(ours, link_sl), official_valid),
    }


def _azul_group_matrix(df, schema, area, co_prova_azul):
    co_col = schema.co_prova[area]
    g = df[df[co_col].astype(str) == str(co_prova_azul)]
    resp, nota_ref = build_group_matrix(g, schema, area)
    # gabarito do grupo (todos compartilham): primeira linha válida
    gab = g[schema.tx_gabarito[area]].astype(str).iloc[0] if len(g) else ""
    return resp, nota_ref, gab


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Arnês de acurácia TRI")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--c-strategy", default="fixed", choices=["fixed", "bayes"])
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--out", default="accuracy_report.json")
    args = parser.parse_args(argv)

    csv_path = args.csv_path or default_csv_path(args.year)
    # FALHA ALTO E CLARO se o dado-base não existe (nada de skip silencioso).
    from pathlib import Path
    if not Path(csv_path).is_file():
        print(
            f"ERRO: microdata não encontrado em {csv_path}. "
            f"Baixe MICRODADOS_ENEM_{args.year}.csv do INEP.",
            file=sys.stderr,
        )
        return 2

    print(f"Carregando microdata {args.year} (amostra {args.sample_size})...", file=sys.stderr)
    whitelist = load_primeira_aplicacao_provas(args.year)
    df = load_microdata(
        args.year, csv_path=csv_path, sample_size=args.sample_size,
        provas_primeira=whitelist,
    )
    schema = detect_schema(csv_path)

    tabelas = TabelaReferenciaTRI.carregar_multiano()
    tabela = tabelas.get(args.year) or TabelaReferenciaTRI()

    # 1) Ponta-a-ponta, 4 áreas
    end_to_end: dict = {}
    for area in ALL_AREAS:
        print(f"k-fold {area}...", file=sys.stderr)
        r = run_kfold_area(df, schema, area, tabela, k=args.kfolds, c_strategy=args.c_strategy)
        if r["n"] == 0:
            continue
        lo, hi = bootstrap_ci(np.abs(r["pred"] - r["ref"]))
        end_to_end[area] = {
            "area": area, "n": r["n"], "mae": mae(r["pred"], r["ref"]),
            "rmse": rmse(r["pred"], r["ref"]), "r": pearson_r(r["pred"], r["ref"]),
            "mae_ci": (lo, hi),
        }

    # 2) Atribuição + recuperação, CN/MT, AZUL
    params_oficiais = load_official_params()
    attribution: dict = {}
    recovery: dict = {}
    for area in ATTR_AREAS:
        co_azul = official_coprova(params_oficiais, args.year, area)
        official = official_items(params_oficiais, args.year, area)
        resp, nota_ref, gab = _azul_group_matrix(df, schema, area, co_azul)
        if len(resp) < 500:
            print(f"AVISO: grupo AZUL {area} pequeno ({len(resp)}) — pulado.", file=sys.stderr)
            continue
        try:
            valid_cols, official_valid = align_official_to_gabarito(official, gab)
        except ValueError as exc:
            print(f"AVISO: alinhamento {area} falhou: {exc}", file=sys.stderr)
            continue

        # Atribuição: OOF nossos params vs scoring com params oficiais
        our_oof = kfold_oof_predictions(resp, area, tabela, k=args.kfolds, c_strategy=args.c_strategy)
        official_pred = score_with_params(resp[:, valid_cols], official_valid, area, tabela)
        mask = ~np.isnan(our_oof) & ~np.isnan(nota_ref) & (nota_ref > 0)
        attribution[area] = decompose_error(our_oof[mask], official_pred[mask], nota_ref[mask])

        # Recuperação de parâmetro (2024 e, se houver dados, 2017 via outro fluxo)
        recovery[area] = {str(args.year): run_recovery_for_group(resp, gab, official)}

    report = build_report(
        year=args.year, end_to_end=end_to_end,
        attribution=attribution, recovery=recovery,
    )
    write_report(report, args.out)
    print(render_text_table(report))
    print(f"\nRelatório JSON salvo em: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_accuracy_harness.py -k "run_recovery_for_group" -v`
Expected: PASS

- [ ] **Step 5: Verificar o CLI manualmente (smoke, dado real)**

Run: `python -m app.evaluation --year 2024 --sample-size 20000 --out /tmp/smoke_report.json`
Expected: imprime a tabela texto com seções de ponta-a-ponta (4 áreas) e atribuição (CN/MT); cria `/tmp/smoke_report.json`. (Amostra reduzida só para checar o encanamento.)

- [ ] **Step 6: Commit**

```bash
git add app/evaluation/__main__.py tests/test_accuracy_harness.py
git commit -m "feat(evaluation): CLI do arnes de acuracia (k-fold + atribuicao + recuperacao)"
```

---

## Task 9: Endurecer `TestRealINEPBenchmark` — caminho correto + skip barulhento

**Files:**
- Modify: `tests/test_accuracy_benchmarks.py:149-220`
- Test: o próprio arquivo (rodar a suíte)

- [ ] **Step 1: Escrever o teste que falha (skip barulhento)**

Adicionar ao final de `tests/test_accuracy_harness.py` um teste que garante que o helper de caminho agora resolve para o diretório correto:

```python
# tests/test_accuracy_harness.py  (adicionar)
@pytest.mark.unit
def test_benchmark_cache_path_uses_corrected_dir():
    from app.microdata.loader import default_csv_path

    # Após Task 1, default_csv_path aponta para o diretório real (sem apps/apps).
    p = default_csv_path(2024)
    assert "apps/apps" not in str(p)
```

- [ ] **Step 2: Rodar e ver passar (já corrigido na Task 1)**

Run: `pytest tests/test_accuracy_harness.py::test_benchmark_cache_path_uses_corrected_dir -v`
Expected: PASS (a correção da Task 1 já garante isso)

- [ ] **Step 3: Tornar o skip do benchmark real barulhento e auto-explicativo**

Em `tests/test_accuracy_benchmarks.py`, na fixture `calibrated_df` (linha ~186-191), trocar o skip mudo por um que diz exatamente o que falta e como gerar:

```python
    @pytest.fixture(scope="class")
    def calibrated_df(self):
        path = _calibrated_cache_path(2024)
        if path is None:
            pytest.skip(
                "Cache 2024 calibrado ausente. Gere com: "
                "`python -m app.evaluation --year 2024` (ou o pipeline V5 'prepare'). "
                "Sem ele, o guardrail de acuracia real NAO roda."
            )
        return pd.read_parquet(path)
```

- [ ] **Step 4: Rodar a suíte de benchmark (confirmar que não quebrou)**

Run: `pytest tests/test_accuracy_benchmarks.py -v`
Expected: testes sintéticos PASS; `TestRealINEPBenchmark` SKIP com a mensagem nova explicando como gerar o cache (não mais skip mudo).

- [ ] **Step 5: Commit**

```bash
git add tests/test_accuracy_benchmarks.py tests/test_accuracy_harness.py
git commit -m "test(accuracy): skip barulhento no benchmark real + verificacao do caminho"
```

---

## Task 10: Rodar e congelar o baseline

**Files:**
- Create: `docs/superpowers/baselines/2026-06-04-accuracy-baseline-2024.json` (saída do arnês)

- [ ] **Step 1: Rodar o arnês completo no dado real**

Run:
```bash
python -m app.evaluation --year 2024 --out docs/superpowers/baselines/2026-06-04-accuracy-baseline-2024.json
```
Expected: tabela texto completa (ponta-a-ponta 4 áreas com IC; atribuição CN/MT com parcela de calibração e IC; recuperação CN/MT sob mean-sigma e Stocking-Lord). JSON salvo.

- [ ] **Step 2: Inspecionar o relatório**

Run: `python -c "import json,sys; d=json.load(open('docs/superpowers/baselines/2026-06-04-accuracy-baseline-2024.json')); print(json.dumps(d['attribution'], indent=2, ensure_ascii=False))"`
Expected: para CN e MT, ver `mae_full`, `mae_anchor`, `calib_contrib_mean` e seu IC — isso revela **qual fonte de erro domina** (insumo da Fase 2).

- [ ] **Step 3: Rodar a suíte completa do novo módulo (regressão)**

Run: `pytest tests/test_accuracy_harness.py -v`
Expected: todos PASS.

- [ ] **Step 4: Commit do baseline**

```bash
git add docs/superpowers/baselines/2026-06-04-accuracy-baseline-2024.json
git commit -m "chore(evaluation): congela baseline de acuracia 2024 (pre-tuning)"
```

- [ ] **Step 5: Registrar o veredito do baseline no spec da Fase 2**

Anotar, em uma nota ao fim do spec, qual área/fonte de erro domina (calibração vs. ancoragem) segundo o baseline — esse é o gatilho que define as alavancas da Fase 2.

---

## Fase 2 (plano-irmão, deferido)

A Fase 2 (tuning dirigido) **só pode ser detalhada após o baseline**, porque as alavancas dependem de qual fonte de erro domina:
- **Calibração domina** → `--c-strategy bayes`; prior/shrinkage em `a`; revisar saturação `b≥4.8`; tolerância de convergência do EM.
- **Ancoragem domina** → interpolação por spline monotônica em `transform._interpolar_faixa_referencia`; colapso de teto em 0/45 acertos; entrada `LC=45`.

Cada candidato é avaliado re-rodando o arnês (`--c-strategy`/flags) e exigindo que `paired_diff_ci(err_candidato, err_baseline)` retorne IC que **exclui 0** para a melhora. Ao aceitar, apertar os floors em `_INEP_BENCHMARK`. Gerar o plano `docs/superpowers/plans/<data>-tri-accuracy-tuning.md` invocando novamente a skill writing-plans, com o baseline em mãos.

---

## Self-Review (preenchido)

- **Cobertura do spec:** §1 bug→Task 1; §2 verdade-base→Tasks 4,5,8; §3 linking duplo→Task 3,8; §4 atribuição→Task 6,8; §5 Fase 1 (k-fold, bootstrap, relatório, baseline)→Tasks 2,5,7,8,10; §6 Fase 2→deferida; §7 estrutura de código→Tasks 2-8; §8 CLI+guard→Tasks 8,9; §9 validação/erros→Task 4 (gabarito), Task 8 (falha alta); §10 testes→todas; §11 critérios→Task 10. ✔
- **Placeholders:** nenhum "TODO/TBD"; todo passo tem código/comando concretos. ✔
- **Consistência de tipos:** `Linking(A,B)` usado igual em 3,8; item param `dict{a,b,c,saturado?}` consistente entre `calibrar_itens_fast`/`eap_theta_batch`/`apply_linking`/`score_with_params`; resultado k-fold `{area,n,pred,ref}` consistente entre 5 e 8; `decompose_error` retorna `calib_contrib_ci` consumido em `render_text_table`. ✔
