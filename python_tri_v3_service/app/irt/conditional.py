"""
Ancoragem condicional aos acertos.

A faixa de referência `tri_min..tri_med..tri_max` descreve as notas do ENEM
entre quem teve o mesmo número de acertos. A posição do aluno dentro dela
precisa, portanto, ser relativa a esse grupo — e não ao theta absoluto, que
acompanha os acertos e empurra a nota para o mínimo (poucos acertos) ou para
o máximo (muitos acertos) da faixa.

    u    = Φ(w · (theta − m(k)) / s(k))
    nota = quantil u das notas de referência para k acertos

- `m(k)` e `s(k)`: média e desvio do theta EAP entre quem faz `k` acertos,
  derivados dos parâmetros dos itens por simulação sob o prior N(0,1).
- `w = min(0,9; n / (n + 25))`: encolhe a posição em direção à mediana quando
  a calibração veio de poucos alunos (parâmetros ruidosos).

Validação contra notas oficiais (ENEM 2024 e 2025, fora da amostra de
calibração): erro absoluto médio de 3–11 pontos, contra 29–39 da regra de
theta absoluto e 10–31 da mediana por acertos.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .scoring import P

# Percentis guardados nas tabelas de quantis (0 = mínimo, 1 = máximo).
PROBS_QUANTIS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
U_MIN, U_MAX = 0.01, 0.99
PESO_MAXIMO = 0.9
PESO_N_DESCONHECIDO = 0.5
N_MEIO_PESO = 25
# Sem percentis, mínimo e máximo são tratados como extremos a ~3,5 desvios.
Z_EXTREMOS = 3.5
N_SIMULACAO = 20_000
SEMENTE_SIMULACAO = 2026
N_QUAD = 61

CurvaCondicional = dict[str, Any]


def peso_por_tamanho(n_calibracao: Optional[float]) -> float:
    """Confiança na posição relativa, crescente com o tamanho da calibração."""

    if n_calibracao is None or not np.isfinite(n_calibracao) or n_calibracao <= 0:
        return PESO_N_DESCONHECIDO
    return float(min(PESO_MAXIMO, n_calibracao / (n_calibracao + N_MEIO_PESO)))


def _isotonica(valores: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Regressão isotônica ponderada (pool adjacent violators), não decrescente."""

    blocos: list[list[float]] = []  # [média, peso, tamanho]
    for v, w in zip(valores, pesos):
        blocos.append([float(v), float(w), 1])
        while len(blocos) > 1 and blocos[-2][0] > blocos[-1][0]:
            m2, w2, n2 = blocos.pop()
            m1, w1, n1 = blocos.pop()
            blocos.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(int(n), m) for m, _, n in blocos])


def ajustar_curva(
    acertos: Sequence[int],
    thetas: Sequence[float],
    n_calibracao: Optional[float] = None,
    n_min_desvio: int = 10,
    piso_desvio: float = 0.03,
) -> CurvaCondicional:
    """Ajusta `m(k)` monotônica e `s(k)` suavizado a partir de pares (acertos, theta)."""

    k = np.asarray(acertos, dtype=int)
    theta = np.asarray(thetas, dtype=float)
    ks = np.unique(k)
    tamanho = np.array([(k == v).sum() for v in ks], dtype=float)
    media = _isotonica(np.array([theta[k == v].mean() for v in ks]), tamanho)

    residuo = theta - np.interp(k, ks, media)
    desvio_geral = float(residuo.std()) if len(theta) > 1 else piso_desvio
    confiavel = tamanho >= n_min_desvio
    if confiavel.sum() >= 2:
        por_grupo = np.array([theta[k == v].std(ddof=1) if c else np.nan for v, c in zip(ks, confiavel)])
        desvio = np.interp(ks, ks[confiavel], por_grupo[confiavel])
        nucleo = np.ones(5)
        desvio = np.convolve(desvio * tamanho, nucleo, "same") / np.convolve(tamanho, nucleo, "same")
    else:
        desvio = np.full(len(ks), desvio_geral)
    desvio = np.maximum(np.nan_to_num(desvio, nan=piso_desvio), piso_desvio)
    return {
        "acertos": [int(v) for v in ks],
        "theta_medio": [round(float(v), 4) for v in media],
        "theta_dp": [round(float(v), 4) for v in desvio],
        "n_calibracao": None if n_calibracao is None else int(n_calibracao),
        "peso": round(peso_por_tamanho(n_calibracao), 4),
    }


def _eap_lote(resp: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, usar: np.ndarray) -> np.ndarray:
    """EAP vetorizado com prior N(0,1), ignorando itens saturados como `eap_theta`."""

    grade = np.linspace(-5, 5, N_QUAD)
    prob = np.clip(P(grade[:, None], a[None, usar], b[None, usar], c[None, usar]), 1e-9, 1 - 1e-9)
    r = resp[:, usar].astype(float)
    log_lk = r @ np.log(prob).T + (1.0 - r) @ np.log(1 - prob).T
    log_post = log_lk + norm.logpdf(grade)[None, :]
    post = np.exp(log_post - log_post.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    return post @ grade


@lru_cache(maxsize=64)
def _curva_teorica_cache(itens: tuple[tuple[float, float, float, bool], ...]) -> tuple:
    a = np.array([i[0] for i in itens])
    b = np.array([i[1] for i in itens])
    c = np.array([i[2] for i in itens])
    usar = ~np.array([i[3] for i in itens], dtype=bool)
    rng = np.random.default_rng(SEMENTE_SIMULACAO)
    theta = rng.standard_normal(N_SIMULACAO)
    prob = P(theta[:, None], a[None, :], b[None, :], c[None, :])
    resp = (rng.random(prob.shape) < prob).astype(np.int8)
    if not usar.any():
        return ((0, len(itens)), (0.0, 0.0), (1.0, 1.0))
    curva = ajustar_curva(resp.sum(axis=1), _eap_lote(resp, a, b, c, usar))
    return (tuple(curva["acertos"]), tuple(curva["theta_medio"]), tuple(curva["theta_dp"]))


def curva_teorica(
    item_params: Sequence[Mapping[str, Any]],
    n_calibracao: Optional[float] = None,
) -> CurvaCondicional:
    """Curva `m(k)`, `s(k)` implicada pelos parâmetros dos itens sob o prior N(0,1)."""

    itens = tuple(
        (round(float(p["a"]), 4), round(float(p["b"]), 4), round(float(p.get("c", 0.20)), 4),
         bool(p.get("saturado", False)))
        for p in item_params
    )
    acertos, media, desvio = _curva_teorica_cache(itens)
    return {
        "acertos": list(acertos),
        "theta_medio": list(media),
        "theta_dp": list(desvio),
        "n_calibracao": None if n_calibracao is None else int(n_calibracao),
        "peso": round(peso_por_tamanho(n_calibracao), 4),
    }


def posicao_relativa(theta: float, acertos: float, curva: CurvaCondicional) -> float:
    """`u` em [0,01; 0,99]: posição do aluno entre quem teve o mesmo acerto."""

    media = float(np.interp(acertos, curva["acertos"], curva["theta_medio"]))
    desvio = max(float(np.interp(acertos, curva["acertos"], curva["theta_dp"])), 1e-6)
    peso = float(curva.get("peso", PESO_N_DESCONHECIDO))
    return float(np.clip(norm.cdf(peso * (float(theta) - media) / desvio), U_MIN, U_MAX))


def nota_por_posicao(
    u: float,
    *,
    ref_min: float,
    ref_med: float,
    ref_max: float,
    quantis: Optional[Sequence[float]] = None,
) -> float:
    """Converte a posição `u` em nota dentro da faixa do número de acertos."""

    if quantis is not None:
        return float(np.interp(u, PROBS_QUANTIS, np.sort(np.asarray(quantis, dtype=float))))
    if ref_min == ref_max:
        return float(ref_med)
    z = float(norm.ppf(u))
    lado = (ref_max - ref_med) if z > 0 else (ref_med - ref_min)
    return float(np.clip(ref_med + z * lado / Z_EXTREMOS, ref_min, ref_max))
