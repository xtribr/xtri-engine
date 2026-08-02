"""Endpoints V6 — TRI em escala reduzida (provas de 10-45 itens)."""

import logging
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import state
from ..csv_parser import parse_single_area
from ..irt.reduced_scale import (
    AVISO_METODOLOGICO_REDUZIDO,
    METODO_ANCORAGEM_REDUZIDA,
    MODO_SCORING_REDUZIDO,
    N_ITENS_REFERENCIA_ENEM,
    ancorar_nota_reduzida,
)
from ..models import ItemParametros, RequestTRIReduzida, ResponseTRIReduzida
from ..motores.cientifico import pipeline_cientifico
from .common import _parse_itens_anulados, _remover_itens_anulados

logger = logging.getLogger(__name__)

router = APIRouter()


def _remover_itens_anulados_por_posicao(
    alunos: list[dict[str, Any]],
    itens_anulados: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove itens anulados de payload JSON usando posição 1-based."""

    anulados = {int(i) for i in itens_anulados if int(i) > 0}
    if not anulados:
        return alunos, []

    removidos = [str(i) for i in sorted(anulados)]
    novos: list[dict[str, Any]] = []
    for aluno in alunos:
        respostas = list(aluno["respostas"])
        filtradas = [
            resp for idx, resp in enumerate(respostas, start=1)
            if idx not in anulados
        ]
        novos.append({**aluno, "respostas": filtradas})
    return novos, removidos


def _validar_matriz_reduzida(alunos: list[dict[str, Any]]) -> int:
    """Valida matriz reduzida e retorna número de itens válidos."""

    if len(alunos) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Mínimo de 10 alunos para calibração reduzida. Recebido: {len(alunos)}",
        )
    n_itens = len(alunos[0]["respostas"]) if alunos else 0
    if n_itens < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Escala reduzida exige mínimo 10 itens válidos. Recebido: {n_itens}",
        )
    if n_itens > N_ITENS_REFERENCIA_ENEM:
        raise HTTPException(
            status_code=400,
            detail=f"Escala reduzida aceita no máximo {N_ITENS_REFERENCIA_ENEM} itens válidos. Recebido: {n_itens}",
        )
    for aluno in alunos:
        respostas = aluno["respostas"]
        if len(respostas) != n_itens:
            raise HTTPException(
                status_code=400,
                detail=f"Aluno {aluno['aluno_id']} tem {len(respostas)} respostas, esperado {n_itens}",
            )
        if any(r not in (0, 1) for r in respostas):
            raise HTTPException(
                status_code=400,
                detail=f"Aluno {aluno['aluno_id']} contém respostas fora de 0/1",
            )
    return n_itens


def _processar_tri_reduzida(
    *,
    simulado_id: str,
    area: str,
    ano_referencia: int,
    c_fixo: float,
    alunos: list[dict[str, Any]],
    avisos: Optional[list[str]] = None,
) -> ResponseTRIReduzida:
    """Executa calibração local e projeta notas em escala ENEM reduzida."""

    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(
            status_code=400,
            detail=f"Área inválida: {area}. Esperado: LC, CH, CN ou MT",
        )
    tabelas_multiano = state.get_tabelas_multiano()
    if ano_referencia not in tabelas_multiano:
        raise HTTPException(
            status_code=400,
            detail=f"Ano {ano_referencia} nao disponivel. Opcoes: {sorted(tabelas_multiano.keys())}",
        )

    n_itens_validos = _validar_matriz_reduzida(alunos)
    tab = tabelas_multiano[ano_referencia]

    try:
        resultado = pipeline_cientifico(
            alunos=alunos,
            area=area,
            tabela=tab,
            c_fixo=c_fixo,
        )
    except Exception as exc:
        logger.exception("tri reduzida failed for area=%s", area)
        raise HTTPException(status_code=500, detail=f"Erro na calibração reduzida: {exc}")

    resultados = []
    notas = []
    for r in resultado["resultados_alunos"]:
        proj = ancorar_nota_reduzida(
            theta=float(r["theta"]),
            se_theta=float(r.get("se", 0.0)),
            area=area,
            acertos_observados=int(r["acertos"]),
            n_itens_validos=n_itens_validos,
            tabela=tab,
        )
        notas.append(proj["nota_enem_projetada"])
        resultados.append({
            "aluno_id": r["aluno_id"],
            "nome": r.get("nome", ""),
            **proj,
        })

    diagnostico = dict(resultado["diagnostico"])
    diagnostico.update({
        "modo_scoring": MODO_SCORING_REDUZIDO,
        "modo_escala": "reduzida",
        "n_itens_validos": n_itens_validos,
        "n_itens_referencia": N_ITENS_REFERENCIA_ENEM,
        "nota_media_projetada": round(float(np.mean(notas)), 1) if notas else 0.0,
        "nota_min_projetada": round(float(np.min(notas)), 1) if notas else 0.0,
        "nota_max_projetada": round(float(np.max(notas)), 1) if notas else 0.0,
    })

    parametros = [
        ItemParametros(
            item_index=i,
            a=p["a"],
            b=p["b"],
            c=p["c"],
            taxa_acerto=p["taxa_acerto"],
            saturado=p["saturado"],
        )
        for i, p in enumerate(resultado["parametros_itens"])
    ]

    return ResponseTRIReduzida(
        status="sucesso",
        simulado_id=simulado_id,
        area=area,
        ano_referencia=ano_referencia,
        modo_scoring=MODO_SCORING_REDUZIDO,
        modo_escala="reduzida",
        n_itens_referencia=N_ITENS_REFERENCIA_ENEM,
        n_itens_validos=n_itens_validos,
        n_alunos=len(alunos),
        metodo_ancoragem=METODO_ANCORAGEM_REDUZIDA,
        limites_uso=AVISO_METODOLOGICO_REDUZIDO,
        resultados=resultados,
        parametros_itens=parametros,
        diagnostico=diagnostico,
        avisos=avisos or [],
    )


@router.post("/api/v6/tri/reduzida", response_model=ResponseTRIReduzida)
async def tri_reduzida(payload: RequestTRIReduzida):
    """Corrige provas de 10-45 itens com nota ENEM projetada."""

    alunos = [
        {
            "aluno_id": a.aluno_id,
            "nome": a.nome or "",
            "respostas": [int(v) for v in a.respostas],
        }
        for a in payload.alunos
    ]
    avisos: list[str] = []
    alunos, removidos = _remover_itens_anulados_por_posicao(alunos, payload.itens_anulados)
    if removidos:
        avisos.append(
            "Itens anulados removidos da calibração e do denominador: "
            + ", ".join(removidos)
        )
    return _processar_tri_reduzida(
        simulado_id=payload.simulado_id,
        area=payload.area,
        ano_referencia=payload.ano_referencia,
        c_fixo=payload.c_fixo,
        alunos=alunos,
        avisos=avisos,
    )


@router.post("/api/v6/tri/reduzida/upload", response_model=ResponseTRIReduzida)
async def tri_reduzida_upload(
    simulado_id: str = Form(...),
    area: str = Form(...),
    ano_referencia: int = Form(...),
    arquivo: UploadFile = File(...),
    c_fixo: float = Form(0.20),
    itens_anulados: str = Form(""),
):
    """Upload CSV/XLSX para TRI reduzida."""

    raw = await arquivo.read()
    filename = (arquivo.filename or "").lower()
    is_xlsx = filename.endswith(".xlsx") or filename.endswith(".xls")
    try:
        if is_xlsx:
            parsed = parse_single_area(raw, is_xlsx=True)
        else:
            try:
                content = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                content = raw.decode("latin-1")
            parsed = parse_single_area(content, is_xlsx=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    avisos: list[str] = list(parsed["avisos"])
    anulados = _parse_itens_anulados(itens_anulados)
    if anulados:
        removidos = _remover_itens_anulados(parsed, anulados)
        if removidos:
            avisos.append(
                "Itens anulados removidos da calibração e do denominador: "
                + ", ".join(removidos)
            )
        else:
            avisos.append(
                "Itens anulados informados, mas nenhum deles foi encontrado "
                f"no arquivo: {sorted(anulados)}"
            )

    alunos = [
        {"aluno_id": a["aluno_id"], "nome": "", "respostas": a["respostas"]}
        for a in parsed["alunos"]
    ]
    return _processar_tri_reduzida(
        simulado_id=simulado_id,
        area=area,
        ano_referencia=ano_referencia,
        c_fixo=c_fixo,
        alunos=alunos,
        avisos=avisos,
    )
