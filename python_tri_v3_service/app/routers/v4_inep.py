"""Endpoints V4 — Validação INEP (usa parâmetros oficiais do Supabase)."""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import state
from ..csv_parser import parse_single_area
from ..db.supabase_client import get_supabase, is_supabase_configured
from ..models import (
    ENEMInfo,
    EstatisticasComparacao,
    ItemINEP,
    ProvaINEP,
    RequestCalcularComINEP,
    ResponseCalcularComINEP,
    ResponseValidarSimulado,
    ResultadoAlunoINEP,
    ResultadoComparacao,
    TransformacaoENEM,
)
from ..motores.cientifico import (
    eap_theta as eap_theta_cientifico,
    estado_tri,
    pipeline_cientifico,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _fetch_all_supabase(client, table: str, filter_eq: Optional[Dict] = None,
                         select: str = "*", page_size: int = 1000):
    """Paginate all rows from a Supabase table with optional filters."""
    rows = []
    offset = 0
    while True:
        q = client.table(table).select(select).range(offset, offset + page_size - 1)
        if filter_eq:
            for k, v in filter_eq.items():
                q = q.eq(k, v)
        r = q.execute()
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < page_size:
            break
        offset += page_size
    return rows


def _require_supabase():
    """Raise HTTPException if Supabase is not configured or unreachable."""
    if not is_supabase_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase não configurado. Defina SUPABASE_URL e SUPABASE_KEY no .env",
        )
    client = get_supabase()
    if client is None:
        raise HTTPException(status_code=503, detail="Cliente Supabase indisponível")
    return client


@router.get("/api/v4/enem/info", response_model=ENEMInfo)
async def enem_info():
    """
    Retorna sumário do que está disponível no item bank ENEM oficial:
    anos com transformações calibradas, totais de itens e provas.
    """
    client = _require_supabase()

    # Transformações
    transf_rows = _fetch_all_supabase(client, "enem_transformacoes")
    transformacoes = [
        TransformacaoENEM(
            ano=r["ano"], area=r["area"],
            sigma=float(r["sigma"]), mu=float(r["mu"]),
            n_alunos=r["n_alunos"] or 0,
            correlacao=float(r["correlacao"] or 0),
            mae=float(r["mae"] or 0),
        )
        for r in transf_rows
    ]

    anos = sorted({t.ano for t in transformacoes})
    areas = sorted({t.area for t in transformacoes})

    # Totais
    total_itens = client.table("enem_itens").select("id", count="exact", head=True).execute().count or 0
    total_provas_r = client.table("enem_itens_prova").select("id", count="exact", head=True).execute().count or 0

    return ENEMInfo(
        anos_disponiveis=anos,
        areas=areas,
        transformacoes=sorted(transformacoes, key=lambda t: (t.ano, t.area)),
        total_itens=total_itens,
        total_provas=total_provas_r,
    )


@router.get("/api/v4/enem/transformacao/{ano}/{area}", response_model=TransformacaoENEM)
async def enem_transformacao(ano: int, area: str):
    """Retorna a transformação empírica (σ, μ) para um ano × área."""
    client = _require_supabase()
    area = area.upper()

    r = client.table("enem_transformacoes").select("*").eq("ano", ano).eq("area", area).limit(1).execute()
    if not r.data:
        raise HTTPException(
            status_code=404,
            detail=f"Transformação não encontrada para {area}/{ano}",
        )
    row = r.data[0]
    return TransformacaoENEM(
        ano=row["ano"], area=row["area"],
        sigma=float(row["sigma"]), mu=float(row["mu"]),
        n_alunos=row["n_alunos"] or 0,
        correlacao=float(row["correlacao"] or 0),
        mae=float(row["mae"] or 0),
    )


@router.get("/api/v4/enem/provas/{ano}/{area}")
async def enem_provas(ano: int, area: str):
    """
    Lista todas as CO_PROVA disponíveis no banco para um ano × área.

    Retorna:
        [{"co_prova": int, "descricao": str, "is_reaplicacao": bool}, ...]
    """
    client = _require_supabase()
    area = area.upper()

    r = client.table("enem_codigos_prova").select(
        "co_prova,descricao_categoria,is_reaplicacao"
    ).eq("ano", ano).eq("area", area).execute()

    items = [
        {
            "co_prova": row["co_prova"],
            "descricao": row.get("descricao_categoria") or "",
            "is_reaplicacao": bool(row.get("is_reaplicacao")),
        }
        for row in r.data
    ]
    items.sort(key=lambda x: x["co_prova"])
    return items


@router.get("/api/v4/enem/itens/{ano}/{area}/{co_prova}", response_model=ProvaINEP)
async def enem_itens_prova(ano: int, area: str, co_prova: int):
    """Retorna parâmetros oficiais dos itens de uma prova específica."""
    client = _require_supabase()
    area = area.upper()

    # Metadata da prova
    meta_r = client.table("enem_codigos_prova").select("*").eq("ano", ano).eq("area", area).eq("co_prova", co_prova).limit(1).execute()
    if not meta_r.data:
        raise HTTPException(status_code=404, detail=f"Prova {co_prova} não encontrada para {area}/{ano}")
    meta = meta_r.data[0]

    # Itens da área
    itens_area_r = client.table("enem_itens").select("*").eq("ano", ano).eq("area", area).execute()
    itens_by_co = {it["co_item"]: it for it in itens_area_r.data}

    # Mapeamentos dessa prova
    maps_r = client.table("enem_itens_prova").select("*").eq("ano", ano).eq("co_prova", co_prova).execute()

    itens = []
    for m in maps_r.data:
        it = itens_by_co.get(m["co_item"])
        if not it or it.get("param_discriminacao") is None:
            continue
        itens.append(ItemINEP(
            co_item=it["co_item"],
            posicao=m["posicao"],
            gabarito=it.get("gabarito"),
            a=float(it["param_discriminacao"]),
            b=float(it["param_dificuldade"]),
            c=float(it.get("param_acerto_casual") or 0.20),
            tp_lingua=it.get("tipo_lingua"),
            abandonado=bool(it.get("item_abandonado")),
        ))

    if not itens:
        raise HTTPException(status_code=404, detail=f"Nenhum item encontrado para prova {co_prova}")

    itens.sort(key=lambda x: x.posicao)

    return ProvaINEP(
        ano=ano, area=area, co_prova=co_prova,
        descricao=meta.get("descricao_categoria"),
        is_reaplicacao=bool(meta.get("is_reaplicacao")),
        itens=itens,
    )


@router.post("/api/v4/enem/calcular-com-inep", response_model=ResponseCalcularComINEP)
async def calcular_com_inep(payload: RequestCalcularComINEP):
    """
    Calcula notas TRI usando parâmetros OFICIAIS do INEP (pré-calibrados).

    Diferente do /api/v3/calibrar-cientifico que calibra localmente, este
    endpoint usa os parâmetros exatos publicados pelo INEP para a prova
    especificada + transformação empírica (σ, μ) derivada dos microdados.

    Precisão empírica alta quando os parâmetros oficiais e a transformação
    sigma/mu daquele ano-área estão carregados. Não afirma equivalência
    bit-a-bit com o software interno do INEP.
    """
    t0 = time.time()
    client = _require_supabase()
    area = payload.area.upper()
    ano = payload.ano

    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Área inválida: {area}")

    # 1. Buscar a transformação (σ, μ)
    transf_r = client.table("enem_transformacoes").select("*").eq("ano", ano).eq("area", area).limit(1).execute()
    if not transf_r.data:
        raise HTTPException(
            status_code=404,
            detail=f"Transformação não disponível para {area}/{ano}. Rode calcular_transformacao.py primeiro.",
        )
    transf_row = transf_r.data[0]
    sigma = float(transf_row["sigma"])
    mu = float(transf_row["mu"])

    # 2. Buscar itens da prova
    itens_area_r = client.table("enem_itens").select("*").eq("ano", ano).eq("area", area).execute()
    itens_by_co = {it["co_item"]: it for it in itens_area_r.data}

    maps_r = client.table("enem_itens_prova").select("*").eq("ano", ano).eq("co_prova", payload.co_prova).execute()
    if not maps_r.data:
        raise HTTPException(
            status_code=404,
            detail=f"Prova {payload.co_prova} não encontrada para {area}/{ano}",
        )

    # Build ordered item list
    all_items: list[dict] = []
    for m in maps_r.data:
        it = itens_by_co.get(m["co_item"])
        if not it or it.get("item_abandonado"):
            continue
        if it.get("param_discriminacao") is None:
            continue
        all_items.append({
            "posicao": m["posicao"],
            "co_item": it["co_item"],
            "a": float(it["param_discriminacao"]),
            "b": float(it["param_dificuldade"]),
            "c": float(it.get("param_acerto_casual") or 0.20),
            "gabarito": it.get("gabarito"),
            "tp_lingua": it.get("tipo_lingua"),
        })

    if not all_items:
        raise HTTPException(status_code=404, detail=f"Nenhum item calibrado na prova {payload.co_prova}")

    # 3. Filter by language for LC
    if area == "LC":
        tp_lingua = payload.tp_lingua
        if tp_lingua not in (0, 1):
            raise HTTPException(
                status_code=400,
                detail="Para área LC é obrigatório fornecer tp_lingua (0=inglês, 1=espanhol)",
            )
        filtered = [
            it for it in all_items
            if it.get("tp_lingua") is None or it.get("tp_lingua") == tp_lingua
        ]
    else:
        filtered = all_items

    filtered.sort(key=lambda x: x["posicao"])
    n_itens = len(filtered)

    if n_itens < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Poucos itens válidos ({n_itens}). Verifique co_prova e tp_lingua.",
        )

    # 4. Process students
    if len(payload.alunos) < 1:
        raise HTTPException(status_code=400, detail="Nenhum aluno fornecido")

    resultados: list[ResultadoAlunoINEP] = []
    notas = []
    for aluno in payload.alunos:
        if len(aluno.respostas) != n_itens:
            raise HTTPException(
                status_code=400,
                detail=f"Aluno {aluno.aluno_id}: {len(aluno.respostas)} respostas, esperado {n_itens}",
            )

        resp = np.array(aluno.respostas, dtype=int)
        acertos = int(resp.sum())

        # EAP theta using INEP's pre-calibrated parameters
        theta, _se = eap_theta_cientifico(resp, filtered)

        # Apply empirical transformation
        nota = sigma * theta + mu
        nota = round(float(nota), 1)
        notas.append(nota)

        resultados.append(ResultadoAlunoINEP(
            aluno_id=aluno.aluno_id,
            acertos=acertos,
            theta=round(float(theta), 4),
            nota_inep=nota,
            estado_tri=estado_tri(nota),
        ))

    # 5. Diagnostic
    diag = {
        "n_alunos": len(resultados),
        "n_itens": n_itens,
        "nota_media": round(float(np.mean(notas)), 1),
        "nota_min": round(float(np.min(notas)), 1),
        "nota_max": round(float(np.max(notas)), 1),
        "nota_desvio": round(float(np.std(notas)), 1),
        "metodo": "INEP (parâmetros oficiais pré-calibrados)",
    }

    transformacao_model = TransformacaoENEM(
        ano=ano, area=area,
        sigma=sigma, mu=mu,
        n_alunos=transf_row["n_alunos"] or 0,
        correlacao=float(transf_row["correlacao"] or 0),
        mae=float(transf_row["mae"] or 0),
    )

    return ResponseCalcularComINEP(
        status="sucesso",
        ano=ano,
        area=area,
        co_prova=payload.co_prova,
        n_alunos=len(resultados),
        n_itens=n_itens,
        transformacao_usada=transformacao_model,
        resultados=resultados,
        diagnostico=diag,
        tempo_processamento=round(time.time() - t0, 3),
    )


_AREA_OFFSETS = {"LC": 1, "CH": 46, "CN": 91, "MT": 136}


def _load_inep_items_for_prova(
    client: Any,
    *,
    ano: int,
    area: str,
    co_prova: int,
    tp_lingua: Optional[int],
) -> tuple[dict, list[dict]]:
    """Fetch INEP transformação + calibrated items for one (ano, area, co_prova).

    Returns (transf_row, inep_items sorted by posicao). Items with missing
    parameters or abandonado=True are dropped. For LC, filters by tp_lingua.
    Raises HTTPException(404) when transformação or prova is missing, or (400)
    when fewer than 10 valid items remain.
    """
    transf_r = (
        client.table("enem_transformacoes")
        .select("*")
        .eq("ano", ano)
        .eq("area", area)
        .limit(1)
        .execute()
    )
    if not transf_r.data:
        # Surface the list of supported years so the caller (or the UI) knows
        # what to pick. Calibrating σ, μ requires INEP microdata for that year
        # which is ~1.6 GB per year and cannot be derived from the reference
        # table alone — must be populated offline via
        # app/scripts/calcular_transformacao.py.
        try:
            anos_r = (
                client.table("enem_transformacoes")
                .select("ano")
                .eq("area", area)
                .execute()
            )
            anos_ok = sorted({row["ano"] for row in (anos_r.data or [])})
        except Exception:
            anos_ok = []
        detalhe = (
            f"V4 INEP não disponível para {area}/{ano}. "
            f"Anos suportados para {area}: {anos_ok or '(nenhum)'}. "
            "Use modo V3 Local para esse ano, ou rode "
            "`python -m app.scripts.calcular_transformacao --ano {ano} --area {area}` "
            "com o MICRODADOS_ENEM_{ano}.csv para habilitar V4."
        ).format(ano=ano, area=area)
        raise HTTPException(status_code=404, detail=detalhe)
    transf_row = transf_r.data[0]

    itens_area_r = (
        client.table("enem_itens")
        .select("*")
        .eq("ano", ano)
        .eq("area", area)
        .execute()
    )
    itens_by_co = {it["co_item"]: it for it in itens_area_r.data}

    maps_r = (
        client.table("enem_itens_prova")
        .select("*")
        .eq("ano", ano)
        .eq("co_prova", co_prova)
        .execute()
    )
    if not maps_r.data:
        raise HTTPException(
            status_code=404,
            detail=f"Prova {co_prova} não encontrada para {area}/{ano}",
        )

    all_items: list[dict] = []
    for m in maps_r.data:
        it = itens_by_co.get(m["co_item"])
        if not it or it.get("item_abandonado") or it.get("param_discriminacao") is None:
            continue
        all_items.append({
            "posicao": m["posicao"],
            "co_item": it["co_item"],
            "a": float(it["param_discriminacao"]),
            "b": float(it["param_dificuldade"]),
            "c": float(it.get("param_acerto_casual") or 0.20),
            "gabarito": it.get("gabarito"),
            "tp_lingua": it.get("tipo_lingua"),
        })

    if area == "LC":
        inep_items = [
            it for it in all_items
            if it.get("tp_lingua") is None or it.get("tp_lingua") == tp_lingua
        ]
    else:
        inep_items = all_items

    inep_items.sort(key=lambda x: x["posicao"])

    if len(inep_items) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Poucos itens INEP válidos: {len(inep_items)}",
        )

    return transf_row, inep_items


def _score_alunos_with_inep_items(
    alunos_parsed: list[dict],
    *,
    inep_items: list[dict],
    offset_area: int,
    sigma: float,
    mu: float,
) -> Dict[str, Dict[str, Any]]:
    """Score each aluno with INEP official parameters using position-aligned responses.

    Each aluno's response vector is aligned to INEP items by position
    (offset_area maps the first file column to a 1-indexed ENEM position).
    Alunos with fewer than 10 aligned items are skipped. Besides θ and nota,
    the result includes ``acert_dif`` — the fraction of INEP-classified HARD
    items (b > 1.5) the student answered correctly. This is the same Primi
    metric the V3 path exposes, but computed against INEP's ground-truth
    difficulty estimates instead of the local calibration.
    """
    v4_results: Dict[str, Dict[str, Any]] = {}
    for aluno in alunos_parsed:
        resp_vec = aluno["respostas"]
        aligned_binary: list[int] = []
        aligned_items: list[dict] = []
        for it in inep_items:
            str_idx = it["posicao"] - offset_area
            if 0 <= str_idx < len(resp_vec):
                aligned_binary.append(resp_vec[str_idx])
                aligned_items.append(it)
        if len(aligned_binary) < 10:
            continue
        resp = np.array(aligned_binary, dtype=int)
        acertos = int(resp.sum())
        theta, _se = eap_theta_cientifico(resp, aligned_items)
        nota = round(float(sigma * theta + mu), 1)

        # Primi acert_dif: fraction of HARD items (INEP b > 1.5) the student
        # got right. Matches the thresholds used in motores/cientifico.py so
        # V3 and V4 dots are coloured on the same scale.
        hard_idx = [i for i, it in enumerate(aligned_items) if float(it["b"]) > 1.5]
        if hard_idx:
            acertos_dificeis = int(resp[hard_idx].sum())
            acert_dif = round(acertos_dificeis / len(hard_idx), 4)
        else:
            acert_dif = 0.0

        v4_results[aluno["aluno_id"]] = {
            "theta": round(float(theta), 4),
            "nota": nota,
            "estado": estado_tri(nota),
            "acertos": acertos,
            "acert_dif": acert_dif,
        }
    return v4_results


def _build_validation_comparison(
    alunos_parsed: list[dict],
    *,
    v3_by_id: Dict[str, Dict[str, Any]],
    v4_results: Dict[str, Dict[str, Any]],
) -> tuple[list[ResultadoComparacao], EstatisticasComparacao]:
    """Pair per-aluno V3 and V4 scores and compute aggregate agreement stats."""
    comparacoes: list[ResultadoComparacao] = []
    diffs: list[float] = []
    notas_v3_list: list[float] = []
    notas_v4_list: list[float] = []
    concordancia = 0

    for aluno in alunos_parsed:
        aid = aluno["aluno_id"]
        v3 = v3_by_id.get(aid)
        v4 = v4_results.get(aid)
        if not v3 or not v4:
            continue
        nota_v3 = v3["nota_ancorada"]
        nota_v4 = v4["nota"]
        diff = nota_v3 - nota_v4
        diffs.append(diff)
        notas_v3_list.append(nota_v3)
        notas_v4_list.append(nota_v4)
        if v3["estado_tri"] == v4["estado"]:
            concordancia += 1

        comparacoes.append(ResultadoComparacao(
            aluno_id=aid,
            acertos=v4["acertos"],
            theta_v3=round(float(v3["theta"]), 4),
            nota_v3=round(float(nota_v3), 1),
            estado_v3=v3["estado_tri"],
            theta_v4=v4["theta"],
            nota_v4=nota_v4,
            estado_v4=v4["estado"],
            acert_dif_v4=float(v4.get("acert_dif", 0.0)),
            diff=round(float(diff), 2),
        ))

    diffs_arr = np.array(diffs)
    notas_v3_arr = np.array(notas_v3_list)
    notas_v4_arr = np.array(notas_v4_list)

    if len(notas_v3_arr) >= 2 and notas_v3_arr.std() > 0 and notas_v4_arr.std() > 0:
        correlacao = float(np.corrcoef(notas_v3_arr, notas_v4_arr)[0, 1])
    else:
        correlacao = 0.0

    n = len(comparacoes)
    estatisticas = EstatisticasComparacao(
        n_alunos=n,
        mae=round(float(np.mean(np.abs(diffs_arr))), 2) if n else 0.0,
        rmse=round(float(np.sqrt(np.mean(diffs_arr ** 2))), 2) if n else 0.0,
        correlacao=round(correlacao, 6),
        mean_diff=round(float(np.mean(diffs_arr)), 2) if n else 0.0,
        max_diff_abs=round(float(np.max(np.abs(diffs_arr))), 2) if n else 0.0,
        pct_concordancia_nivel=round(concordancia / n * 100, 2) if n else 0.0,
        nota_media_v3=round(float(np.mean(notas_v3_arr)), 1) if n else 0.0,
        nota_media_v4=round(float(np.mean(notas_v4_arr)), 1) if n else 0.0,
    )
    return comparacoes, estatisticas


@router.post("/api/v4/enem/validar-simulado", response_model=ResponseValidarSimulado)
async def validar_simulado(
    area: str = Form(...),
    ano: int = Form(...),
    co_prova: int = Form(...),
    arquivo: UploadFile = File(...),
    tp_lingua: Optional[int] = Form(None),
):
    """
    Valida um simulado executando dois cálculos paralelos nos MESMOS alunos:

    1. V3 — Calibração LOCAL do XTRI Engine (calibra itens do próprio simulado)
    2. V4 — Parâmetros OFICIAIS INEP (do Supabase) + transformação empírica

    Retorna comparação lado-a-lado para medir o quão próximo nossa calibração
    local está do ground truth INEP.
    """
    t0 = time.time()
    area = area.upper()
    avisos: list[str] = []

    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Área inválida: {area}")

    if area == "LC" and tp_lingua not in (0, 1):
        raise HTTPException(
            status_code=400,
            detail="Para área LC é obrigatório fornecer tp_lingua (0=inglês, 1=espanhol)",
        )

    client = _require_supabase()

    # Parse upload
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erro no parsing: {e}")

    avisos.extend(parsed["avisos"])
    alunos_parsed = parsed["alunos"]
    if len(alunos_parsed) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Mínimo 10 alunos. Recebido: {len(alunos_parsed)}",
        )
    if parsed["n_itens"] < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo tem apenas {parsed['n_itens']} itens por aluno, mínimo 30.",
        )

    transf_row, inep_items = _load_inep_items_for_prova(
        client, ano=ano, area=area, co_prova=co_prova, tp_lingua=tp_lingua,
    )
    sigma = float(transf_row["sigma"])
    mu = float(transf_row["mu"])
    n_itens = len(inep_items)

    # V3 — local calibration
    tabelas_multiano = state.get_tabelas_multiano()
    tabela = tabelas_multiano.get(ano) or tabelas_multiano.get(2022)
    if tabela is None:
        raise HTTPException(status_code=500, detail="Tabela de referência multi-ano não carregada")

    alunos_motor = [
        {"aluno_id": a["aluno_id"], "nome": "", "respostas": a["respostas"]}
        for a in alunos_parsed
    ]
    try:
        resultado_v3 = pipeline_cientifico(alunos=alunos_motor, area=area, tabela=tabela)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na calibração V3: {e}")
    v3_by_id = {r["aluno_id"]: r for r in resultado_v3["resultados_alunos"]}

    # V4 — INEP official parameters
    v4_results = _score_alunos_with_inep_items(
        alunos_parsed,
        inep_items=inep_items,
        offset_area=_AREA_OFFSETS[area],
        sigma=sigma,
        mu=mu,
    )

    comparacoes, estatisticas = _build_validation_comparison(
        alunos_parsed, v3_by_id=v3_by_id, v4_results=v4_results,
    )

    transformacao_model = TransformacaoENEM(
        ano=ano, area=area,
        sigma=sigma, mu=mu,
        n_alunos=transf_row["n_alunos"] or 0,
        correlacao=float(transf_row["correlacao"] or 0),
        mae=float(transf_row["mae"] or 0),
    )

    return ResponseValidarSimulado(
        status="sucesso",
        ano=ano, area=area, co_prova=co_prova,
        n_alunos=len(comparacoes),
        n_itens=n_itens,
        transformacao_inep=transformacao_model,
        estatisticas=estatisticas,
        resultados=comparacoes,
        avisos=avisos,
        tempo_processamento=round(time.time() - t0, 3),
    )
