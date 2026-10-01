"""Endpoints V3 — Motor Científico (MML + EAP + Ancoragem) e workflow de CSV."""

import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import state
from ..csv_parser import parse_single_area
from ..db.supabase_client import get_supabase, is_supabase_configured
from ..enriquecimento import (
    build_coorte_stats,
    build_metodologia_banner,
    build_resultado_estruturado,
)
from ..models import (
    DiagnosticoArea,
    ItemParametros,
    PontoReferencia,
    RequestCalibrarCientifico,
    RequestEstimarTheta,
    ResponseCalibrarCientifico,
    ResponseEstimarTheta,
    RequestSimuladoBinarizado,
    ResponseProcessarCSV,
    ResponseSimuladoBinarizado,
    ResultadoAlunoCientifico,
    ResultadoAlunoFrontend,
)
from ..motores.cientifico import estimar_theta_individual, pipeline_cientifico
from ..simulado_binarizado import SimuladoInvalidoError, processar_simulado_binarizado
from ..runtime.unidim import (
    RuntimeArtifactAmbiguousError,
    RuntimeArtifactNotFoundError,
    resolve_unidim_artifact,
)
from ..storage.artifacts import save_unidim_artifact
from ..storage.model_registry import set_active_model
from ..tabela_referencia import TabelaReferenciaTRI
from .common import _parse_itens_anulados, _remover_itens_anulados

logger = logging.getLogger(__name__)

router = APIRouter()


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT — MOTOR CIENTÍFICO (MML + EAP)
# ════════════════════════════════════════════════════════════════════════════════

@router.post("/api/v3/calibrar-cientifico", response_model=ResponseCalibrarCientifico)
async def calibrar_cientifico(payload: RequestCalibrarCientifico):
    """
    Calibra itens e estima notas TRI usando modelo 2PL real.

    Ideal para:
    - Fechamento de simulados oficiais
    - Geração de relatórios pedagógicos detalhados
    - Alimentação do banco de itens com parâmetros a, b, c

    Regra inviolável: cada combinação simulado × área é um modelo separado.
    """
    tabela = state.get_tabela()
    if tabela is None:
        raise HTTPException(
            status_code=500,
            detail="Tabela de referência TRI não carregada"
        )

    area = payload.area.upper()
    if area not in ['LC', 'CH', 'CN', 'MT']:
        raise HTTPException(
            status_code=400,
            detail=f"Área inválida: {area}. Esperado: LC, CH, CN ou MT"
        )

    if len(payload.alunos) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Mínimo de 10 alunos para calibração científica. Recebido: {len(payload.alunos)}"
        )

    # Verificar consistência do número de itens
    n_itens_esperado = len(payload.alunos[0].respostas)
    for a in payload.alunos:
        if len(a.respostas) != n_itens_esperado:
            raise HTTPException(
                status_code=400,
                detail=f"Aluno {a.aluno_id} tem {len(a.respostas)} respostas, esperado {n_itens_esperado}"
            )

    t0 = time.time()
    logger.info(
        "calibrar-cientifico: simulado=%s area=%s N=%d itens=%d c=%s",
        payload.simulado_id,
        area,
        len(payload.alunos),
        n_itens_esperado,
        payload.c_fixo,
    )

    try:
        # Converter para formato esperado pelo motor
        alunos_motor = [
            {
                'aluno_id': a.aluno_id,
                'nome': a.nome or '',
                'respostas': a.respostas,
            }
            for a in payload.alunos
        ]

        # Executar pipeline
        resultado = pipeline_cientifico(
            alunos=alunos_motor,
            area=area,
            tabela=tabela,
            c_fixo=payload.c_fixo,
        )

        elapsed = time.time() - t0
        diag = resultado['diagnostico']
        logger.info(
            "calibrar-cientifico done in %.2fs — b_medio=%s saturados=%s nota_media=%s pct_dentro=%.1f",
            elapsed,
            diag['b_medio'],
            diag['itens_saturados'],
            diag['nota_media'],
            diag['pct_dentro_intervalo'],
        )

        # Montar resposta tipada
        parametros = [
            ItemParametros(
                item_index=i,
                a=p['a'],
                b=p['b'],
                c=p['c'],
                taxa_acerto=p['taxa_acerto'],
                saturado=p['saturado'],
            )
            for i, p in enumerate(resultado['parametros_itens'])
        ]

        resultados_alunos = [
            ResultadoAlunoCientifico(
                aluno_id=r['aluno_id'],
                nome=r.get('nome', ''),
                acertos=r['acertos'],
                theta=r['theta'],
                se=r['se'],
                nota_ancorada=r['nota_ancorada'],
                estado_tri=r['estado_tri'],
            )
            for r in resultado['resultados_alunos']
        ]

        diagnostico = dict(resultado["diagnostico"])
        try:
            manifesto = save_unidim_artifact(
                simulado_id=payload.simulado_id,
                area=area,
                ano_referencia=payload.ano_referencia,
                c_fixo=payload.c_fixo,
                parametros_itens=resultado["parametros_itens"],
                transformacao=resultado["transformacao"],
                diagnostico=diagnostico,
                n_alunos=len(payload.alunos),
                n_itens=n_itens_esperado,
            )
            ativo = set_active_model(manifesto)
            diagnostico["artefato_persistido"] = True
            diagnostico["artefato_id"] = manifesto["artifact_id"]
            diagnostico["artefato_dir"] = manifesto["artifact_dir"]
            diagnostico["registry_modelo"] = ativo["modelo"]
        except Exception as storage_exc:
            diagnostico["artefato_persistido"] = False
            diagnostico["artefato_erro"] = str(storage_exc)

        return ResponseCalibrarCientifico(
            status="sucesso",
            simulado_id=payload.simulado_id,
            area=area,
            n_alunos=len(payload.alunos),
            n_itens=n_itens_esperado,
            parametros_itens=parametros,
            resultados_alunos=resultados_alunos,
            transformacao=resultado.get("transformacao"),
            artefato_id=diagnostico.get("artefato_id"),
            diagnostico=diagnostico,
        )

    except Exception as e:
        logger.exception("calibrar-cientifico: pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT — ESTIMAÇÃO RÁPIDA COM PARÂMETROS PRÉ-CALIBRADOS
# ════════════════════════════════════════════════════════════════════════════════

@router.post("/api/v3/estimar-theta", response_model=ResponseEstimarTheta)
async def estimar_theta(payload: RequestEstimarTheta):
    """
    Estima nota de UM aluno usando parâmetros de itens já calibrados.

    Ideal para:
    - Alunos retardatários (prova já calibrada)
    - Processamento assíncrono de cartões-resposta via OMR
    - Recálculo rápido sem recalibrar toda a turma
    """
    tabela = state.get_tabela()
    if tabela is None:
        raise HTTPException(
            status_code=500,
            detail="Tabela de referência TRI não carregada"
        )

    area = payload.area.upper()
    if area not in ['LC', 'CH', 'CN', 'MT']:
        raise HTTPException(
            status_code=400,
            detail=f"Área inválida: {area}"
        )

    params = None
    artefato_id = None
    transformacao = payload.transformacao

    if payload.parametros_itens:
        params = [
            {'a': p.a, 'b': p.b, 'c': p.c}
            for p in payload.parametros_itens
        ]
    else:
        try:
            artefato = resolve_unidim_artifact(
                ano_referencia=payload.ano_referencia,
                area=area,
                artifact_id=payload.artefato_id,
                scope_id=payload.simulado_id,
            )
        except RuntimeArtifactNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Envie `parametros_itens` ou informe `simulado_id`/`artefato_id` "
                    f"de um modelo local salvo. {exc}"
                ),
            )
        except RuntimeArtifactAmbiguousError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        params = [
            {'a': p['a'], 'b': p['b'], 'c': p.get('c', 0.20)}
            for p in artefato["parametros_itens"]
        ]
        if transformacao is None:
            transformacao = artefato.get("transformacao")
        artefato_id = artefato["manifest"]["artifact_id"]

    if len(payload.respostas) != len(params):
        raise HTTPException(
            status_code=400,
            detail=f"Número de respostas ({len(payload.respostas)}) != número de itens ({len(params)})"
        )

    if transformacao is None and is_supabase_configured():
        try:
            sb = get_supabase()
            tr = (
                sb.table("enem_transformacoes")
                .select("sigma,mu")
                .eq("ano", payload.ano_referencia)
                .eq("area", area)
                .limit(1)
                .execute()
            )
            if tr.data:
                transformacao = {
                    "sigma": float(tr.data[0]["sigma"]),
                    "mu": float(tr.data[0]["mu"]),
                }
        except Exception:
            transformacao = None

    resultado = estimar_theta_individual(
        respostas=payload.respostas,
        parametros_itens=params,
        area=area,
        tabela=tabela,
        transformacao=transformacao,
    )

    return ResponseEstimarTheta(**resultado, artefato_id=artefato_id)


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT — TABELA DE REFERÊNCIA (Útil para o frontend)
# ════════════════════════════════════════════════════════════════════════════════

@router.get("/api/v3/referencias/{area}")
async def obter_referencia(area: str):
    """Retorna a tabela de referência oficial para uma área."""
    tabela = state.get_tabela()
    if tabela is None:
        raise HTTPException(status_code=500, detail="Tabela não carregada")

    area = area.upper()
    if area not in tabela.lookup:
        raise HTTPException(
            status_code=404,
            detail=f"Área {area} não encontrada. Disponíveis: {tabela.areas}"
        )

    dados = []
    for acertos in sorted(tabela.lookup[area].keys()):
        ref = tabela.lookup[area][acertos]
        dados.append({
            'acertos': acertos,
            'tri_min': ref['tri_min'],
            'tri_med': ref['tri_med'],
            'tri_max': ref['tri_max'],
        })

    return {
        'area': area,
        'total_registros': len(dados),
        'dados': dados,
    }


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — FRONTEND (CSV Upload Workflow)
# ════════════════════════════════════════════════════════════════════════════════

@router.get("/api/v3/anos-referencia")
async def anos_referencia():
    """Returns available reference years for the year selector dropdown."""
    return sorted(state.get_tabelas_multiano().keys())


def _build_reference_scale_points(
    tab: TabelaReferenciaTRI, area: str,
) -> list[PontoReferencia]:
    """Flatten the (area, acertos) → {tri_min, tri_med, tri_max} table into
    a list ordered by acertos, suitable for the frontend's anchor graph."""
    pontos: list[PontoReferencia] = []
    for acertos in sorted(tab.lookup.get(area, {}).keys()):
        ref = tab.lookup[area][acertos]
        pontos.append(PontoReferencia(
            acertos=acertos,
            tri_min=ref["tri_min"],
            tri_med=ref["tri_med"],
            tri_max=ref["tri_max"],
        ))
    return pontos


def _fetch_inep_scale_or_empirical_default(
    ano_referencia: int, area: str,
) -> tuple[float, float]:
    """Return (sigma, mu) for the INEP empirical transformação, or the empirical
    defaults (100, 500) when Supabase is unavailable or missing that row."""
    sigma_oficial, mu_oficial = 100.0, 500.0
    if not is_supabase_configured():
        return sigma_oficial, mu_oficial
    try:
        sb = get_supabase()
        tr = (
            sb.table("enem_transformacoes")
            .select("*")
            .eq("ano", ano_referencia)
            .eq("area", area)
            .limit(1)
            .execute()
        )
        if tr.data:
            sigma_oficial = float(tr.data[0]["sigma"])
            mu_oficial = float(tr.data[0]["mu"])
    except Exception:
        pass
    return sigma_oficial, mu_oficial


def _build_enrichment_payload(
    raw_results: list[dict],
    *,
    diag: dict,
    tab: TabelaReferenciaTRI,
    area: str,
    ano_referencia: int,
) -> tuple[list[dict], Any, Any]:
    """Assemble the V2.1 per-student enrichment + methodology + cohort stats.

    Returns (resultados_estruturados, metodologia_banner, coorte_stats). Pure
    aggregation; all heavy IRT math is already done in ``raw_results``.
    """
    all_thetas_list = [float(r["theta"]) for r in raw_results]
    all_notas_list = [float(r["nota_ancorada"]) for r in raw_results]
    all_acertos_list = [int(r["acertos"]) for r in raw_results]

    cohort_mean = float(np.mean(all_thetas_list)) if all_thetas_list else 0.0
    cohort_std_val = float(np.std(all_thetas_list)) if all_thetas_list else 1.0

    sorted_by_nota = sorted(raw_results, key=lambda r: -r["nota_ancorada"])
    rank_by_id = {r["aluno_id"]: i + 1 for i, r in enumerate(sorted_by_nota)}

    sigma_oficial, mu_oficial = _fetch_inep_scale_or_empirical_default(
        ano_referencia, area,
    )

    n_dificeis_total = diag.get("itens_dificeis", 0)

    resultados_estruturados: list[dict] = []
    for r in raw_results:
        ref = tab.obter(area, r["acertos"]) if r["acertos"] in tab.lookup.get(area, {}) else {
            "tri_min": 0, "tri_med": 0, "tri_max": 1000,
        }
        estr = build_resultado_estruturado(
            aluno_id=r["aluno_id"],
            acertos=r["acertos"],
            total_itens=diag["n_itens"],
            theta=r["theta"],
            se=r.get("se", 0.0),
            nota=r["nota_ancorada"],
            cohort_mean=cohort_mean,
            cohort_std=cohort_std_val,
            rank_in_cohort=rank_by_id[r["aluno_id"]],
            total_in_cohort=len(raw_results),
            area=area,
            ano=ano_referencia,
            sigma_oficial=sigma_oficial,
            mu_oficial=mu_oficial,
            ref_min=ref["tri_min"],
            ref_med=ref["tri_med"],
            ref_max=ref["tri_max"],
            acert_dif=r.get("acert_dif", 0.0),
            n_itens_faceis_acertados=r.get("acertos_faceis", 0),
            n_itens_medios_acertados=r.get("acertos_medios", 0),
            n_itens_dificeis_acertados=r.get("acertos_dificeis", 0),
            n_itens_dificeis_total=n_dificeis_total,
        )
        resultados_estruturados.append(estr)

    metodologia = build_metodologia_banner(
        area=area,
        ano=ano_referencia,
        n_alunos=len(raw_results),
        cohort_mean=cohort_mean,
        cohort_std=cohort_std_val,
        acertos_medio=float(np.mean(all_acertos_list)) if all_acertos_list else 0.0,
        n_itens=diag["n_itens"],
        nota_media=diag["nota_media"],
    )

    coorte_stats = build_coorte_stats(
        thetas=all_thetas_list,
        notas=all_notas_list,
        acertos=all_acertos_list,
        n_itens=diag["n_itens"],
        area=area,
    )

    return resultados_estruturados, metodologia, coorte_stats


@router.post("/api/v3/processar-csv", response_model=ResponseProcessarCSV)
async def processar_csv(
    area: str = Form(...),
    ano_referencia: int = Form(...),
    arquivo: UploadFile = File(...),
    c_fixo: float = Form(0.20),
    itens_anulados: str = Form(""),
):
    """
    Process ONE uploaded file (CSV or XLSX) for ONE ENEM area.

    Each area has its own reference scale and is calibrated independently.
    Upload one file per area — never combine areas in a single request.
    """
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(
            status_code=400,
            detail=f"Area invalida: {area}. Esperado: LC, CH, CN ou MT"
        )

    tabelas_multiano = state.get_tabelas_multiano()
    if ano_referencia not in tabelas_multiano:
        raise HTTPException(
            status_code=400,
            detail=f"Ano {ano_referencia} nao disponivel. Opcoes: {sorted(tabelas_multiano.keys())}"
        )

    tab = tabelas_multiano[ano_referencia]
    t0 = time.time()

    # Read and parse file
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
        raise HTTPException(status_code=400, detail=str(e))

    avisos: list[str] = list(parsed["avisos"])
    anulados = _parse_itens_anulados(itens_anulados)
    if anulados:
        removidos = _remover_itens_anulados(parsed, anulados)
        if not removidos:
            avisos.append(
                "Itens anulados informados, mas nenhum deles foi encontrado "
                f"no arquivo: {sorted(anulados)}"
            )
        else:
            avisos.append(
                "Itens anulados removidos da calibração e do denominador: "
                + ", ".join(removidos)
            )

    if parsed["n_alunos"] < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Minimo de 10 alunos para calibracao. Recebido: {parsed['n_alunos']}"
        )

    # Calibrate
    try:
        alunos_motor = [
            {"aluno_id": a["aluno_id"], "nome": "", "respostas": a["respostas"]}
            for a in parsed["alunos"]
        ]

        resultado = pipeline_cientifico(
            alunos=alunos_motor,
            area=area,
            tabela=tab,
            c_fixo=c_fixo,
        )
    except Exception as e:
        logger.exception("processar-csv calibration failed for area=%s", area)
        raise HTTPException(status_code=500, detail=f"Erro na calibracao: {e}")

    diag = resultado["diagnostico"]
    logger.info(
        "processar-csv %s/%d: n_alunos=%d nota_media=%.1f pct_dentro=%.1f",
        area,
        ano_referencia,
        diag["n_alunos"],
        diag["nota_media"],
        diag["pct_dentro_intervalo"],
    )

    diagnosticos = [DiagnosticoArea(
        area=area,
        n_itens=diag["n_itens"],
        n_alunos=diag["n_alunos"],
        b_medio=diag["b_medio"],
        b_desvio=diag.get("b_desvio", 0.0),
        nota_media=diag["nota_media"],
        nota_min=diag["nota_min"],
        nota_max=diag["nota_max"],
        pct_dentro_intervalo=diag["pct_dentro_intervalo"],
        itens_faceis=diag.get("itens_faceis", 0),
        itens_medios=diag.get("itens_medios", 0),
        itens_dificeis=diag.get("itens_dificeis", 0),
        acert_dif_medio=diag.get("acert_dif_medio", 0.0),
    )]

    # Build per-student results
    area_lower = area.lower()
    resultados: list[ResultadoAlunoFrontend] = []
    for r in resultado["resultados_alunos"]:
        row = ResultadoAlunoFrontend(aluno_id=r["aluno_id"])
        setattr(row, f"acertos_{area_lower}", r["acertos"])
        setattr(row, f"theta_{area_lower}", round(r["theta"], 3))
        setattr(row, f"nota_{area_lower}", r["nota_ancorada"])
        setattr(row, f"estado_{area_lower}", r["estado_tri"])
        setattr(row, f"acert_dif_{area_lower}", r.get("acert_dif", 0.0))
        resultados.append(row)

    referencia = _build_reference_scale_points(tab, area)

    resultados_estruturados, metodologia, coorte_stats = _build_enrichment_payload(
        resultado["resultados_alunos"],
        diag=diag,
        tab=tab,
        area=area,
        ano_referencia=ano_referencia,
    )

    elapsed = time.time() - t0
    logger.info(
        "processar-csv done in %.2fs — V2.1 enrichment for %d students",
        elapsed,
        len(resultados_estruturados),
    )

    return ResponseProcessarCSV(
        status="sucesso",
        ano_referencia=ano_referencia,
        n_alunos=len(resultados),
        areas_processadas=[area],
        resultados=resultados,
        diagnosticos=diagnosticos,
        referencia=referencia,
        avisos=avisos,
        tempo_processamento=round(elapsed, 2),
        resultados_estruturados=resultados_estruturados,
        metodologia=metodologia,
        coorte_stats=coorte_stats,
    )


@router.post("/api/v3/processar-json", response_model=ResponseSimuladoBinarizado)
def processar_json(payload: RequestSimuladoBinarizado):
    """
    Corrige um simulado ENEM inteiro enviado como JSON binarizado.

    Cada área é calibrada separadamente (3PL; c fixo em 0.20, estimado por
    item com ≥ 500 alunos na área) e ancorada na escala do ENEM do
    ``ano_referencia`` pela posição condicional aos acertos. Função síncrona
    de propósito: o FastAPI a executa no threadpool, sem travar o event loop.
    """
    tabelas_multiano = state.get_tabelas_multiano()
    if payload.ano_referencia not in tabelas_multiano:
        raise HTTPException(
            status_code=400,
            detail=f"Ano {payload.ano_referencia} nao disponivel. Opcoes: {sorted(tabelas_multiano.keys())}",
        )

    t0 = time.time()
    try:
        resultado = processar_simulado_binarizado(
            payload.model_dump(),
            tabelas_multiano[payload.ano_referencia],
            payload.ano_referencia,
        )
    except SimuladoInvalidoError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("processar-json calibration failed")
        raise HTTPException(status_code=500, detail=f"Erro na calibracao: {e}")

    if not resultado["areas"]:
        raise HTTPException(status_code=400, detail="; ".join(resultado["avisos"]) or "Nenhuma área calibrada.")

    tempo = round(time.time() - t0, 2)
    logger.info(
        "processar-json ano=%d alunos=%d areas=%s em %.2fs",
        payload.ano_referencia, resultado["total_alunos"], list(resultado["areas"]), tempo,
    )
    return ResponseSimuladoBinarizado(**resultado, tempo_processamento=tempo)
