"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    XTRI ENGINE API — TRI V3 (FastAPI)                        ║
║                                                                              ║
║  API híbrida com dois motores de estimação TRI:                              ║
║  • Motor Heurístico: Rápido, compatível com V2 (coerência pedagógica)       ║
║  • Motor Científico: MML + EAP + Ancoragem (calibração real)                ║
║                                                                              ║
║  Porta: 5003 (retrocompatível com V2)                                        ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import csv
import io
import logging
import os
import sys
import tempfile
import time
import traceback
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Load .env if available (from project root or current dir)
try:
    from dotenv import load_dotenv
    for candidate in (
        Path(__file__).resolve().parents[2] / ".env",  # /Volumes/KINGSTON/apps/apps/MIRT/.env
        Path(__file__).resolve().parents[1] / ".env",  # python_tri_v3_service/.env
        Path(".env"),
    ):
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:
    pass

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    RequestCalcularTRI,
    ResponseCalcularTRI,
    RequestCalibrarCientifico,
    ResponseCalibrarCientifico,
    ItemParametros,
    ResultadoAlunoCientifico,
    RequestEstimarTheta,
    ResponseEstimarTheta,
    RequestEstimarThetaMIRT,
    ResponseEstimarThetaMIRT,
    DiagnosticoArea,
    PontoReferencia,
    ResultadoAlunoFrontend,
    ResponseProcessarCSV,
    TransformacaoENEM,
    ItemINEP,
    ProvaINEP,
    ENEMInfo,
    ResultadoAlunoINEP,
    RequestCalcularComINEP,
    ResponseCalcularComINEP,
    ResultadoComparacao,
    EstatisticasComparacao,
    ResponseValidarSimulado,
    # V5 — Microdata Analysis (Phase 8)
    MicrodataAvailabilityResponse,
    MicrodataYearStatus,
    MicrodataPrepareRequest,
    MicrodataPrepareResponse,
    MicrodataJobStatusResponse,
    MicrodataAreaSummary,
    MicrodataDifficultyRow,
    MicrodataAreaDashboardResponse,
    MicrodataFullReportResponse,
    RuntimeDirectoryStatus,
    RuntimeModelStatus,
    RuntimeJobStatus,
    RuntimeDesktopStatusResponse,
)
from .tabela_referencia import TabelaReferenciaTRI
from .csv_parser import parse_single_area
from .motores.heuristico import processar_turma
from .motores.cientifico import (
    pipeline_cientifico,
    estimar_theta_individual,
    eap_theta as eap_theta_cientifico,
    estado_tri,
)
from .irt.transform import ancorar_nota_individual
from .db.supabase_client import get_supabase, is_supabase_configured
from .enriquecimento import (
    build_resultado_estruturado,
    build_metodologia_banner,
    build_coorte_stats,
    NACIONAL_STATS,
)
from .storage.artifacts import save_unidim_artifact
from .storage.model_registry import (
    get_active_model,
    list_active_models,
    list_active_models_by_family,
    set_active_model,
)
from .runtime.local_paths import describe_runtime_layout, ensure_app_support_layout
from .runtime.unidim import (
    RuntimeArtifactAmbiguousError,
    RuntimeArtifactNotFoundError,
    resolve_unidim_artifact,
)
from .runtime.mirt import (
    RuntimeMIRTArtifactAmbiguousError,
    RuntimeMIRTArtifactError,
    RuntimeMIRTArtifactNotFoundError,
    resolve_mirt_artifact,
)
from .mirt.runner import (
    MIRTWorkerExecutionError,
    mirt_runtime_available,
    run_mirt_score,
)


# ════════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ════════════════════════════════════════════════════════════════════════════════

tabela: TabelaReferenciaTRI = None  # type: ignore
tabelas_multiano: Dict[int, TabelaReferenciaTRI] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega a tabela de referência na inicialização."""
    global tabela, tabelas_multiano
    ensure_app_support_layout()
    print("=" * 80)
    print("  XTRI ENGINE API — TRI V3 (FastAPI)")
    print("=" * 80)

    try:
        restored_jobs = job_manager.reload_from_disk()
        print(f"  Jobs restaurados do disco: {restored_jobs}")
    except Exception as e:
        print(f"  AVISO: jobs locais não restaurados: {e}")

    try:
        tabela = TabelaReferenciaTRI()
        tabela.validar()
        info = tabela.info()
        print(f"  Tabela de referência carregada: {info['areas']}")
        print(f"  Acertos por área: {info['acertos_por_area']}")
        print(f"  Limites oficiais: {info['limites_oficiais']}")
    except Exception as e:
        print(f"  ERRO ao carregar tabela: {e}")
        tabela = None

    try:
        tabelas_multiano = TabelaReferenciaTRI.carregar_multiano()
        anos = sorted(tabelas_multiano.keys())
        print(f"  Tabelas multi-ano carregadas: {anos}")
    except Exception as e:
        print(f"  AVISO: tabelas multi-ano não carregadas: {e}")
        tabelas_multiano = {}

    print(f"  Python: {sys.version}")
    print(f"  Servidor: http://0.0.0.0:5003")
    print("=" * 80)

    yield  # A aplicação roda aqui

    print("Encerrando XTRI Engine API...")


# ════════════════════════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

app = FastAPI(
    title="XTRI Engine API",
    description="API de estimação TRI para simulados ENEM — Motor Heurístico + Científico",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS aberto (mesmo comportamento do Flask V2)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware para converter tipos numpy no JSON de resposta
def convert_numpy(obj: Any) -> Any:
    """Converte recursivamente tipos numpy para Python nativos."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — SAÚDE E DEBUG
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Endpoint de saúde (compatível com V2 e Docker healthcheck)."""
    return {
        "status": "online",
        "service": "xtri_engine_api",
        "version": "3.0.0",
        "motores": ["heuristico_v2", "cientifico_mirt"],
        "tabela_carregada": tabela is not None,
    }


@app.get("/api/debug")
async def debug():
    """Endpoint de debug para verificar configuração."""
    return {
        "service": "XTRI Engine API — TRI V3",
        "version": "3.0.0",
        "tabela_carregada": tabela is not None,
        "tabela_info": tabela.info() if tabela else None,
        "python_version": sys.version,
        "motores_disponiveis": ["heuristico_v2", "cientifico_mirt"],
    }


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT — COMPATIBILIDADE V2 (Motor Heurístico)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/api/calcular-tri", response_model=ResponseCalcularTRI)
async def calcular_tri(request: Request):
    """
    Calcula TRI com motor heurístico (coerência pedagógica).

    Drop-in replacement para a rota V2 do Flask.
    O Node.js chama essa rota e recebe o mesmo JSON de antes.
    """
    if tabela is None:
        raise HTTPException(
            status_code=500,
            detail="Tabela de referência TRI não carregada"
        )

    try:
        # Aceitar o JSON raw para máxima compatibilidade com V2
        data = await request.json()

        if not data or 'alunos' not in data or 'gabarito' not in data:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "erro",
                    "mensagem": "Dados inválidos. Necessário: alunos, gabarito"
                }
            )

        alunos_raw = data['alunos']
        gabarito_raw = data['gabarito']
        areas_config_raw = data.get('areas_config', {
            'LC': [1, 45],
            'CH': [46, 90],
            'CN': [1, 45],
            'MT': [46, 90],
        })

        # Converter gabarito (aceita lista ou dicionário)
        if isinstance(gabarito_raw, list):
            gabarito = {str(i + 1): v for i, v in enumerate(gabarito_raw)}
        else:
            gabarito = {str(k): v for k, v in gabarito_raw.items()}

        # Converter areas_config de list para tuple
        areas_config = {k: tuple(v) for k, v in areas_config_raw.items()}

        # Converter alunos do formato lista para formato qN (compatível com V2)
        alunos = []
        for aluno in alunos_raw:
            aluno_conv = {
                'id': aluno.get('id', ''),
                'nome': aluno.get('nome', ''),
            }

            if 'respostas' in aluno and isinstance(aluno['respostas'], list):
                for i, resp in enumerate(aluno['respostas']):
                    aluno_conv[f'q{i + 1}'] = resp if resp else ''
            else:
                for key, val in aluno.items():
                    if key.startswith('q') or key in [
                        'id', 'nome', 'studentNumber', 'studentName', 'turma'
                    ]:
                        aluno_conv[key] = val

            alunos.append(aluno_conv)

        t0 = time.time()
        logger.info(
            "heuristic: processing %d alunos, %d gabarito items, areas=%s",
            len(alunos),
            len(gabarito),
            list(areas_config.keys()),
        )

        # Processar com motor heurístico
        prova_analysis, resultados = processar_turma(
            alunos=alunos,
            gabarito=gabarito,
            areas_config=areas_config,
            tabela=tabela,
        )

        elapsed = time.time() - t0
        logger.info("heuristic: done in %.2fs — %d resultados", elapsed, len(resultados))

        # Converter tipos numpy para JSON
        return JSONResponse(content=convert_numpy({
            "status": "sucesso",
            "total_alunos": len(alunos),
            "prova_analysis": prova_analysis,
            "resultados": resultados,
        }))

    except Exception as e:
        logger.exception("heuristic: /api/calcular-tri failed")
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "erro",
                "mensagem": str(e),
                "trace": error_trace,
            }
        )


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT — MOTOR CIENTÍFICO (MML + EAP)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/api/v3/calibrar-cientifico", response_model=ResponseCalibrarCientifico)
async def calibrar_cientifico(payload: RequestCalibrarCientifico):
    """
    Calibra itens e estima notas TRI usando modelo 2PL real.

    Ideal para:
    - Fechamento de simulados oficiais
    - Geração de relatórios pedagógicos detalhados
    - Alimentação do banco de itens com parâmetros a, b, c

    Regra inviolável: cada combinação simulado × área é um modelo separado.
    """
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

@app.post("/api/v3/estimar-theta", response_model=ResponseEstimarTheta)
async def estimar_theta(payload: RequestEstimarTheta):
    """
    Estima nota de UM aluno usando parâmetros de itens já calibrados.

    Ideal para:
    - Alunos retardatários (prova já calibrada)
    - Processamento assíncrono de cartões-resposta via OMR
    - Recálculo rápido sem recalibrar toda a turma
    """
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


def _resolve_transformacao_supabase(
    *,
    ano_referencia: int,
    area: str,
) -> Optional[Dict[str, float]]:
    """Busca a transformação `theta -> nota` no Supabase quando disponível."""

    if not is_supabase_configured():
        return None

    try:
        sb = get_supabase()
        tr = (
            sb.table("enem_transformacoes")
            .select("sigma,mu")
            .eq("ano", ano_referencia)
            .eq("area", area)
            .limit(1)
            .execute()
        )
        if tr.data:
            return {
                "sigma": float(tr.data[0]["sigma"]),
                "mu": float(tr.data[0]["mu"]),
            }
    except Exception:
        return None

    return None


def _build_mirt_response(
    *,
    respostas: list[int],
    area: str,
    ano_referencia: int,
    artefato_id: str,
    scoring_factor_id: str,
    scores_mirt: Dict[str, float],
    se_mirt: Dict[str, float],
    transformacao: Optional[Dict[str, float]],
) -> ResponseEstimarThetaMIRT:
    """Normaliza o output multidimensional para o contrato HTTP do backend."""

    if scoring_factor_id not in scores_mirt:
        raise RuntimeMIRTArtifactError(
            f"O fator principal {scoring_factor_id} não foi retornado pelo scoring MIRT."
        )
    if scoring_factor_id not in se_mirt:
        raise RuntimeMIRTArtifactError(
            f"O erro padrão do fator principal {scoring_factor_id} não foi retornado pelo scoring MIRT."
        )

    acertos = int(sum(respostas))
    theta = float(scores_mirt[scoring_factor_id])
    se = float(se_mirt[scoring_factor_id])

    nota = ancorar_nota_individual(
        theta=theta,
        area=area,
        acertos=acertos,
        tabela=tabela,
        transformacao=transformacao,
    )

    return ResponseEstimarThetaMIRT(
        acertos=acertos,
        theta=round(theta, 6),
        se=round(se, 4),
        nota_ancorada=round(float(nota), 1),
        estado_tri=estado_tri(nota),
        artefato_id=artefato_id,
        modelo_usado="mirt",
        fallback_modelo=None,
        scores_mirt={factor_id: round(float(value), 6) for factor_id, value in scores_mirt.items()},
        se_mirt={factor_id: round(float(value), 4) for factor_id, value in se_mirt.items()},
    )


def _resolve_unidim_fallback_response(
    *,
    payload: RequestEstimarThetaMIRT,
    area: str,
    fallback_from: str,
) -> ResponseEstimarThetaMIRT:
    """Reaproveita o fluxo unidimensional atual para o fallback do MIRT."""

    artefato = resolve_unidim_artifact(
        ano_referencia=payload.ano_referencia,
        area=area,
        scope_id=payload.simulado_id,
    )
    params = [
        {"a": p["a"], "b": p["b"], "c": p.get("c", 0.20)}
        for p in artefato["parametros_itens"]
    ]
    if len(payload.respostas) != len(params):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Número de respostas ({len(payload.respostas)}) != número de itens "
                f"do fallback unidimensional ({len(params)})"
            ),
        )
    transformacao = payload.transformacao
    if transformacao is None:
        transformacao = artefato.get("transformacao")
    if transformacao is None:
        transformacao = _resolve_transformacao_supabase(
            ano_referencia=payload.ano_referencia,
            area=area,
        )

    resultado = estimar_theta_individual(
        respostas=payload.respostas,
        parametros_itens=params,
        area=area,
        tabela=tabela,
        transformacao=transformacao,
    )
    return ResponseEstimarThetaMIRT(
        **resultado,
        artefato_id=artefato["manifest"]["artifact_id"],
        modelo_usado="unidim",
        fallback_modelo=fallback_from,
        scores_mirt=None,
        se_mirt=None,
    )


def _try_mirt_score(
    *,
    payload: RequestEstimarThetaMIRT,
    area: str,
) -> ResponseEstimarThetaMIRT:
    """Executa o fluxo de score MIRT a partir de um artefato já calibrado."""

    artefato = resolve_mirt_artifact(
        ano_referencia=payload.ano_referencia,
        area=area,
        artifact_id=payload.artefato_id,
        scope_id=payload.simulado_id,
    )
    qmatrix_rows = artefato["qmatrix"]["rows"]
    item_ids = [row["item_id"] for row in qmatrix_rows]
    if len(payload.respostas) != len(item_ids):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Número de respostas ({len(payload.respostas)}) != número de itens "
                f"na Q-matrix ({len(item_ids)})"
            ),
        )

    if not mirt_runtime_available():
        raise MIRTWorkerExecutionError("`Rscript` não está disponível no runtime local.")

    with tempfile.TemporaryDirectory(prefix="xtri_mirt_score_") as tmpdir:
        input_path = Path(tmpdir) / "single_student.csv"
        output_path = Path(tmpdir) / "scores.csv"

        with open(input_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["student_id", *item_ids])
            writer.writerow(["student_1", *payload.respostas])

        run_mirt_score(
            model_rds_path=artefato["model_rds_path"],
            response_matrix_path=input_path,
            output_path=output_path,
        )

        with open(output_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
        if row is None:
            raise RuntimeMIRTArtifactError("O scoring MIRT não retornou nenhuma linha.")

    scores_mirt: Dict[str, float] = {}
    se_mirt: Dict[str, float] = {}
    for column, raw_value in row.items():
        if column == "student_id" or raw_value in (None, ""):
            continue
        value = float(raw_value)
        if column.startswith("SE_"):
            se_mirt[column[3:]] = value
        else:
            scores_mirt[column] = value

    transformacao = payload.transformacao or _resolve_transformacao_supabase(
        ano_referencia=payload.ano_referencia,
        area=area,
    )
    scoring_factor_id = artefato["worker_output"]["model_spec"]["scoring_factor_id"]
    return _build_mirt_response(
        respostas=payload.respostas,
        area=area,
        ano_referencia=payload.ano_referencia,
        artefato_id=artefato["manifest"]["artifact_id"],
        scoring_factor_id=scoring_factor_id,
        scores_mirt=scores_mirt,
        se_mirt=se_mirt,
        transformacao=transformacao,
    )


@app.post("/api/v5/mirt/estimar-theta", response_model=ResponseEstimarThetaMIRT)
async def estimar_theta_mirt(payload: RequestEstimarThetaMIRT):
    """Estima score multidimensional a partir de artefatos MIRT locais."""

    if tabela is None:
        raise HTTPException(
            status_code=500,
            detail="Tabela de referência TRI não carregada",
        )

    area = payload.area.upper()
    if area not in ["LC", "CH", "CN", "MT"]:
        raise HTTPException(
            status_code=400,
            detail=f"Área inválida: {area}",
        )

    try:
        return _try_mirt_score(payload=payload, area=area)
    except HTTPException:
        raise
    except (
        RuntimeMIRTArtifactNotFoundError,
        RuntimeMIRTArtifactAmbiguousError,
        RuntimeMIRTArtifactError,
        MIRTWorkerExecutionError,
    ) as exc:
        if not payload.permitir_fallback_unidim:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            return _resolve_unidim_fallback_response(
                payload=payload,
                area=area,
                fallback_from="mirt",
            )
        except (RuntimeArtifactNotFoundError, RuntimeArtifactAmbiguousError) as fallback_exc:
            raise HTTPException(
                status_code=400,
                detail=f"{exc} Nenhum fallback unidimensional disponível: {fallback_exc}",
            )


@app.get("/api/v5/mirt/availability")
async def mirt_availability(ano_referencia: int = 2024):
    """Per-area availability of MIRT artifacts for a given reference year.

    Frontend calls this on page load to render the 'MIRT Avançado' status
    card. Returns a boolean per LC/CH/CN/MT plus the total count. Default
    scoring flow is unaffected — this endpoint only reports what's there.
    """
    artefatos = list_active_models_by_family(
        family="mirt", ano_referencia=ano_referencia,
    )
    disponibilidade = {"LC": False, "CH": False, "CN": False, "MT": False}
    for a in artefatos:
        area = str(a.get("area", "")).upper()
        if area in disponibilidade:
            disponibilidade[area] = True
    return {
        "ano_referencia": ano_referencia,
        "disponibilidade": disponibilidade,
        "total_artefatos": len(artefatos),
    }


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT — TABELA DE REFERÊNCIA (Útil para o frontend)
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/v3/referencias/{area}")
async def obter_referencia(area: str):
    """Retorna a tabela de referência oficial para uma área."""
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

@app.get("/api/v3/anos-referencia")
async def anos_referencia():
    """Returns available reference years for the year selector dropdown."""
    return sorted(tabelas_multiano.keys())


def _build_reference_scale_points(
    tab: "TabelaReferencia", area: str,
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
    tab: "TabelaReferencia",
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


@app.post("/api/v3/processar-csv", response_model=ResponseProcessarCSV)
async def processar_csv(
    area: str = Form(...),
    ano_referencia: int = Form(...),
    arquivo: UploadFile = File(...),
    c_fixo: float = Form(0.20),
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


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — V4 VALIDAÇÃO INEP (usa parâmetros oficiais do Supabase)
# ════════════════════════════════════════════════════════════════════════════════

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


@app.get("/api/v4/enem/info", response_model=ENEMInfo)
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


@app.get("/api/v4/enem/transformacao/{ano}/{area}", response_model=TransformacaoENEM)
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


@app.get("/api/v4/enem/provas/{ano}/{area}")
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


@app.get("/api/v4/enem/itens/{ano}/{area}/{co_prova}", response_model=ProvaINEP)
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


@app.post("/api/v4/enem/calcular-com-inep", response_model=ResponseCalcularComINEP)
async def calcular_com_inep(payload: RequestCalcularComINEP):
    """
    Calcula notas TRI usando parâmetros OFICIAIS do INEP (pré-calibrados).

    Diferente do /api/v3/calibrar-cientifico que calibra localmente, este
    endpoint usa os parâmetros exatos publicados pelo INEP para a prova
    especificada + transformação empírica (σ, μ) derivada dos microdados.

    Precisão: bit-a-bit equivalente ao cálculo oficial do INEP.
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
        raise HTTPException(
            status_code=404,
            detail=f"Transformação empírica não disponível para {area}/{ano}",
        )
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


@app.post("/api/v4/enem/validar-simulado", response_model=ResponseValidarSimulado)
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


# ════════════════════════════════════════════════════════════════════════════════
# V5 — MICRODATA ANALYSIS ENDPOINTS (Phase 8)
# ════════════════════════════════════════════════════════════════════════════════
# These routes wire the app/microdata/ pipeline (loader → cache → calibration
# → matplotlib dashboards) into the FastAPI surface. The frontend kicks off
# a long-running pipeline via POST /prepare, polls /job/{id} for status, then
# fetches base64-encoded PNG dashboards once status==done.

import base64  # noqa: PLC0415

from .jobs.calibration_jobs import submit_microdata_job  # noqa: PLC0415
from .microdata import (  # noqa: PLC0415
    build_microdata_csv,
    build_microdata_excel,
    calibrate_all_areas,
    calibration_summary,
    compute_breakdown,
    compute_dashboard_data,
    compute_difficulty_table,
    detect_schema,
    generate_full_report,
    generate_report,
    load_microdata,
    render_area_dashboard,
    render_full_report,
)
from .microdata.cache import (  # noqa: PLC0415
    cache_key,
    load_cached,
    save_cached,
)
from .microdata.jobs import (  # noqa: PLC0415
    JOB_DONE,
    JOB_ERROR,
    JOB_QUEUED,
    JOB_RUNNING,
    job_manager,
)
from .microdata.loader import (  # noqa: PLC0415
    default_csv_path,
    discover_available_years,
    load_primeira_aplicacao_provas,
    MICRODATA_DIR,
)


def _runtime_directory_models() -> dict[str, RuntimeDirectoryStatus]:
    """Normaliza o snapshot de diretórios locais para o response model."""

    snapshot = describe_runtime_layout()
    return {
        name: RuntimeDirectoryStatus(**payload)
        for name, payload in snapshot.items()
    }


def _runtime_job_counts(jobs: list) -> dict[str, int]:
    """Conta jobs por status para o desktop não depender de lógica no cliente."""

    counts = {
        "queued": 0,
        "running": 0,
        "done": 0,
        "error": 0,
    }
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    return counts


def _runtime_model_statuses() -> list[RuntimeModelStatus]:
    """Converte o registry bruto em uma lista ordenada de modelos ativos."""

    models: list[RuntimeModelStatus] = []
    for entry in list_active_models().values():
        artifact_dir = Path(entry["artifact_dir"])
        status = "pronto" if artifact_dir.is_dir() else "desatualizado"
        models.append(
            RuntimeModelStatus(
                ano_referencia=int(entry["ano_referencia"]),
                area=str(entry["area"]),
                scope_id=str(entry["scope_id"]),
                modelo=str(entry["modelo"]),
                artifact_id=entry.get("artifact_id"),
                artifact_dir=str(artifact_dir),
                saved_at_utc=entry.get("saved_at_utc"),
                status=status,
            )
        )
    return sorted(
        models,
        key=lambda model: (
            model.ano_referencia,
            model.area,
            model.scope_id,
            model.saved_at_utc or "",
        ),
        reverse=True,
    )


def _runtime_area_status(
    *,
    ano_referencia: int,
    simulado_id: Optional[str],
    jobs: list,
) -> dict[str, str]:
    """Deriva o status operacional por área para o desktop."""

    if not simulado_id:
        return {}

    has_running = any(job.status in {JOB_QUEUED, JOB_RUNNING} for job in jobs)
    has_error = any(job.status == JOB_ERROR for job in jobs)
    statuses: dict[str, str] = {}

    for area in ("LC", "CH", "CN", "MT"):
        active = get_active_model(
            ano_referencia=ano_referencia,
            area=area,
            scope_id=simulado_id,
        )
        if active is not None:
            statuses[area] = (
                "pronto" if Path(active["artifact_dir"]).is_dir() else "desatualizado"
            )
        elif has_running:
            statuses[area] = "calibrando"
        elif has_error:
            statuses[area] = "falhou"
        else:
            statuses[area] = "fallback_heuristico"

    return statuses


def _run_microdata_pipeline(
    job_id: str, year: int, sample_size: int, rng_seed: int, force_rerun: bool
) -> None:
    """Background runner for one microdata pipeline job.

    Runs in a thread (via asyncio.run_in_executor) so it doesn't block the
    uvicorn worker. Updates job state via the singleton job_manager.
    """
    try:
        job_manager.mark_running(job_id)
        csv_path = default_csv_path(year)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Microdata CSV not found at {csv_path}. "
                f"Drop MICRODADOS_ENEM_{year}.csv in {MICRODATA_DIR}/"
            )

        raw_key = cache_key(csv_path, year=year, sample_size=sample_size, rng_seed=rng_seed)
        calib_key = raw_key.with_stage("calibrated")

        # Stage 1: load + filter + sample (or load from cache)
        job_manager.update_progress(job_id, progress=0.05, stage="loading microdata")
        if not force_rerun:
            cached_calib = load_cached(calib_key)
            if cached_calib is not None:
                # Calibrated cache hit — done immediately
                job_manager.mark_done(job_id, result_key=calib_key)
                return

        df = None
        if not force_rerun:
            df = load_cached(raw_key)
        if df is None:
            job_manager.update_progress(job_id, progress=0.10, stage="streaming CSV")
            df = load_microdata(
                year=year,
                sample_size=sample_size,
                rng_seed=rng_seed,
                provas_primeira=load_primeira_aplicacao_provas(year),
            )
            save_cached(df, raw_key, source_path=csv_path, filters_description="v5 raw")
        job_manager.update_progress(job_id, progress=0.30, stage="raw sample ready")

        # Stage 2: calibrate all areas
        schema = detect_schema(csv_path)
        tabela = tabelas_multiano.get(year)
        if tabela is None:
            raise ValueError(
                f"No reference table loaded for year {year}. "
                f"Available: {sorted(tabelas_multiano.keys())}"
            )

        def on_progress(area, idx, total, n_done):
            # Each area is one quarter of the calibration phase
            area_idx = ["LC", "CH", "CN", "MT"].index(area)
            within_area = idx / total if total > 0 else 1.0
            overall = 0.30 + 0.65 * (area_idx + within_area) / 4
            job_manager.update_progress(
                job_id, progress=overall, stage=f"calibrating {area} ({idx}/{total})"
            )

        enriched = calibrate_all_areas(df, schema, tabela, progress_callback=on_progress)
        job_manager.update_progress(job_id, progress=0.95, stage="caching results")
        save_cached(
            enriched, calib_key, source_path=csv_path, filters_description="v5 calibrated"
        )

        job_manager.mark_done(job_id, result_key=calib_key)
    except Exception as exc:  # noqa: BLE001
        job_manager.mark_error(job_id, f"{type(exc).__name__}: {exc}")


def _summary_to_model(summary_dict: dict) -> "MicrodataAreaSummary":
    """Convert calibration_summary() output for one area into the response model."""
    return MicrodataAreaSummary(
        n=int(summary_dict.get("n", 0)),
        correlation=summary_dict.get("correlation"),
        mae=summary_dict.get("mae"),
        rmse=summary_dict.get("rmse"),
        mean_v3=summary_dict.get("mean_v3"),
        mean_ref=summary_dict.get("mean_ref"),
        mean_shift=summary_dict.get("mean_shift"),
    )


@app.get("/api/v5/microdata/anos-disponiveis", response_model=MicrodataAvailabilityResponse)
async def v5_anos_disponiveis():
    """List ENEM years that have a MICRODADOS_ENEM_<year>.csv on disk."""
    return MicrodataAvailabilityResponse(
        anos=discover_available_years(),
        microdata_dir=str(MICRODATA_DIR),
    )


@app.get("/api/v5/microdata/status/{year}", response_model=MicrodataYearStatus)
async def v5_microdata_status(year: int):
    """Report file presence + cache state for one year. Always 200 (no 404)."""
    csv_path = default_csv_path(year)
    csv_exists = csv_path.is_file()

    csv_size_mb = None
    csv_mtime = None
    raw_present = False
    calib_present = False
    n_sample = None

    if csv_exists:
        stat = csv_path.stat()
        csv_size_mb = round(stat.st_size / (1024 * 1024), 1)
        csv_mtime = stat.st_mtime
        # Check both raw and calibrated cache existence for the default 200K sample
        try:
            raw_key = cache_key(csv_path, year=year, sample_size=200_000, rng_seed=42)
            raw_present = raw_key.parquet_path.is_file()
            calib_present = raw_key.with_stage("calibrated").parquet_path.is_file()
            if calib_present:
                n_sample = 200_000
            elif raw_present:
                n_sample = 200_000
        except Exception:  # noqa: BLE001
            pass

    return MicrodataYearStatus(
        year=year,
        csv_path=str(csv_path),
        csv_exists=csv_exists,
        csv_size_mb=csv_size_mb,
        csv_mtime=csv_mtime,
        cache_raw_present=raw_present,
        cache_calibrated_present=calib_present,
        n_sample=n_sample,
    )


@app.post("/api/v5/microdata/prepare", response_model=MicrodataPrepareResponse)
async def v5_microdata_prepare(req: MicrodataPrepareRequest):
    """Kick off the load + calibrate pipeline asynchronously.

    Returns immediately with a job_id. Frontend polls /job/{id} for progress.
    If the calibrated cache for this (year, sample_size, seed) already exists
    AND force_rerun is False, the job completes synchronously inside the
    executor and the next /job/{id} call returns ``done``.
    """
    csv_path = default_csv_path(req.year)
    if not csv_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Microdata CSV not found at {csv_path}. "
                f"Download MICRODADOS_ENEM_{req.year}.csv from INEP and copy to "
                f"{MICRODATA_DIR}/"
            ),
        )

    job = job_manager.create(
        year=req.year, sample_size=req.sample_size, rng_seed=req.rng_seed
    )

    # Enfileirar no executor dedicado do runtime local.
    submit_microdata_job(
        _run_microdata_pipeline,
        job_id=job.job_id,
        year=req.year,
        sample_size=req.sample_size,
        rng_seed=req.rng_seed,
        force_rerun=req.force_rerun,
    )

    return MicrodataPrepareResponse(
        job_id=job.job_id,
        status=job.status,
        message=f"Pipeline started for year {req.year}, sample={req.sample_size}",
    )


@app.get("/api/v5/microdata/job/{job_id}", response_model=MicrodataJobStatusResponse)
async def v5_microdata_job(job_id: str):
    """Poll a running pipeline job."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return MicrodataJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        started_at=job.started_at,
        elapsed_seconds=job.elapsed_seconds,
        error=job.error,
    )


@app.get("/api/v5/runtime/status", response_model=RuntimeDesktopStatusResponse)
async def v5_runtime_status(
    ano_referencia: int = 2024,
    simulado_id: Optional[str] = None,
):
    """Snapshot do runtime local para o desktop/WebView."""

    ensure_app_support_layout()
    jobs = job_manager.snapshot()
    directories = _runtime_directory_models()

    return RuntimeDesktopStatusResponse(
        app_support_dir=directories["app_support"].path,
        directories=directories,
        models=_runtime_model_statuses(),
        job_counts=_runtime_job_counts(jobs),
        jobs=[
            RuntimeJobStatus(
                job_id=job.job_id,
                year=job.year,
                sample_size=job.sample_size,
                rng_seed=job.rng_seed,
                status=job.status,
                progress=job.progress,
                stage=job.stage,
                elapsed_seconds=job.elapsed_seconds,
                error=job.error,
            )
            for job in jobs
        ],
        ano_referencia=ano_referencia,
        simulado_id=simulado_id,
        status_areas=_runtime_area_status(
            ano_referencia=ano_referencia,
            simulado_id=simulado_id,
            jobs=jobs,
        ),
    )


def _load_calibrated_or_404(year: int) -> "tuple":
    """Helper: load the calibrated DataFrame for a year or raise 404.

    Returns ``(df, csv_path)``. Used by both dashboard endpoints below.
    """
    csv_path = default_csv_path(year)
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"Microdata CSV not found for year {year}")
    raw_key = cache_key(csv_path, year=year, sample_size=200_000, rng_seed=42)
    calib_key = raw_key.with_stage("calibrated")
    df = load_cached(calib_key)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No calibrated cache for year {year}. "
                f"POST /api/v5/microdata/prepare first."
            ),
        )
    return df, csv_path


@app.get(
    "/api/v5/microdata/dashboard/{year}",
    response_model=MicrodataAreaDashboardResponse,
)
async def v5_microdata_dashboard(year: int, area: str):
    """Single-area dashboard: 1×3 PNG + summary + difficulty table."""
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Invalid area: {area}")

    df, _ = _load_calibrated_or_404(year)

    summary = calibration_summary(df).get(area.lower(), {"n": 0})
    png_bytes = render_area_dashboard(df, area)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    diff_rows = compute_difficulty_table(df, area)

    return MicrodataAreaDashboardResponse(
        year=year,
        area=area,
        n_students=int(summary.get("n", 0)),
        summary=_summary_to_model(summary),
        dashboard_png_base64=png_b64,
        difficulty_table=[MicrodataDifficultyRow(**r) for r in diff_rows],
    )


@app.get(
    "/api/v5/microdata/full-report/{year}",
    response_model=MicrodataFullReportResponse,
)
async def v5_microdata_full_report(year: int):
    """4×3 full-report PNG + per-area summaries."""
    df, _ = _load_calibrated_or_404(year)

    summaries_raw = calibration_summary(df)
    summaries_models = {area: _summary_to_model(s) for area, s in summaries_raw.items()}
    png_bytes = render_full_report(df)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")

    return MicrodataFullReportResponse(
        year=year,
        n_students=len(df),
        summaries=summaries_models,
        full_report_png_base64=png_b64,
    )


@app.get("/api/v5/microdata/dashboard-data/{year}")
async def v5_microdata_dashboard_data(year: int, area: str):
    """INEP-only JSON payload for the interactive Plotly frontend.

    Returns histogram bins, scatter points (acertos × nota colored by acert_dif),
    and per-acertos boxplot data — all computed from the official ``NU_NOTA_*``
    columns in the microdata. NO V3 Local references; this endpoint is for the
    Phase 8 microdata page which is strictly an INEP-only analysis.

    Use this instead of /dashboard/{year} (which returns a static base64 PNG).
    """
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Invalid area: {area}")
    df, _ = _load_calibrated_or_404(year)
    payload = compute_dashboard_data(df, area)
    payload["year"] = year
    return payload


@app.get("/api/v5/microdata/breakdown/{year}")
async def v5_microdata_breakdown(
    year: int,
    area: str,
    by: str = "uf",
    min_n: Optional[int] = None,
    limit: Optional[int] = None,
):
    """Per-group INEP aggregation for the drill-down UI.

    Query params:
        area:  LC | CH | CN | MT
        by:    uf | municipio | escola
        min_n: drop groups with fewer students (default 1 for uf, 5 otherwise)
        limit: max rows (default None for uf, 500 municipio, 1000 escola)

    Returns a list of dicts sorted by median desc. Stats come exclusively from
    the official ``NU_NOTA_*`` columns (``nota_ref_*``) — V3 is ignored.
    """
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Invalid area: {area}")
    by_lower = by.lower()
    if by_lower not in ("uf", "municipio", "escola"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid by: {by} (must be uf, municipio, or escola)",
        )
    df, _ = _load_calibrated_or_404(year)
    rows = compute_breakdown(
        df, area, group_by=by_lower, min_n=min_n, limit=limit  # type: ignore[arg-type]
    )
    return {
        "year": year,
        "area": area,
        "by": by_lower,
        "count": len(rows),
        "rows": rows,
    }


@app.post("/api/v5/microdata/report/{year}")
async def v5_microdata_report(year: int, area: str):
    """Generate a Claude-authored markdown narrative for one area of one year.

    Assembles INEP summary stats + difficulty-table highlights + top-10 UFs,
    cities and schools into a grounded prompt, then calls the Anthropic API
    (model defaults to ``claude-sonnet-4-6``, overridable via ``XTRI_REPORT_MODEL``).

    Requires ``ANTHROPIC_API_KEY`` to be set either in the environment or in
    ``~/Library/Application Support/XTRI Engine/.env``. Returns ``501`` when
    the key is missing so the frontend can show a clear fallback message.
    """
    area = area.upper()
    if area not in ("LC", "CH", "CN", "MT"):
        raise HTTPException(status_code=400, detail=f"Invalid area: {area}")
    df, _ = _load_calibrated_or_404(year)

    try:
        result = generate_report(df, year, area)
    except RuntimeError as exc:
        # Missing API key → 501 Not Implemented (feature unavailable)
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="anthropic package not installed in this build",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        import logging as _logging  # noqa: PLC0415
        _logging.getLogger(__name__).exception("AI report generation failed")
        raise HTTPException(
            status_code=502, detail=f"Claude API call failed: {exc}"
        ) from exc

    return result


@app.post("/api/v5/microdata/report/{year}/full")
async def v5_microdata_report_full(year: int):
    """Generate a single Claude narrative covering all four ENEM areas.

    Builds payloads for LC, CH, CN and MT simultaneously, then sends one
    comprehensive prompt to Claude Opus requesting a comparative analysis.
    Expect 30–90 s of processing time.  Returns ``501`` when the API key is
    missing, ``502`` on API failure.
    """
    df, _ = _load_calibrated_or_404(year)

    try:
        result = generate_full_report(df, year)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="anthropic package not installed in this build",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        import logging as _logging  # noqa: PLC0415
        _logging.getLogger(__name__).exception("Full AI report generation failed")
        raise HTTPException(
            status_code=502, detail=f"Claude API call failed: {exc}"
        ) from exc

    return result


@app.get("/api/v5/microdata/export-excel/{year}")
async def v5_microdata_export_excel(year: int, include_students: bool = False):
    """Stream the multi-sheet Excel report for one ENEM year.

    Sheets:
        Resumo               — title + per-area metrics
        Relatorio Completo   — embedded 4×3 full report PNG
        <area> sheets        — embedded per-area dashboard PNG + INEP empirical
                               difficulty table (one per LC/CH/CN/MT)
        Alunos (optional)    — full per-student data, only when
                               ``?include_students=true``. Adds ~50s + 25 MB.
                               For raw per-student data the CSV endpoint is
                               much faster — see /api/v5/microdata/export-csv/

    Default mode (without ``include_students``) builds in ~10s and is ~5 MB.

    Returns 404 if the calibrated cache for ``year`` is not on disk.
    """
    df, _ = _load_calibrated_or_404(year)
    summaries = calibration_summary(df)
    xlsx_bytes = build_microdata_excel(
        df, year, summaries, include_students=include_students
    )

    suffix = "_full" if include_students else ""
    filename = f"xtri_microdados_{year}{suffix}.xlsx"
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(xlsx_bytes)),
        },
    )


@app.get("/api/v5/microdata/export-csv/{year}")
async def v5_microdata_export_csv(year: int):
    """Stream the per-student CSV with diff columns for one ENEM year.

    Much faster than the Excel "Alunos" sheet (pandas to_csv vs openpyxl
    serialization is ~30x). UTF-8 with BOM so Excel auto-detects encoding.
    Use this when you need the raw per-student data; use the Excel endpoint
    for the dashboards + summary report.

    Returns 404 if the calibrated cache for ``year`` is not on disk.
    """
    df, _ = _load_calibrated_or_404(year)
    csv_bytes = build_microdata_csv(df)
    filename = f"xtri_microdados_alunos_{year}.csv"
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(csv_bytes)),
        },
    )


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT — EXPORTAR EXCEL
# ════════════════════════════════════════════════════════════════════════════════

def _downloads_export_root() -> Path:
    """Diretório visível ao usuário para exportações geradas localmente."""

    export_root = Path.home() / "Downloads" / "XTRI Engine"
    export_root.mkdir(parents=True, exist_ok=True)
    return export_root


def _save_local_export(filename: str, payload: bytes) -> dict[str, Any]:
    """Salva um arquivo de exportação em Downloads e retorna o metadata básico."""

    destination = _downloads_export_root() / filename
    destination.write_bytes(payload)
    return {
        "status": "sucesso",
        "nome_arquivo": destination.name,
        "caminho": str(destination),
        "bytes": len(payload),
    }


def _build_resultados_csv_bytes(payload: Dict[str, Any]) -> tuple[bytes, str]:
    """Gera o CSV do motor principal com BOM para abrir corretamente no Excel."""

    from datetime import datetime

    area = str(payload.get("area", "")).upper()
    ano = int(payload.get("ano_referencia", 0) or 0)
    resultados = payload.get("resultados", [])
    if not area or not isinstance(resultados, list):
        raise ValueError("Payload inválido para exportação CSV")

    area_lower = area.lower()
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(["ID", "Acertos", "Theta", "Nota_TRI", "Pct_Dificeis", "Nivel"])
    for row in resultados:
        writer.writerow(
            [
                row.get("aluno_id", ""),
                row.get(f"acertos_{area_lower}", ""),
                row.get(f"theta_{area_lower}", ""),
                row.get(f"nota_{area_lower}", ""),
                row.get(f"acert_dif_{area_lower}", ""),
                row.get(f"estado_{area_lower}", ""),
            ]
        )

    csv_bytes = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    filename = f"xtri_{area}_{ano}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return csv_bytes, filename


def _build_resultados_excel_bytes(payload: Dict[str, Any]) -> tuple[bytes, str]:
    """Gera o XLSX do motor principal para streaming ou save local."""

    import base64
    from datetime import datetime

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    area = payload.get("area", "").upper()
    ano = payload.get("ano_referencia", 0)
    metodo = payload.get("metodo", "")
    n_alunos = payload.get("n_alunos", 0)
    resultados = payload.get("resultados", [])
    referencia = payload.get("referencia", [])
    diagnosticos = payload.get("diagnosticos", [])
    chart_b64 = payload.get("chart_png_base64", "")
    area_lower = area.lower()

    wb = Workbook()

    # Styles
    header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    title_font = Font(bold=True, size=14, color="2C3E50")
    subtitle_font = Font(italic=True, size=10, color="666666")
    thin = Side(border_style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")

    # ═══ Sheet 1: RESUMO ═══
    ws1 = wb.active
    ws1.title = "Resumo"
    ws1["A1"] = "XTRI ENGINE — Relatorio de Correcao TRI"
    ws1["A1"].font = title_font
    ws1.merge_cells("A1:D1")
    ws1["A2"] = f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    ws1["A2"].font = subtitle_font
    ws1.merge_cells("A2:D2")

    ws1["A4"] = "Parametros da Correcao"
    ws1["A4"].font = Font(bold=True, size=12, color="2C3E50")
    params = [
        ("Area", area),
        ("Ano de Referencia ENEM", ano),
        ("Metodo de Ancoragem", metodo),
        ("Total de Alunos", n_alunos),
    ]
    for i, (k, v) in enumerate(params, start=5):
        ws1[f"A{i}"] = k
        ws1[f"A{i}"].font = Font(bold=True)
        ws1[f"B{i}"] = v

    ws1["A10"] = "Estatisticas da Turma"
    ws1["A10"].font = Font(bold=True, size=12, color="2C3E50")
    if diagnosticos:
        d = diagnosticos[0]
        stats = [
            ("Nota Media", d.get("nota_media", "")),
            ("Nota Minima", d.get("nota_min", "")),
            ("Nota Maxima", d.get("nota_max", "")),
            ("b medio (dificuldade media)", d.get("b_medio", "")),
            ("b desvio padrao", d.get("b_desvio", "")),
            ("% dentro do intervalo ENEM", f"{d.get('pct_dentro_intervalo', 0)}%"),
            ("Itens faceis (b < -1.5)", d.get("itens_faceis", 0)),
            ("Itens medios", d.get("itens_medios", 0)),
            ("Itens dificeis (b > 1.5)", d.get("itens_dificeis", 0)),
            ("acert_dif medio (Primi)", d.get("acert_dif_medio", 0)),
        ]
        for i, (k, v) in enumerate(stats, start=11):
            ws1[f"A{i}"] = k
            ws1[f"A{i}"].font = Font(bold=True)
            ws1[f"B{i}"] = v

    niveis_count: Dict[str, int] = {}
    for row in resultados:
        estado = row.get(f"estado_{area_lower}")
        if estado:
            niveis_count[estado] = niveis_count.get(estado, 0) + 1

    ws1["A23"] = "Distribuicao por Nivel Pedagogico"
    ws1["A23"].font = Font(bold=True, size=12, color="2C3E50")
    ordem = ["AVANCADO", "PROFICIENTE", "BASICO", "ELEMENTAR", "INICIANTE"]
    for i, nivel in enumerate(ordem, start=24):
        ws1[f"A{i}"] = nivel
        ws1[f"A{i}"].font = Font(bold=True)
        ws1[f"B{i}"] = niveis_count.get(nivel, 0)
        if n_alunos > 0:
            ws1[f"C{i}"] = f"{niveis_count.get(nivel, 0) / n_alunos * 100:.1f}%"

    ws1.column_dimensions["A"].width = 35
    ws1.column_dimensions["B"].width = 18
    ws1.column_dimensions["C"].width = 12

    # ═══ Sheet 2: ALUNOS ═══
    ws2 = wb.create_sheet("Alunos")
    headers = ["ID", "Acertos", "Theta", "Nota TRI", "% Dificeis (acert_dif)", "Nivel"]
    for col_idx, header in enumerate(headers, start=1):
        cell = ws2.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center
        cell.border = border

    for row_idx, row in enumerate(resultados, start=2):
        ws2.cell(row=row_idx, column=1, value=row.get("aluno_id", ""))
        ws2.cell(row=row_idx, column=2, value=row.get(f"acertos_{area_lower}"))
        ws2.cell(row=row_idx, column=3, value=row.get(f"theta_{area_lower}"))
        ws2.cell(row=row_idx, column=4, value=row.get(f"nota_{area_lower}"))
        ws2.cell(row=row_idx, column=5, value=row.get(f"acert_dif_{area_lower}"))
        ws2.cell(row=row_idx, column=6, value=row.get(f"estado_{area_lower}"))
        for c in range(1, 7):
            ws2.cell(row=row_idx, column=c).border = border
            ws2.cell(row=row_idx, column=c).alignment = center if c > 1 else left

    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 10
    ws2.column_dimensions["C"].width = 12
    ws2.column_dimensions["D"].width = 12
    ws2.column_dimensions["E"].width = 22
    ws2.column_dimensions["F"].width = 15
    ws2.freeze_panes = "A2"

    # ═══ Sheet 3: TABELA DIFICULDADES ═══
    ws3 = wb.create_sheet("Tabela Dificuldades")
    ws3["A1"] = f"Tabela de Dificuldades — {area} / ENEM {ano}"
    ws3["A1"].font = title_font
    ws3.merge_cells("A1:G1")
    ws3["A2"] = "Comparacao entre a distribuicao do simulado e os limites oficiais ENEM"
    ws3["A2"].font = subtitle_font
    ws3.merge_cells("A2:G2")

    from collections import defaultdict

    freq_ac: Dict[int, int] = defaultdict(int)
    notas_ac: Dict[int, list] = defaultdict(list)
    for row in resultados:
        acertos = row.get(f"acertos_{area_lower}")
        nota = row.get(f"nota_{area_lower}")
        if acertos is not None and nota is not None:
            freq_ac[acertos] += 1
            notas_ac[acertos].append(nota)

    table_headers = [
        "Acertos",
        "Freq Simulado",
        "Ref Min",
        "Ref Med",
        "Ref Max",
        "Nota Sim (media)",
        "Nota Sim (min-max)",
    ]
    for col_idx, header in enumerate(table_headers, start=1):
        cell = ws3.cell(row=4, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center
        cell.border = border

    for i, ponto in enumerate(referencia, start=5):
        acertos = ponto["acertos"]
        notas = notas_ac.get(acertos, [])
        ws3.cell(row=i, column=1, value=acertos)
        ws3.cell(row=i, column=2, value=freq_ac.get(acertos, 0))
        ws3.cell(row=i, column=3, value=ponto["tri_min"])
        ws3.cell(row=i, column=4, value=ponto["tri_med"])
        ws3.cell(row=i, column=5, value=ponto["tri_max"])
        if notas:
            ws3.cell(row=i, column=6, value=round(sum(notas) / len(notas), 1))
            ws3.cell(row=i, column=7, value=f"{min(notas):.0f} - {max(notas):.0f}")
        for c in range(1, 8):
            ws3.cell(row=i, column=c).border = border
            ws3.cell(row=i, column=c).alignment = center

    ws3.column_dimensions["A"].width = 10
    ws3.column_dimensions["B"].width = 14
    ws3.column_dimensions["C"].width = 12
    ws3.column_dimensions["D"].width = 12
    ws3.column_dimensions["E"].width = 12
    ws3.column_dimensions["F"].width = 18
    ws3.column_dimensions["G"].width = 20
    ws3.freeze_panes = "A5"

    # ═══ Sheet 4: GRAFICO (if chart image provided) ═══
    if chart_b64:
        try:
            from openpyxl.drawing.image import Image as XLImage

            if "," in chart_b64:
                chart_b64 = chart_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(chart_b64)
            img_stream = io.BytesIO(img_bytes)
            ws4 = wb.create_sheet("Grafico")
            ws4["A1"] = f"Grafico de Ancoragem — {area} / ENEM {ano}"
            ws4["A1"].font = title_font
            ws4.merge_cells("A1:H1")
            xlimg = XLImage(img_stream)
            xlimg.width = 900
            xlimg.height = 520
            ws4.add_image(xlimg, "A3")
        except Exception as exc:  # noqa: BLE001
            logger.warning("excel: failed to embed chart image: %s", exc)

    buf = io.BytesIO()
    wb.save(buf)
    filename = f"xtri_{area}_{ano}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return buf.getvalue(), filename


def _build_validator_csv_bytes(payload: Dict[str, Any]) -> tuple[bytes, str]:
    """Gera o CSV da comparação V3 x V4 do validator."""

    from datetime import datetime

    area = str(payload.get("area", "")).upper()
    ano = int(payload.get("ano", 0) or 0)
    resultados = payload.get("resultados", [])
    if not area or not isinstance(resultados, list):
        raise ValueError("Payload inválido para exportação do validator")

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(
        [
            "ID",
            "Acertos",
            "Theta_V3",
            "Theta_V4",
            "Nota_V3",
            "Nota_V4",
            "Diff",
            "Estado_V3",
            "Estado_V4",
            "Match",
        ]
    )
    for row in resultados:
        writer.writerow(
            [
                row.get("aluno_id", ""),
                row.get("acertos", ""),
                row.get("theta_v3", ""),
                row.get("theta_v4", ""),
                row.get("nota_v3", ""),
                row.get("nota_v4", ""),
                row.get("diff", ""),
                row.get("estado_v3", ""),
                row.get("estado_v4", ""),
                "SIM" if row.get("estado_v3") == row.get("estado_v4") else "NAO",
            ]
        )

    csv_bytes = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    filename = (
        f"xtri_validator_{area}_{ano}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    return csv_bytes, filename


def _build_validator_excel_bytes(payload: Dict[str, Any]) -> tuple[bytes, str]:
    """Gera o XLSX da comparação V3 x V4 para uso no validator."""

    from datetime import datetime

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    area = str(payload.get("area", "")).upper()
    ano = int(payload.get("ano", 0) or 0)
    co_prova = str(payload.get("co_prova", ""))
    estatisticas = payload.get("estatisticas", {}) or {}
    transformacao = payload.get("transformacao_inep", {}) or {}
    resultados = payload.get("resultados", []) or []
    avisos = payload.get("avisos", []) or []

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Resumo"

    header_fill = PatternFill(start_color="1E3C72", end_color="1E3C72", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    title_font = Font(bold=True, size=14, color="1E3C72")
    subtitle_font = Font(italic=True, size=10, color="666666")
    thin = Side(border_style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")

    ws1["A1"] = "XTRI Validator — Comparação V3 vs V4"
    ws1["A1"].font = title_font
    ws1.merge_cells("A1:D1")
    ws1["A2"] = f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    ws1["A2"].font = subtitle_font
    ws1.merge_cells("A2:D2")

    params = [
        ("Ano ENEM", ano),
        ("Área", area),
        ("CO_PROVA", co_prova),
        ("Alunos", estatisticas.get("n_alunos", len(resultados))),
        ("Itens", payload.get("n_itens", "")),
        ("Tempo (s)", payload.get("tempo_processamento", "")),
    ]
    ws1["A4"] = "Parâmetros"
    ws1["A4"].font = Font(bold=True, size=12, color="1E3C72")
    for i, (k, v) in enumerate(params, start=5):
        ws1[f"A{i}"] = k
        ws1[f"A{i}"].font = Font(bold=True)
        ws1[f"B{i}"] = v

    metrics = [
        ("Correlação", estatisticas.get("correlacao", "")),
        ("MAE", estatisticas.get("mae", "")),
        ("RMSE", estatisticas.get("rmse", "")),
        ("Max |diff|", estatisticas.get("max_diff_abs", "")),
        ("Concordância de nível (%)", estatisticas.get("pct_concordancia_nivel", "")),
        ("Média V3", estatisticas.get("nota_media_v3", "")),
        ("Média V4", estatisticas.get("nota_media_v4", "")),
        ("Sigma INEP", transformacao.get("sigma", "")),
        ("Mu INEP", transformacao.get("mu", "")),
    ]
    ws1["A13"] = "Métricas"
    ws1["A13"].font = Font(bold=True, size=12, color="1E3C72")
    for i, (k, v) in enumerate(metrics, start=14):
        ws1[f"A{i}"] = k
        ws1[f"A{i}"].font = Font(bold=True)
        ws1[f"B{i}"] = v

    if avisos:
        ws1["A25"] = "Avisos"
        ws1["A25"].font = Font(bold=True, size=12, color="1E3C72")
        for i, warning in enumerate(avisos, start=26):
            ws1[f"A{i}"] = f"• {warning}"
            ws1.merge_cells(start_row=i, start_column=1, end_row=i, end_column=4)

    ws1.column_dimensions["A"].width = 34
    ws1.column_dimensions["B"].width = 18
    ws1.column_dimensions["C"].width = 12
    ws1.column_dimensions["D"].width = 12

    ws2 = wb.create_sheet("Comparacao")
    headers = [
        "ID",
        "Acertos",
        "Theta V3",
        "Theta V4",
        "Nota V3",
        "Nota V4",
        "Diff",
        "Nivel V3",
        "Nivel V4",
        "Match",
    ]
    for col_idx, header in enumerate(headers, start=1):
        cell = ws2.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center
        cell.border = border

    for row_idx, row in enumerate(resultados, start=2):
        values = [
            row.get("aluno_id", ""),
            row.get("acertos", ""),
            row.get("theta_v3", ""),
            row.get("theta_v4", ""),
            row.get("nota_v3", ""),
            row.get("nota_v4", ""),
            row.get("diff", ""),
            row.get("estado_v3", ""),
            row.get("estado_v4", ""),
            "SIM" if row.get("estado_v3") == row.get("estado_v4") else "NAO",
        ]
        for col_idx, value in enumerate(values, start=1):
            cell = ws2.cell(row=row_idx, column=col_idx, value=value)
            cell.border = border
            cell.alignment = center

    for col in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"):
        ws2.column_dimensions[col].width = 16 if col != "A" else 28
    ws2.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    filename = (
        f"xtri_validator_{area}_{ano}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    return buf.getvalue(), filename


@app.post("/api/v3/exportar-csv")
async def exportar_csv(payload: Dict[str, Any] = Body(...), salvar_localmente: bool = False):
    """Exporta o CSV do motor principal; no app local pode salvar em Downloads."""

    try:
        csv_bytes, filename = _build_resultados_csv_bytes(payload)
        if salvar_localmente:
            return JSONResponse(_save_local_export(filename, csv_bytes))
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(csv_bytes)),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("exportar-csv failed")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar CSV: {exc}")


@app.post("/api/v3/exportar-excel")
async def exportar_excel(payload: Dict[str, Any] = Body(...), salvar_localmente: bool = False):
    """
    Generate an XLSX file with all results: students, difficulty table,
    diagnostics, and the anchor chart as an embedded image.

    When ``salvar_localmente`` is true, writes the file to Downloads and
    returns the destination path instead of streaming it through the WebView.
    """
    try:
        xlsx_bytes, filename = _build_resultados_excel_bytes(payload)
        if salvar_localmente:
            return JSONResponse(_save_local_export(filename, xlsx_bytes))
        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(xlsx_bytes)),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("exportar-excel failed")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar Excel: {exc}")


@app.post("/api/v4/enem/exportar-validacao-csv")
async def exportar_validacao_csv(
    payload: Dict[str, Any] = Body(...), salvar_localmente: bool = False
):
    """Exporta o comparativo do validator em CSV."""

    try:
        csv_bytes, filename = _build_validator_csv_bytes(payload)
        if salvar_localmente:
            return JSONResponse(_save_local_export(filename, csv_bytes))
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(csv_bytes)),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("exportar-validacao-csv failed")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar CSV do validator: {exc}")


@app.post("/api/v4/enem/exportar-validacao-excel")
async def exportar_validacao_excel(
    payload: Dict[str, Any] = Body(...), salvar_localmente: bool = False
):
    """Exporta o comparativo do validator em XLSX."""

    try:
        xlsx_bytes, filename = _build_validator_excel_bytes(payload)
        if salvar_localmente:
            return JSONResponse(_save_local_export(filename, xlsx_bytes))
        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(xlsx_bytes)),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("exportar-validacao-excel failed")
        raise HTTPException(
            status_code=500, detail=f"Erro ao gerar Excel do validator: {exc}"
        )


# ════════════════════════════════════════════════════════════════════════════════
# FRONTEND — Static Files (must be AFTER all API routes)
# ════════════════════════════════════════════════════════════════════════════════

_static_dir = os.path.join(os.path.dirname(__file__), "static")


_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/", include_in_schema=False)
async def frontend():
    """Serve the frontend application."""
    return FileResponse(os.path.join(_static_dir, "index.html"), headers=_NO_CACHE_HEADERS)


@app.get("/validator", include_in_schema=False)
async def frontend_validator():
    """Serve the INEP validator page."""
    return FileResponse(os.path.join(_static_dir, "validator.html"), headers=_NO_CACHE_HEADERS)


@app.get("/microdata", include_in_schema=False)
async def frontend_microdata():
    """Serve the microdata analysis page (Phase 8)."""
    return FileResponse(os.path.join(_static_dir, "microdata.html"), headers=_NO_CACHE_HEADERS)


if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
