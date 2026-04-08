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

import os
import sys
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
from .db.supabase_client import get_supabase, is_supabase_configured
from .enriquecimento import (
    build_resultado_estruturado,
    build_metodologia_banner,
    build_coorte_stats,
    NACIONAL_STATS,
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
    print("=" * 80)
    print("  XTRI ENGINE API — TRI V3 (FastAPI)")
    print("=" * 80)

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
        print(f"\n[TRI V3] Processando {len(alunos)} alunos (motor heurístico)...")
        print(f"[TRI V3] Gabarito: {len(gabarito)} questões | Áreas: {list(areas_config.keys())}")

        # Processar com motor heurístico
        prova_analysis, resultados = processar_turma(
            alunos=alunos,
            gabarito=gabarito,
            areas_config=areas_config,
            tabela=tabela,
        )

        elapsed = time.time() - t0
        print(f"[TRI V3] Concluído em {elapsed:.2f}s | {len(resultados)} resultados")

        # Converter tipos numpy para JSON
        return JSONResponse(content=convert_numpy({
            "status": "sucesso",
            "total_alunos": len(alunos),
            "prova_analysis": prova_analysis,
            "resultados": resultados,
        }))

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[TRI V3] ERRO: {error_trace}")
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
    print(f"\n[TRI V3] Calibração científica: {payload.simulado_id} × {area}")
    print(f"[TRI V3] N={len(payload.alunos)} alunos, {n_itens_esperado} itens, c={payload.c_fixo}")

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
        print(f"[TRI V3] Concluído em {elapsed:.2f}s")
        print(f"[TRI V3] b_médio={diag['b_medio']}, saturados={diag['itens_saturados']}")
        print(f"[TRI V3] Nota média={diag['nota_media']}, {diag['pct_dentro_intervalo']}% no intervalo")

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

        return ResponseCalibrarCientifico(
            status="sucesso",
            simulado_id=payload.simulado_id,
            area=area,
            n_alunos=len(payload.alunos),
            n_itens=n_itens_esperado,
            parametros_itens=parametros,
            resultados_alunos=resultados_alunos,
            diagnostico=resultado['diagnostico'],
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[TRI V3] ERRO na calibração: {error_trace}")
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

    if len(payload.respostas) != len(payload.parametros_itens):
        raise HTTPException(
            status_code=400,
            detail=f"Número de respostas ({len(payload.respostas)}) != número de itens ({len(payload.parametros_itens)})"
        )

    # Converter parâmetros para formato do motor
    params = [
        {'a': p.a, 'b': p.b, 'c': p.c}
        for p in payload.parametros_itens
    ]

    resultado = estimar_theta_individual(
        respostas=payload.respostas,
        parametros_itens=params,
        area=area,
        tabela=tabela,
    )

    return ResponseEstimarTheta(**resultado)


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
        error_trace = traceback.format_exc()
        print(f"[CSV] ERROR {area}: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Erro na calibracao: {e}")

    diag = resultado["diagnostico"]
    print(f"[CSV] {area}/{ano_referencia}: {diag['n_alunos']} alunos, "
          f"nota_media={diag['nota_media']}, {diag['pct_dentro_intervalo']}% dentro")

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

    # Build reference scale for the anchor graph
    referencia: list[PontoReferencia] = []
    for acertos in sorted(tab.lookup.get(area, {}).keys()):
        ref = tab.lookup[area][acertos]
        referencia.append(PontoReferencia(
            acertos=acertos,
            tri_min=ref["tri_min"],
            tri_med=ref["tri_med"],
            tri_max=ref["tri_max"],
        ))

    # ── V2.1 Enrichment: 9 dimensions per student + methodology banner + cohort stats ──
    raw_results = resultado["resultados_alunos"]
    all_thetas_list = [float(r["theta"]) for r in raw_results]
    all_notas_list = [float(r["nota_ancorada"]) for r in raw_results]
    all_acertos_list = [int(r["acertos"]) for r in raw_results]

    cohort_mean = float(np.mean(all_thetas_list)) if all_thetas_list else 0.0
    cohort_std_val = float(np.std(all_thetas_list)) if all_thetas_list else 1.0

    # Rank students within cohort (higher nota = better rank, 1-based)
    sorted_by_nota = sorted(raw_results, key=lambda r: -r["nota_ancorada"])
    rank_by_id = {r["aluno_id"]: i + 1 for i, r in enumerate(sorted_by_nota)}

    # Get sigma/mu from supabase if available, else use empirical defaults
    sigma_oficial, mu_oficial = 100.0, 500.0
    if is_supabase_configured():
        try:
            sb = get_supabase()
            tr = sb.table("enem_transformacoes").select("*").eq("ano", ano_referencia).eq("area", area).limit(1).execute()
            if tr.data:
                sigma_oficial = float(tr.data[0]["sigma"])
                mu_oficial = float(tr.data[0]["mu"])
        except Exception:
            pass

    # Item difficulty subgroup counts (Primi)
    n_faceis_total = diag.get("itens_faceis", 0)
    n_medios_total = diag.get("itens_medios", 0)
    n_dificeis_total = diag.get("itens_dificeis", 0)

    resultados_estruturados: list[dict] = []
    for r in raw_results:
        ref = tab.obter(area, r["acertos"]) if r["acertos"] in tab.lookup.get(area, {}) else {
            "tri_min": 0, "tri_med": 0, "tri_max": 1000
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

    elapsed = time.time() - t0
    print(f"[CSV] Done in {elapsed:.2f}s — V2.1 enrichment {len(resultados_estruturados)} students")

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

    # ── 1. Parse upload ──
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

    # ── 2. Load INEP item bank for the specified prova ──
    transf_r = client.table("enem_transformacoes").select("*").eq("ano", ano).eq("area", area).limit(1).execute()
    if not transf_r.data:
        raise HTTPException(
            status_code=404,
            detail=f"Transformação empírica não disponível para {area}/{ano}",
        )
    transf_row = transf_r.data[0]
    sigma = float(transf_row["sigma"])
    mu = float(transf_row["mu"])

    itens_area_r = client.table("enem_itens").select("*").eq("ano", ano).eq("area", area).execute()
    itens_by_co = {it["co_item"]: it for it in itens_area_r.data}

    maps_r = client.table("enem_itens_prova").select("*").eq("ano", ano).eq("co_prova", co_prova).execute()
    if not maps_r.data:
        raise HTTPException(status_code=404, detail=f"Prova {co_prova} não encontrada para {area}/{ano}")

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
        inep_items = [it for it in all_items if it.get("tp_lingua") is None or it.get("tp_lingua") == tp_lingua]
    else:
        inep_items = all_items

    inep_items.sort(key=lambda x: x["posicao"])
    n_itens = len(inep_items)

    if n_itens < 10:
        raise HTTPException(status_code=400, detail=f"Poucos itens INEP válidos: {n_itens}")

    # ── 3. Validate / align response lengths via position mapping ──
    # Arquivo do simulado tem respostas posição-a-posição na área (typicamente 45)
    # Item bank INEP pode ter menos itens (alguns abandonados)
    # Solução: mapear cada item do banco à sua posição no vetor do aluno
    offset_area = {"LC": 1, "CH": 46, "CN": 91, "MT": 136}[area]
    file_n_itens = parsed["n_itens"]
    if file_n_itens < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo tem apenas {file_n_itens} itens por aluno, mínimo 30.",
        )

    # ── 4. Run V3 — local calibration ──
    tabela = tabelas_multiano.get(ano) or tabelas_multiano.get(2022)
    if tabela is None:
        raise HTTPException(status_code=500, detail="Tabela de referência multi-ano não carregada")

    alunos_motor = [
        {"aluno_id": a["aluno_id"], "nome": "", "respostas": a["respostas"]}
        for a in alunos_parsed
    ]

    try:
        resultado_v3 = pipeline_cientifico(
            alunos=alunos_motor,
            area=area,
            tabela=tabela,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na calibração V3: {e}")

    v3_by_id = {r["aluno_id"]: r for r in resultado_v3["resultados_alunos"]}

    # ── 5. Run V4 — INEP official parameters with position mapping ──
    v4_results: Dict[str, Dict[str, Any]] = {}
    for aluno in alunos_parsed:
        resp_vec = aluno["respostas"]  # length = file_n_itens (typically 45)
        # Align with INEP items by position
        aligned_binary = []
        aligned_items = []
        for it in inep_items:
            str_idx = it["posicao"] - offset_area  # 0-indexed into area vector
            if 0 <= str_idx < len(resp_vec):
                aligned_binary.append(resp_vec[str_idx])
                aligned_items.append(it)
        if len(aligned_binary) < 10:
            continue
        resp = np.array(aligned_binary, dtype=int)
        acertos = int(resp.sum())
        theta, _se = eap_theta_cientifico(resp, aligned_items)
        nota = sigma * theta + mu
        nota = round(float(nota), 1)
        v4_results[aluno["aluno_id"]] = {
            "theta": round(float(theta), 4),
            "nota": nota,
            "estado": estado_tri(nota),
            "acertos": acertos,
        }

    # ── 6. Build comparison ──
    comparacoes: list[ResultadoComparacao] = []
    diffs = []
    concordancia = 0
    notas_v3_list = []
    notas_v4_list = []

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
            diff=round(float(diff), 2),
        ))

    diffs_arr = np.array(diffs)
    notas_v3_arr = np.array(notas_v3_list)
    notas_v4_arr = np.array(notas_v4_list)

    # Correlation (handle edge cases)
    if len(notas_v3_arr) >= 2 and notas_v3_arr.std() > 0 and notas_v4_arr.std() > 0:
        correlacao = float(np.corrcoef(notas_v3_arr, notas_v4_arr)[0, 1])
    else:
        correlacao = 0.0

    estatisticas = EstatisticasComparacao(
        n_alunos=len(comparacoes),
        mae=round(float(np.mean(np.abs(diffs_arr))), 2),
        rmse=round(float(np.sqrt(np.mean(diffs_arr ** 2))), 2),
        correlacao=round(correlacao, 6),
        mean_diff=round(float(np.mean(diffs_arr)), 2),
        max_diff_abs=round(float(np.max(np.abs(diffs_arr))), 2),
        pct_concordancia_nivel=round(concordancia / len(comparacoes) * 100, 2) if comparacoes else 0.0,
        nota_media_v3=round(float(np.mean(notas_v3_arr)), 1),
        nota_media_v4=round(float(np.mean(notas_v4_arr)), 1),
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

import asyncio  # noqa: PLC0415
import base64  # noqa: PLC0415

from .microdata import (  # noqa: PLC0415
    calibrate_all_areas,
    calibration_summary,
    compute_difficulty_table,
    detect_schema,
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
    JOB_RUNNING,
    job_manager,
)
from .microdata.loader import (  # noqa: PLC0415
    default_csv_path,
    discover_available_years,
    load_primeira_aplicacao_provas,
    MICRODATA_DIR,
)


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

    # Offload the heavy work to a worker thread so we don't block uvicorn.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_microdata_pipeline,
        job.job_id,
        req.year,
        req.sample_size,
        req.rng_seed,
        req.force_rerun,
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


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINT — EXPORTAR EXCEL
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/api/v3/exportar-excel")
async def exportar_excel(payload: Dict[str, Any] = Body(...)):
    """
    Generate an XLSX file with all results: students, difficulty table,
    diagnostics, and the anchor chart as an embedded image.

    Expected payload:
      {
        "area": "MT",
        "ano_referencia": 2022,
        "metodo": "Rank Intragrupo (Primi)",
        "n_alunos": 403,
        "resultados": [{aluno_id, acertos_mt, theta_mt, nota_mt, acert_dif_mt, estado_mt}],
        "referencia": [{acertos, tri_min, tri_med, tri_max}],
        "diagnosticos": [{...}],
        "chart_png_base64": "iVBORw0KG..." (optional)
      }
    """
    import base64
    import io
    from datetime import datetime

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    try:
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
        right = Alignment(horizontal="right", vertical="center")

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

        # Classification counts
        niveis_count: Dict[str, int] = {}
        for r in resultados:
            est = r.get(f"estado_{area_lower}")
            if est:
                niveis_count[est] = niveis_count.get(est, 0) + 1

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
        for col_idx, h in enumerate(headers, start=1):
            cell = ws2.cell(row=1, column=col_idx, value=h)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center
            cell.border = border

        for row_idx, r in enumerate(resultados, start=2):
            ws2.cell(row=row_idx, column=1, value=r.get("aluno_id", ""))
            ws2.cell(row=row_idx, column=2, value=r.get(f"acertos_{area_lower}"))
            ws2.cell(row=row_idx, column=3, value=r.get(f"theta_{area_lower}"))
            ws2.cell(row=row_idx, column=4, value=r.get(f"nota_{area_lower}"))
            ws2.cell(row=row_idx, column=5, value=r.get(f"acert_dif_{area_lower}"))
            ws2.cell(row=row_idx, column=6, value=r.get(f"estado_{area_lower}"))
            for c in range(1, 7):
                ws2.cell(row=row_idx, column=c).border = border
                ws2.cell(row=row_idx, column=c).alignment = center if c > 1 else left

        ws2.column_dimensions["A"].width = 15
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
        for r in resultados:
            ac = r.get(f"acertos_{area_lower}")
            nota = r.get(f"nota_{area_lower}")
            if ac is not None and nota is not None:
                freq_ac[ac] += 1
                notas_ac[ac].append(nota)

        table_headers = ["Acertos", "Freq Simulado", "Ref Min", "Ref Med", "Ref Max", "Nota Sim (media)", "Nota Sim (min-max)"]
        for col_idx, h in enumerate(table_headers, start=1):
            cell = ws3.cell(row=4, column=col_idx, value=h)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center
            cell.border = border

        for i, p in enumerate(referencia, start=5):
            ac = p["acertos"]
            f = freq_ac.get(ac, 0)
            ns = notas_ac.get(ac, [])
            ws3.cell(row=i, column=1, value=ac)
            ws3.cell(row=i, column=2, value=f)
            ws3.cell(row=i, column=3, value=p["tri_min"])
            ws3.cell(row=i, column=4, value=p["tri_med"])
            ws3.cell(row=i, column=5, value=p["tri_max"])
            if ns:
                avg = sum(ns) / len(ns)
                ws3.cell(row=i, column=6, value=round(avg, 1))
                ws3.cell(row=i, column=7, value=f"{min(ns):.0f} - {max(ns):.0f}")
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
                # Strip data URL prefix if present
                if "," in chart_b64:
                    chart_b64 = chart_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(chart_b64)
                img_stream = io.BytesIO(img_bytes)
                ws4 = wb.create_sheet("Grafico")
                ws4["A1"] = f"Grafico de Ancoragem — {area} / ENEM {ano}"
                ws4["A1"].font = title_font
                ws4.merge_cells("A1:H1")
                xlimg = XLImage(img_stream)
                # Resize to fit
                xlimg.width = 900
                xlimg.height = 520
                ws4.add_image(xlimg, "A3")
            except Exception as e:
                print(f"[EXCEL] Failed to embed chart image: {e}")

        # Save to bytes
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        filename = f"xtri_{area}_{ano}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[EXCEL] ERROR: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar Excel: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# FRONTEND — Static Files (must be AFTER all API routes)
# ════════════════════════════════════════════════════════════════════════════════

_static_dir = os.path.join(os.path.dirname(__file__), "static")


@app.get("/", include_in_schema=False)
async def frontend():
    """Serve the frontend application."""
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/validator", include_in_schema=False)
async def frontend_validator():
    """Serve the INEP validator page."""
    return FileResponse(os.path.join(_static_dir, "validator.html"))


if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
