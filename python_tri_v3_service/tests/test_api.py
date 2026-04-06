"""API endpoint tests (regression baseline for V2/V3 routes + new frontend endpoints)."""

import io
import pytest
from fastapi.testclient import TestClient


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Debug
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoints:

    @pytest.mark.api
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "online"
        assert "heuristico_v2" in body["motores"]
        assert "cientifico_mirt" in body["motores"]

    @pytest.mark.api
    def test_debug(self, client: TestClient):
        resp = client.get("/api/debug")
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == "3.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# V2 Heuristic Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcularTRI:

    @pytest.mark.api
    def test_payload_valido(self, client: TestClient, alunos_v2_30, gabarito_45):
        resp = client.post("/api/calcular-tri", json={
            "alunos": alunos_v2_30,
            "gabarito": gabarito_45,
            "areas_config": {"MT": [1, 45]},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["total_alunos"] == 30
        assert len(body["resultados"]) == 30

    @pytest.mark.api
    def test_payload_sem_alunos(self, client: TestClient):
        resp = client.post("/api/calcular-tri", json={
            "gabarito": {"1": "A"},
        })
        assert resp.status_code == 400

    @pytest.mark.api
    def test_payload_vazio(self, client: TestClient):
        resp = client.post("/api/calcular-tri", json={})
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# V3 Scientific Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrarCientifico:

    @pytest.mark.api
    def test_payload_valido(self, client: TestClient, respostas_binarias_50x45):
        alunos = [
            {"aluno_id": str(i), "nome": f"Aluno {i}", "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        resp = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "test_sim",
            "area": "MT",
            "alunos": alunos,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["n_alunos"] == 50
        assert body["n_itens"] == 45

    @pytest.mark.api
    def test_area_invalida(self, client: TestClient, respostas_binarias_50x45):
        alunos = [
            {"aluno_id": str(i), "respostas": r}
            for i, r in enumerate(respostas_binarias_50x45)
        ]
        resp = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "test_sim",
            "area": "XX",
            "alunos": alunos,
        })
        assert resp.status_code == 400

    @pytest.mark.api
    def test_menos_de_10_alunos(self, client: TestClient):
        alunos = [
            {"aluno_id": str(i), "respostas": [1, 0, 1] * 5}
            for i in range(5)
        ]
        resp = client.post("/api/v3/calibrar-cientifico", json={
            "simulado_id": "test_sim",
            "area": "MT",
            "alunos": alunos,
        })
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# V3 Individual Theta Estimation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimarTheta:

    @pytest.mark.api
    def test_payload_valido(self, client: TestClient):
        resp = client.post("/api/v3/estimar-theta", json={
            "area": "MT",
            "parametros_itens": [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10,
            "respostas": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "theta" in body
        assert "nota_ancorada" in body
        assert "estado_tri" in body

    @pytest.mark.api
    def test_respostas_e_itens_tamanho_diferente(self, client: TestClient):
        resp = client.post("/api/v3/estimar-theta", json={
            "area": "MT",
            "parametros_itens": [{"a": 1.0, "b": 0.0, "c": 0.20}] * 10,
            "respostas": [1, 0, 1],  # 3 != 10
        })
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# V3 Reference Table Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestReferencias:

    @pytest.mark.api
    def test_area_valida(self, client: TestClient):
        resp = client.get("/api/v3/referencias/MT")
        assert resp.status_code == 200
        body = resp.json()
        assert body["area"] == "MT"
        assert body["total_registros"] == 46
        assert len(body["dados"]) == 46

    @pytest.mark.api
    def test_area_invalida(self, client: TestClient):
        resp = client.get("/api/v3/referencias/XX")
        assert resp.status_code == 404

    @pytest.mark.api
    def test_area_case_insensitive(self, client: TestClient):
        resp = client.get("/api/v3/referencias/mt")
        assert resp.status_code == 200
        assert resp.json()["area"] == "MT"


# ═══════════════════════════════════════════════════════════════════════════════
# Frontend Endpoints — Years + CSV Processing
# ═══════════════════════════════════════════════════════════════════════════════

def _make_single_area_csv(n: int = 30, n_itens: int = 45) -> str:
    """Synthetic single-area CSV with binary responses."""
    import numpy as np
    rng = np.random.default_rng(42)
    headers = ["ID"] + [str(i) for i in range(1, n_itens + 1)]
    lines = [",".join(headers)]
    for i in range(n):
        row = [f"S{i:03d}"] + [str(int(rng.random() > 0.5)) for _ in range(n_itens)]
        lines.append(",".join(row))
    return "\n".join(lines)


class TestAnosReferencia:

    @pytest.mark.api
    def test_retorna_lista_de_anos(self, client: TestClient):
        resp = client.get("/api/v3/anos-referencia")
        assert resp.status_code == 200
        anos = resp.json()
        assert isinstance(anos, list)
        assert len(anos) >= 1

    @pytest.mark.api
    def test_anos_sao_inteiros_ordenados(self, client: TestClient):
        resp = client.get("/api/v3/anos-referencia")
        anos = resp.json()
        assert anos == sorted(anos)
        assert all(isinstance(a, int) for a in anos)


class TestProcessarCSV:

    @pytest.mark.api
    def test_upload_area_mt(self, client: TestClient):
        csv_content = _make_single_area_csv(30)
        anos = client.get("/api/v3/anos-referencia").json()
        ano = anos[-1]

        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": ("mt.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "sucesso"
        assert body["areas_processadas"] == ["MT"]
        assert body["n_alunos"] == 30
        r0 = body["resultados"][0]
        assert r0["nota_mt"] is not None
        assert r0["estado_mt"] is not None
        # Other areas should be null
        assert r0["nota_lc"] is None

    @pytest.mark.api
    def test_area_invalida_retorna_400(self, client: TestClient):
        csv_content = _make_single_area_csv(15)
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "XX", "ano_referencia": "2022"},
            files={"arquivo": ("test.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 400

    @pytest.mark.api
    def test_ano_invalido_retorna_400(self, client: TestClient):
        csv_content = _make_single_area_csv(15)
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": "1999"},
            files={"arquivo": ("test.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 400

    @pytest.mark.api
    def test_csv_malformado_retorna_400(self, client: TestClient):
        csv_bad = "ID,Nome,Turma\nA,Joao,3A"
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": "2022"},
            files={"arquivo": ("bad.csv", csv_bad, "text/csv")},
        )
        assert resp.status_code == 400

    @pytest.mark.api
    def test_menos_de_10_alunos_retorna_400(self, client: TestClient):
        csv_content = _make_single_area_csv(5)
        resp = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": "2022"},
            files={"arquivo": ("small.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 400


class TestFrontendStatic:

    @pytest.mark.api
    def test_root_retorna_html(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "XTRI Engine" in resp.text


class TestExportarExcel:

    @pytest.mark.api
    def test_exportar_excel_retorna_xlsx(self, client: TestClient):
        # First process a CSV to get real data
        csv_content = _make_single_area_csv(30)
        anos = client.get("/api/v3/anos-referencia").json()
        ano = anos[-1]
        proc = client.post(
            "/api/v3/processar-csv",
            data={"area": "MT", "ano_referencia": str(ano)},
            files={"arquivo": ("mt.csv", csv_content, "text/csv")},
        )
        data = proc.json()

        payload = {
            "area": "MT",
            "ano_referencia": ano,
            "metodo": "Rank Intragrupo (Primi)",
            "n_alunos": data["n_alunos"],
            "resultados": data["resultados"],
            "referencia": data["referencia"],
            "diagnosticos": data["diagnosticos"],
        }

        resp = client.post("/api/v3/exportar-excel", json=payload)
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]
        assert "attachment" in resp.headers["content-disposition"]
        assert len(resp.content) > 1000  # non-trivial xlsx file

    @pytest.mark.api
    def test_exportar_excel_com_grafico_base64(self, client: TestClient):
        # Minimal valid 1x1 PNG in base64
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII="
        payload = {
            "area": "MT",
            "ano_referencia": 2022,
            "metodo": "Test",
            "n_alunos": 2,
            "resultados": [
                {"aluno_id": "A1", "acertos_mt": 20, "theta_mt": 0.5, "nota_mt": 600.0, "acert_dif_mt": 0.3, "estado_mt": "PROFICIENTE"},
                {"aluno_id": "A2", "acertos_mt": 30, "theta_mt": 1.5, "nota_mt": 750.0, "acert_dif_mt": 0.7, "estado_mt": "AVANCADO"},
            ],
            "referencia": [
                {"acertos": 20, "tri_min": 500.0, "tri_med": 600.0, "tri_max": 700.0},
                {"acertos": 30, "tri_min": 650.0, "tri_med": 750.0, "tri_max": 850.0},
            ],
            "diagnosticos": [{
                "area": "MT", "n_itens": 45, "n_alunos": 2, "b_medio": 0.0,
                "b_desvio": 1.0, "nota_media": 675.0, "nota_min": 600.0,
                "nota_max": 750.0, "pct_dentro_intervalo": 100.0,
                "itens_faceis": 15, "itens_medios": 20, "itens_dificeis": 10,
                "acert_dif_medio": 0.5,
            }],
            "chart_png_base64": png_b64,
        }

        resp = client.post("/api/v3/exportar-excel", json=payload)
        assert resp.status_code == 200
        assert len(resp.content) > 1000
