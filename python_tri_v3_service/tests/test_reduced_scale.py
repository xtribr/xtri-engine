"""Testes da API v6 para TRI em escalas reduzidas."""

import io

import pytest
from openpyxl import Workbook

from app.irt.reduced_scale import (
    ancorar_nota_reduzida,
    obter_faixa_referencia_continua,
    precisao_por_tamanho,
)


def _alunos(n_alunos: int, n_itens: int) -> list[dict]:
    alunos = []
    for i in range(n_alunos):
        respostas = [1 if ((i + j) % 3 != 0) else 0 for j in range(n_itens)]
        alunos.append({"aluno_id": f"S{i:03d}", "respostas": respostas})
    return alunos


def _xlsx_bytes(n_alunos: int, n_itens: int) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.append(["ID"] + [str(i + 1) for i in range(n_itens)])
    for aluno in _alunos(n_alunos, n_itens):
        ws.append([aluno["aluno_id"]] + aluno["respostas"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestReducedScaleCore:

    @pytest.mark.unit
    def test_referencia_continua_interpola_entre_acertos_inteiros(self, tabela):
        ref_10 = tabela.obter("MT", 10)
        ref_11 = tabela.obter("MT", 11)

        ref = obter_faixa_referencia_continua(
            area="MT",
            acertos_equivalentes_45=10.5,
            tabela=tabela,
        )

        assert ref["acertos_base_inferior"] == 10
        assert ref["acertos_base_superior"] == 11
        assert ref["tri_med"] == pytest.approx((ref_10["tri_med"] + ref_11["tri_med"]) / 2)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "acertos,n_itens,esperado",
        [(10, 10, 45.0), (15, 30, 22.5), (30, 45, 30.0)],
    )
    def test_acertos_equivalentes_45(self, tabela, acertos, n_itens, esperado):
        result = ancorar_nota_reduzida(
            theta=0.0,
            se_theta=0.35,
            area="MT",
            acertos_observados=acertos,
            n_itens_validos=n_itens,
            tabela=tabela,
        )

        assert result["acertos_equivalentes_45"] == pytest.approx(esperado)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n_itens,precisao",
        [(10, "BAIXA"), (19, "BAIXA"), (20, "MEDIA"), (29, "MEDIA"), (30, "ALTA")],
    )
    def test_precisao_por_tamanho(self, n_itens, precisao):
        assert precisao_por_tamanho(n_itens) == precisao


class TestReducedScaleAPI:

    @pytest.mark.api
    def test_json_10_itens_retorna_baixa_precisao(self, client):
        payload = {
            "simulado_id": "reduzido-10",
            "area": "MT",
            "ano_referencia": 2024,
            "alunos": _alunos(20, 10),
        }

        resp = client.post("/api/v6/tri/reduzida", json=payload)

        assert resp.status_code == 200, resp.text[:500]
        body = resp.json()
        assert body["modo_scoring"] == "enem_projected_reduced"
        assert body["modo_escala"] == "reduzida"
        assert body["n_itens_validos"] == 10
        assert body["resultados"][0]["precisao"] == "BAIXA"
        assert body["resultados"][0]["intervalo_95"]["limite_inferior"] <= body["resultados"][0]["nota_enem_projetada"]
        assert "projeção" in body["limites_uso"].lower()

    @pytest.mark.api
    def test_json_30_itens_retorna_alta_precisao(self, client):
        payload = {
            "simulado_id": "reduzido-30",
            "area": "CN",
            "ano_referencia": 2024,
            "alunos": _alunos(25, 30),
        }

        resp = client.post("/api/v6/tri/reduzida", json=payload)

        assert resp.status_code == 200, resp.text[:500]
        body = resp.json()
        assert body["n_itens_validos"] == 30
        assert body["resultados"][0]["precisao"] == "ALTA"

    @pytest.mark.api
    def test_menos_de_10_itens_retorna_400(self, client):
        payload = {
            "simulado_id": "curto-demais",
            "area": "MT",
            "ano_referencia": 2024,
            "alunos": _alunos(20, 9),
        }

        resp = client.post("/api/v6/tri/reduzida", json=payload)

        assert resp.status_code == 400
        assert "mínimo 10" in resp.text.lower()

    @pytest.mark.api
    def test_item_anulado_reduz_denominador(self, client):
        alunos = _alunos(20, 11)
        payload = {
            "simulado_id": "reduzido-anulado",
            "area": "MT",
            "ano_referencia": 2024,
            "itens_anulados": [3],
            "alunos": alunos,
        }

        resp = client.post("/api/v6/tri/reduzida", json=payload)

        assert resp.status_code == 200, resp.text[:500]
        body = resp.json()
        assert body["n_itens_validos"] == 10
        assert any("3" in aviso for aviso in body["avisos"])

    @pytest.mark.api
    @pytest.mark.parametrize("n_itens", [10, 30, 45])
    def test_upload_xlsx_funciona_para_tamanhos_validos(self, client, n_itens):
        content = _xlsx_bytes(20, n_itens)

        resp = client.post(
            "/api/v6/tri/reduzida/upload",
            data={"simulado_id": f"upload-{n_itens}", "area": "MT", "ano_referencia": "2024"},
            files={
                "arquivo": (
                    "reduzida.xlsx",
                    content,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

        assert resp.status_code == 200, resp.text[:500]
        assert resp.json()["n_itens_validos"] == n_itens
