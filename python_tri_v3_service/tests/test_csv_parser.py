"""Tests for the CSV parser module."""

import pytest

from app.csv_parser import parse_single_area


def _make_csv(n_alunos: int = 15, n_itens: int = 45, delimiter: str = ",") -> str:
    """Generate a synthetic single-area CSV."""
    headers = ["ID"] + [str(i) for i in range(1, n_itens + 1)]
    lines = [delimiter.join(headers)]
    for i in range(n_alunos):
        row = [f"A{i:03d}"] + [str((i + j) % 2) for j in range(n_itens)]
        lines.append(delimiter.join(row))
    return "\n".join(lines)


class TestParseSingleArea:

    @pytest.mark.unit
    def test_retorna_alunos_e_contagens(self):
        csv = _make_csv(n_alunos=20, n_itens=45)
        result = parse_single_area(csv)
        assert result["n_alunos"] == 20
        assert result["n_itens"] == 45
        assert len(result["alunos"]) == 20

    @pytest.mark.unit
    def test_respostas_corretas(self):
        csv = _make_csv(n_alunos=5, n_itens=10)
        result = parse_single_area(csv)
        for aluno in result["alunos"]:
            assert len(aluno["respostas"]) == 10
            assert all(v in (0, 1) for v in aluno["respostas"])

    @pytest.mark.unit
    def test_aluno_id_extraido(self):
        csv = _make_csv(n_alunos=3)
        result = parse_single_area(csv)
        assert result["alunos"][0]["aluno_id"] == "A000"
        assert result["alunos"][2]["aluno_id"] == "A002"

    @pytest.mark.unit
    def test_ignora_colunas_metadata(self):
        """Metadata columns like Acerto, TRI are skipped."""
        headers = "ID,1,2,3,Acerto,TRI 1,TRI 2,4,5"
        row = "X001,1,0,1,3,500.0,510.0,0,1"
        csv = headers + "\n" + row
        result = parse_single_area(csv)
        assert result["n_itens"] == 5  # only 1,2,3,4,5
        assert result["alunos"][0]["respostas"] == [1, 0, 1, 0, 1]

    @pytest.mark.unit
    def test_formato_questao_n(self):
        headers = ",".join(["ID"] + [f"Questao {i}" for i in range(1, 11)])
        row = ",".join(["A001"] + ["1"] * 10)
        csv = headers + "\n" + row
        result = parse_single_area(csv)
        assert result["n_itens"] == 10
        assert sum(result["alunos"][0]["respostas"]) == 10

    @pytest.mark.unit
    def test_delimitador_ponto_virgula(self):
        csv = _make_csv(n_alunos=10, delimiter=";")
        result = parse_single_area(csv)
        assert result["n_alunos"] == 10

    @pytest.mark.unit
    def test_valor_nao_binario_vira_zero_com_aviso(self):
        csv = "ID,1,2,3\nX001,1,A,0"
        result = parse_single_area(csv)
        assert result["alunos"][0]["respostas"] == [1, 0, 0]
        assert len(result["avisos"]) > 0
        assert "non-binary" in result["avisos"][0]

    @pytest.mark.unit
    def test_linhas_vazias_ignoradas(self):
        csv = _make_csv(n_alunos=5) + "\n\n\n"
        result = parse_single_area(csv)
        assert result["n_alunos"] == 5

    @pytest.mark.unit
    def test_sem_questoes_levanta_erro(self):
        csv = "ID,Nome,Turma\nA,Joao,3A"
        with pytest.raises(ValueError, match="No question columns"):
            parse_single_area(csv)

    @pytest.mark.unit
    def test_headers_com_numeros_grandes(self):
        """Question numbers like 91-135 should work."""
        headers = ",".join(["ID"] + [str(i) for i in range(91, 136)])
        row = ",".join(["A001"] + ["1"] * 45)
        csv = headers + "\n" + row
        result = parse_single_area(csv)
        assert result["n_itens"] == 45


class TestParseXLSX:

    @pytest.mark.unit
    def test_parse_xlsx_real_file(self):
        """Test with the actual example XLSX if available."""
        import os
        path = "/Users/home/Desktop/testes MIRT/dados-mat.xlsx"
        if not os.path.exists(path):
            pytest.skip("Example XLSX not available")
        with open(path, "rb") as f:
            raw = f.read()
        result = parse_single_area(raw, is_xlsx=True)
        assert result["n_alunos"] == 403
        assert result["n_itens"] == 45
        assert len(result["avisos"]) == 0
