"""Regressões para validação em massa contra microdados INEP."""

import pytest

from app.scripts.validacao_massa import load_item_bank_and_transformacao


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters = []
        self._limit = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters.append((key, value))
        return self

    def limit(self, value):
        self._limit = value
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        rows = self._rows
        for key, value in self._filters:
            rows = [row for row in rows if row.get(key) == value]
        if hasattr(self, "_range"):
            start, end = self._range
            rows = rows[start : end + 1]
        if self._limit is not None:
            rows = rows[: self._limit]
        return _FakeResult(rows)


class _FakeClient:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return _FakeQuery(self._tables[name])


@pytest.mark.unit
def test_validacao_massa_carrega_whitelist_primeira_aplicacao():
    """A validação real não pode incluir reaplicação por acidente."""
    client = _FakeClient(
        {
            "enem_itens": [
                {
                    "ano": 2024,
                    "area": "MT",
                    "co_item": 10,
                    "param_discriminacao": 1.2,
                    "param_dificuldade": 0.3,
                    "param_acerto_casual": 0.2,
                    "gabarito": "A",
                    "tipo_lingua": None,
                    "item_abandonado": False,
                }
            ],
            "enem_itens_prova": [
                {"ano": 2024, "co_prova": 1001, "co_item": 10, "posicao": 136},
                {"ano": 2024, "co_prova": 9001, "co_item": 10, "posicao": 136},
            ],
            "enem_codigos_prova": [
                {"ano": 2024, "area": "MT", "co_prova": 1001, "is_reaplicacao": False},
                {"ano": 2024, "area": "MT", "co_prova": 9001, "is_reaplicacao": True},
            ],
            "enem_transformacoes": [
                {
                    "ano": 2024,
                    "area": "MT",
                    "sigma": 120.0,
                    "mu": 500.0,
                    "n_alunos": 5000,
                }
            ],
        }
    )

    _item_bank, _transf, provas_primeira = load_item_bank_and_transformacao(
        client, 2024, "MT"
    )

    assert provas_primeira == {1001}
