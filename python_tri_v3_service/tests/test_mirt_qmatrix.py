"""Tests estruturais do contrato MIRT/Q-matrix.

Os exemplos abaixo são templates sintéticos de metadados de item, usados apenas
para validar o formato do código. Não representam dados educacionais reais.
"""

import pytest

from app.mirt.contracts import MIRTWorkerInput, MIRTWorkerOutput
from app.mirt.qmatrix import (
    DEFAULT_GENERAL_FACTOR_ID,
    build_empty_qmatrix_template,
    build_habilidade_qmatrix,
    build_model_spec_from_qmatrix,
    qmatrix_to_binary_rows,
)


class TestQMatrixTemplate:
    @pytest.mark.unit
    def test_build_empty_qmatrix_template(self):
        qmatrix = build_empty_qmatrix_template(
            area="mt",
            item_ids=["item_1", "item_2"],
        )

        assert qmatrix.area == "MT"
        assert qmatrix.is_template is True
        assert qmatrix.source == "template_vazio"
        assert len(qmatrix.rows) == 2
        assert qmatrix.rows[0].cargas == {}


class TestQMatrixFromItemBank:
    @pytest.mark.unit
    def test_build_habilidade_qmatrix(self):
        qmatrix = build_habilidade_qmatrix(
            area="MT",
            item_rows=[
                {
                    "item_id": "mt_001",
                    "co_item": 101,
                    "numero_habilidade": 7,
                    "co_prova": 999,
                    "posicao": 1,
                },
                {
                    "item_id": "mt_002",
                    "co_item": 102,
                    "numero_habilidade": 12,
                    "co_prova": 999,
                    "posicao": 2,
                },
            ],
        )

        factor_ids = [factor.factor_id for factor in qmatrix.factors]
        assert qmatrix.is_template is False
        assert DEFAULT_GENERAL_FACTOR_ID in factor_ids
        assert "MT_H07" in factor_ids
        assert "MT_H12" in factor_ids
        assert qmatrix.rows[0].cargas[DEFAULT_GENERAL_FACTOR_ID] == 1
        assert qmatrix.rows[0].cargas["MT_H07"] == 1

    @pytest.mark.unit
    def test_fail_when_item_bank_has_conflicting_habilidade(self):
        with pytest.raises(ValueError, match="habilidades conflitantes"):
            build_habilidade_qmatrix(
                area="CN",
                item_rows=[
                    {
                        "item_id": "cn_001",
                        "co_item": 201,
                        "numero_habilidade": 3,
                    },
                    {
                        "item_id": "cn_001",
                        "co_item": 201,
                        "numero_habilidade": 4,
                    },
                ],
            )


class TestMIRTContracts:
    @pytest.mark.unit
    def test_build_model_spec_from_qmatrix(self):
        qmatrix = build_habilidade_qmatrix(
            area="CH",
            item_rows=[
                {"item_id": "ch_001", "co_item": 301, "numero_habilidade": 2},
            ],
        )

        model = build_model_spec_from_qmatrix(qmatrix)

        assert model.family == "bifactor"
        assert model.initial_structure == "bifactor_habilidade"
        assert model.scoring_factor_id == DEFAULT_GENERAL_FACTOR_ID
        assert model.metadata["n_itens"] == 1

    @pytest.mark.unit
    def test_qmatrix_to_binary_rows(self):
        qmatrix = build_habilidade_qmatrix(
            area="LC",
            item_rows=[
                {"item_id": "lc_001", "co_item": 401, "numero_habilidade": 5},
            ],
        )

        rows = qmatrix_to_binary_rows(qmatrix)

        assert len(rows) == 1
        assert rows[0]["item_id"] == "lc_001"
        assert rows[0]["G"] == 1
        assert rows[0]["LC_H05"] == 1

    @pytest.mark.unit
    def test_worker_input_output_contracts(self):
        qmatrix = build_habilidade_qmatrix(
            area="MT",
            item_rows=[
                {"item_id": "mt_900", "co_item": 900, "numero_habilidade": 1},
            ],
        )
        model = build_model_spec_from_qmatrix(qmatrix)

        payload = MIRTWorkerInput(
            job_id="job_mirt_001",
            year=2024,
            area="MT",
            response_matrix_path="/tmp/resp.parquet",
            qmatrix_path="/tmp/qmatrix.json",
            output_dir="/tmp/mirt_job",
            model_spec=model,
            sample_size=1000,
        )
        result = MIRTWorkerOutput(
            status="done",
            artifact_dir="/tmp/mirt_job/output",
            n_students=1000,
            n_items=1,
            model_spec=model,
        )

        assert payload.model_spec.model_id.startswith("mirt_bifactor_mt_")
        assert result.status == "done"
        assert result.n_students == 1000
