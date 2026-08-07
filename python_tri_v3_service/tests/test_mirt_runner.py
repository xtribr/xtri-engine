"""Tests do runner Python do worker MIRT."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from app.mirt.qmatrix import build_habilidade_qmatrix, build_model_spec_from_qmatrix
from app.mirt.runner import (
    MIRTWorkerExecutionError,
    build_mirt_output_dir,
    prepare_mirt_fit_input,
    run_mirt_fit,
    run_mirt_score,
)


def _make_qmatrix_and_model():
    qmatrix = build_habilidade_qmatrix(
        area="MT",
        item_rows=[
            {"item_id": "item_1", "co_item": 1001, "numero_habilidade": 1, "posicao": 1},
            {"item_id": "item_2", "co_item": 1002, "numero_habilidade": 2, "posicao": 2},
        ],
    )
    return qmatrix, build_model_spec_from_qmatrix(qmatrix)


class TestMIRTRunner:
    @pytest.mark.unit
    def test_build_mirt_output_dir(self, tmp_path):
        output_dir = build_mirt_output_dir(
            year=2024,
            area="mt",
            model_id="mirt_bifactor_mt_v1",
            job_id="job_001",
            base_dir=tmp_path,
        )

        assert output_dir == tmp_path / "2024" / "MT" / "mirt_bifactor_mt_v1" / "job_001"
        assert output_dir.is_dir()

    @pytest.mark.unit
    def test_prepare_mirt_fit_input_persists_payload(self, tmp_path):
        qmatrix, model = _make_qmatrix_and_model()
        response_df = pd.DataFrame(
            {
                "student_id": ["a1", "a2"],
                "item_1": [1, 0],
                "item_2": [0, 1],
            }
        )

        worker_input = prepare_mirt_fit_input(
            response_df=response_df,
            qmatrix=qmatrix,
            model_spec=model,
            year=2024,
            area="MT",
            job_id="job_fit_001",
            base_dir=tmp_path,
        )

        output_dir = Path(worker_input.output_dir)
        assert (output_dir / "response_matrix.csv").is_file()
        assert (output_dir / "qmatrix.json").is_file()
        assert (output_dir / "qmatrix_table.csv").is_file()
        assert (output_dir / "worker_input.json").is_file()

        payload = json.loads((output_dir / "worker_input.json").read_text(encoding="utf-8"))
        assert payload["model_spec"]["model_id"] == model.model_id
        assert payload["metadata"]["item_ids"] == ["item_1", "item_2"]

    @pytest.mark.unit
    def test_prepare_mirt_fit_input_rejects_item_mismatch(self, tmp_path):
        qmatrix, model = _make_qmatrix_and_model()
        response_df = pd.DataFrame(
            {
                "student_id": ["a1"],
                "item_1": [1],
                "item_3": [0],
            }
        )

        with pytest.raises(Exception, match="mesmo conjunto de itens"):
            prepare_mirt_fit_input(
                response_df=response_df,
                qmatrix=qmatrix,
                model_spec=model,
                year=2024,
                area="MT",
                job_id="job_fit_002",
                base_dir=tmp_path,
            )

    @pytest.mark.unit
    def test_prepare_mirt_fit_input_preserves_missing_as_na(self, tmp_path):
        qmatrix, model = _make_qmatrix_and_model()
        response_df = pd.DataFrame(
            {
                "student_id": ["a1", "a2"],
                "item_1": [1, 0],
                "item_2": [None, 1],
            }
        )

        worker_input = prepare_mirt_fit_input(
            response_df=response_df,
            qmatrix=qmatrix,
            model_spec=model,
            year=2024,
            area="MT",
            job_id="job_fit_missing",
            base_dir=tmp_path,
        )

        saved = pd.read_csv(worker_input.response_matrix_path)
        assert pd.isna(saved.loc[0, "item_2"])
        assert saved.loc[0, "item_1"] == 1

    @pytest.mark.unit
    def test_run_mirt_fit_reads_worker_output(self, tmp_path):
        qmatrix, model = _make_qmatrix_and_model()
        response_df = pd.DataFrame(
            {
                "student_id": ["a1", "a2"],
                "item_1": [1, 0],
                "item_2": [0, 1],
            }
        )
        worker_input = prepare_mirt_fit_input(
            response_df=response_df,
            qmatrix=qmatrix,
            model_spec=model,
            year=2024,
            area="MT",
            job_id="job_fit_003",
            base_dir=tmp_path,
        )
        output_dir = Path(worker_input.output_dir)
        worker_output_path = output_dir / "worker_output.json"
        worker_output_path.write_text(
            json.dumps(
                {
                    "status": "done",
                    "artifact_dir": str(output_dir),
                    "n_students": 2,
                    "n_items": 2,
                    "model_spec": model.model_dump(mode="json"),
                    "artifact_files": {
                        "model_rds_path": str(output_dir / "model.rds"),
                        "item_params_path": str(output_dir / "item_params.json"),
                        "factor_scores_path": str(output_dir / "factor_scores.csv"),
                        "fit_summary_path": str(output_dir / "fit_summary.json"),
                        "diagnostics_path": str(output_dir / "diagnostics.json"),
                        "qmatrix_snapshot_path": str(output_dir / "qmatrix_snapshot.json"),
                    },
                    "warnings": [],
                    "error": None,
                }
            ),
            encoding="utf-8",
        )

        with mock.patch("app.mirt.runner.mirt_runtime_available", return_value=True), mock.patch(
            "app.mirt.runner._run_subprocess",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            result = run_mirt_fit(worker_input)

        assert result.status == "done"
        assert result.artifact_files is not None
        assert result.artifact_files.model_rds_path.endswith("model.rds")

    @pytest.mark.unit
    def test_run_mirt_fit_raises_when_rscript_fails(self, tmp_path):
        qmatrix, model = _make_qmatrix_and_model()
        response_df = pd.DataFrame(
            {
                "student_id": ["a1"],
                "item_1": [1],
                "item_2": [0],
            }
        )
        worker_input = prepare_mirt_fit_input(
            response_df=response_df,
            qmatrix=qmatrix,
            model_spec=model,
            year=2024,
            area="MT",
            job_id="job_fit_004",
            base_dir=tmp_path,
        )

        with mock.patch("app.mirt.runner.mirt_runtime_available", return_value=True), mock.patch(
            "app.mirt.runner._run_subprocess",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="erro no worker"),
        ):
            with pytest.raises(MIRTWorkerExecutionError, match="erro no worker"):
                run_mirt_fit(worker_input)

    @pytest.mark.unit
    def test_run_mirt_score_raises_when_output_missing(self, tmp_path):
        model_path = tmp_path / "model.rds"
        response_path = tmp_path / "responses.csv"
        output_path = tmp_path / "scores.csv"
        model_path.write_text("placeholder", encoding="utf-8")
        response_path.write_text("student_id,item_1\ns1,1\n", encoding="utf-8")

        with mock.patch("app.mirt.runner.mirt_runtime_available", return_value=True), mock.patch(
            "app.mirt.runner._run_subprocess",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            with pytest.raises(MIRTWorkerExecutionError, match="não produziu o arquivo"):
                run_mirt_score(
                    model_rds_path=model_path,
                    response_matrix_path=response_path,
                    output_path=output_path,
                )

    @pytest.mark.unit
    def test_run_mirt_score_normaliza_falha_ao_iniciar_rscript(self, tmp_path):
        model_path = tmp_path / "model.rds"
        response_path = tmp_path / "responses.csv"
        output_path = tmp_path / "scores.csv"
        model_path.write_text("placeholder", encoding="utf-8")
        response_path.write_text("student_id,item_1\ns1,1\n", encoding="utf-8")

        with mock.patch(
            "app.mirt.runner.mirt_runtime_available", return_value=True
        ), mock.patch(
            "app.mirt.runner.subprocess.run",
            side_effect=PermissionError("execução negada"),
        ):
            with pytest.raises(MIRTWorkerExecutionError, match="iniciar o worker"):
                run_mirt_score(
                    model_rds_path=model_path,
                    response_matrix_path=response_path,
                    output_path=output_path,
                )
