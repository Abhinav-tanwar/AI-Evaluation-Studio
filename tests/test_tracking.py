import pytest

from studio.evaluation import run_comparison
from studio.schemas import RunRequest
from studio.tracking import export_mlflow


def test_mlflow_export_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = (tmp_path / "tracking").as_uri()
    result = run_comparison(RunRequest())
    run_id = export_mlflow(result, uri)
    client = MlflowClient(uri)
    saved = client.get_run(run_id)
    assert (
        saved.data.metrics["candidate.exact_match"]
        == result["pipelines"]["candidate"]["metrics"]["exact_match"]
    )
    assert saved.data.tags["dataset_sha256"] == result["dataset_sha256"]
    assert any(a.path == "evaluation.json" for a in client.list_artifacts(run_id))
