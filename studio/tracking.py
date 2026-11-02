from pathlib import Path
from tempfile import TemporaryDirectory


def export_mlflow(result, uri):
    if not uri:
        raise ValueError("Set MLFLOW_TRACKING_URI to enable tracking export.")
    import json

    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("AI Evaluation Studio")
    with mlflow.start_run(run_name=result["name"]) as run:
        mlflow.set_tags(
            {
                "dataset_sha256": result["dataset_sha256"],
                "revision": result["revision"],
                "provider": result["provider"],
                "split": result["split"],
            }
        )
        mlflow.log_params({"scorer_version": result["scorer_version"], "dataset": result["dataset"]})
        for label, pipeline in result["pipelines"].items():
            mlflow.log_params({f"{label}.{k}": v for k, v in result["config"][label].items()})
            mlflow.log_metrics({f"{label}.{k}": v for k, v in pipeline["metrics"].items() if v is not None})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps(result, indent=2))
            mlflow.log_artifact(str(path))
        return run.info.run_id
