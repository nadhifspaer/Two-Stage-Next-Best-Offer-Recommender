# central MLflow tracking configuration
"""The only place the MLflow tracking URI and artifact root are set; every module imports configure_mlflow() from here.
Local SQLite tracking store plus a filesystem artifact root, no server (MLflow 3.x deprecated the plain file-store backend)."""
from pathlib import Path

import mlflow

MLRUNS_DIR = Path("mlruns")
TRACKING_URI = f"sqlite:///{MLRUNS_DIR.as_posix()}/mlflow.db"
DEFAULT_ARTIFACT_ROOT = (MLRUNS_DIR / "artifacts").resolve().as_uri()


def configure_mlflow() -> None:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)


def ensure_experiment(name: str, artifact_location: str = DEFAULT_ARTIFACT_ROOT) -> None:
    """Creates the experiment with a local artifact root if absent, then makes it active. configure_mlflow() must be called first."""
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)


def find_child_run_id(client: "mlflow.tracking.MlflowClient", parent_run_id: str, child_run_name: str) -> str:
    """Looks up a nested run's id by run name under a parent run, e.g. "stage4_1_classifier". configure_mlflow() must be called first."""
    parent = client.get_run(parent_run_id)
    return next(
        c.info.run_id
        for c in client.search_runs(
            experiment_ids=[parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        )
        if c.data.tags.get("mlflow.runName") == child_run_name
    )
