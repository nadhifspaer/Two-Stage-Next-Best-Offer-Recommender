# Pinned production model identity, read from the models manifest and never resolved as the latest MLflow run;
# the backtest ranker shares the stage4_ranking experiment, so a latest-run lookup could silently serve it.
import json
from dataclasses import dataclass
from pathlib import Path

MODELS_MANIFEST_PATH = Path("data/processed/models/_manifest.json")


@dataclass
class ProductionModels:
    parent_run_id: str
    ranker_run_id: str
    classifier_run_id: str
    calibrator_path: Path


def load_production_models(manifest_path: Path = MODELS_MANIFEST_PATH) -> ProductionModels:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return ProductionModels(
        parent_run_id=manifest["mlflow_run_id"],
        ranker_run_id=manifest["ranker_run_id"],
        classifier_run_id=manifest["classifier_run_id"],
        calibrator_path=Path(manifest["calibrator_path"]),
    )
