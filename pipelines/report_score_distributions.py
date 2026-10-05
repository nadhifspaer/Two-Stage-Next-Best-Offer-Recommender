# Ranker score and classifier probability distributions by label; re-runnable against the pinned models.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import os
import time
from pathlib import Path

import mlflow.lightgbm
import polars as pl

import src.production_models as production_models
import src.rank as rank
import src.tracking as tracking

TRAIN_GLOB = "data/processed/features/train/window-*.parquet"
MODELS_DIR = Path("data/processed/models")
DISTRIBUTIONS_PATH = MODELS_DIR / "_score_distributions.json"


def write(ranker, classifier, train_df: pl.DataFrame, categorical_mapping: dict, pin: dict, path: Path = DISTRIBUTIONS_PATH) -> dict:
    """Computes the distributions and writes them atomically. pin: ranker_run_id, classifier_run_id."""
    report = {**pin, **rank.score_distributions(ranker, classifier, train_df, categorical_mapping)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return report


def run() -> dict:
    t0 = time.time()
    tracking.configure_mlflow()
    production = production_models.load_production_models()
    ranker = mlflow.lightgbm.load_model(f"runs:/{production.ranker_run_id}/ranker")
    classifier = mlflow.lightgbm.load_model(f"runs:/{production.classifier_run_id}/classifier")
    mapping = rank.load_categorical_mapping(MODELS_DIR / "categorical_mapping.json")
    train_df = pl.scan_parquet(TRAIN_GLOB).collect()
    pin = {"ranker_run_id": production.ranker_run_id, "classifier_run_id": production.classifier_run_id}
    report = write(ranker, classifier, train_df, mapping, pin)
    return {"rows": report["rows"], "positives": report["positives"], "wall_time_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
