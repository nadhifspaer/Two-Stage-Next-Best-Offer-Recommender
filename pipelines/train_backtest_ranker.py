# Retrains the ranker on the first three training windows at a fixed iteration count, no early stopping.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import threading
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import polars as pl
import psutil

import src.rank as rank
import src.tracking as tracking

TRAIN_DIR = Path("data/processed/features/train")
BACKTEST_TRAIN_WINDOWS = ["2020-08-19", "2020-08-26", "2020-09-02"]
# 2020-09-09 is the evaluation week: no early stopping and no category vocabulary from it in the fit
FIXED_ITERATIONS = 212
MODELS_DIR = Path("data/processed/models")
BACKTEST_MAPPING_PATH = MODELS_DIR / "backtest_categorical_mapping.json"
MANIFEST_PATH = Path("data/processed/evaluation/_backtest_train_manifest.json")


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def run() -> dict:
    t0 = time.time()
    peak = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(peak, stop_flag), daemon=True)
    sampler.start()

    pl.enable_string_cache()
    train_df = pl.concat(
        [pl.read_parquet(TRAIN_DIR / f"window-{w}.parquet") for w in BACKTEST_TRAIN_WINDOWS]
    )
    categorical_mapping = rank.fit_categorical_mapping(train_df)
    fit_df = rank.prepare_matrix(train_df, categorical_mapping)

    params = {**rank.LGBM_PARAMS, "n_estimators": FIXED_ITERATIONS}
    model = lgb.LGBMRanker(objective="lambdarank", **params)
    model.fit(
        rank.to_pandas_X(fit_df),
        fit_df["label"].to_numpy(),
        group=rank._group_sizes(fit_df),
        categorical_feature=rank.CATEGORICAL_COLUMNS,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rank.save_categorical_mapping(categorical_mapping, BACKTEST_MAPPING_PATH)

    tracking.configure_mlflow()
    tracking.ensure_experiment("stage4_ranking")

    with mlflow.start_run(run_name="stage4_2b_backtest_ranker") as run_:
        mlflow.log_params(model.get_params())
        mlflow.log_param("train_windows", BACKTEST_TRAIN_WINDOWS)
        mlflow.log_param("evaluation_window", "2020-09-09")
        mlflow.log_param("fixed_iterations", FIXED_ITERATIONS)
        mlflow.log_param(
            "fixed_iterations_source",
            "early stopping on 2020-09-09 in the production ranker training (src/rank.py train_ranker); "
            "reused fixed here, no early stopping run against 2020-09-09 in this training",
        )
        mlflow.log_dict(categorical_mapping, "categorical_mapping.json")
        mlflow.log_dict(rank.FEATURE_COLUMNS, "feature_columns.json")
        mlflow.lightgbm.log_model(
            model, name="ranker", skops_trusted_types=[f"lightgbm.sklearn.{type(model).__name__}"]
        )
        run_id = run_.info.run_id

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    manifest = {
        "train_windows": BACKTEST_TRAIN_WINDOWS,
        "train_rows": train_df.height,
        "fixed_iterations": FIXED_ITERATIONS,
        "mlflow_run_id": run_id,
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak["bytes"] / 1e9, 2),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
