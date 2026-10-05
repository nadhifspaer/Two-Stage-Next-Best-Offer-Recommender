# Trains the LGBMRanker and LGBMClassifier on the training matrix; calibration and MAP@12 run elsewhere.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import threading
import time
from pathlib import Path

import mlflow
import polars as pl
import psutil

import src.candidates as candidates
import src.dataset as dataset
import pipelines.report_score_distributions as report_score_distributions
import src.tracking as tracking
import src.rank as rank

TRAIN_DIR = Path("data/processed/features/train")
MODELS_DIR = Path("data/processed/models")
MANIFEST_PATH = MODELS_DIR / "_manifest.json"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"


def _peak_rss_sampler(peak: dict, stop_flag: threading.Event, interval_s: float = 0.3) -> None:
    proc = psutil.Process()
    while not stop_flag.is_set():
        rss = proc.memory_info().rss
        if rss > peak["bytes"]:
            peak["bytes"] = rss
        time.sleep(interval_s)


def _candidate_generation_params() -> dict:
    return {
        "strategy_names": candidates.STRATEGY_NAMES,
        "repurchase_cap": candidates.REPURCHASE_CAP,
        "colour_variant_cap": candidates.COLOUR_VARIANT_CAP,
        "recent_popularity_cap": candidates.RECENT_POPULARITY_CAP,
        "segment_popularity_cap": candidates.SEGMENT_POPULARITY_CAP,
        "als_cap": candidates.ALS_CAP,
        "pool_cap": candidates.POOL_CAP,
        "als_random_state": candidates.ALS_RANDOM_STATE,
        "customer_chunk_size": candidates.CUSTOMER_CHUNK_SIZE,
        "negatives_per_positive": dataset.NEGATIVES_PER_POSITIVE,
        "negative_sampling_seed": dataset.NEGATIVE_SAMPLING_SEED,
    }


def run() -> dict:
    t0 = time.time()
    peak_rss = {"bytes": 0}
    stop_flag = threading.Event()
    sampler = threading.Thread(target=_peak_rss_sampler, args=(peak_rss, stop_flag), daemon=True)
    sampler.start()

    pl.enable_string_cache()
    window_paths = sorted(TRAIN_DIR.glob("window-*.parquet"))
    if not window_paths:
        raise FileNotFoundError(f"no window-*.parquet files under {TRAIN_DIR}")
    train_df = pl.concat([pl.read_parquet(p) for p in window_paths])

    categorical_mapping = rank.fit_categorical_mapping(train_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rank.save_categorical_mapping(categorical_mapping, MODELS_DIR / "categorical_mapping.json")
    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(rank.FEATURE_COLUMNS, f, indent=2)

    tracking.configure_mlflow()
    tracking.ensure_experiment("stage4_ranking")

    shared_params = {
        "feature_column_count": len(rank.FEATURE_COLUMNS),
        "categorical_column_count": len(rank.CATEGORICAL_COLUMNS),
        "early_stopping_train_windows": rank.EARLY_STOPPING_TRAIN_WINDOWS,
        "early_stopping_validation_window": rank.EARLY_STOPPING_VALIDATION_WINDOW,
        "early_stopping_rounds": rank.EARLY_STOPPING_ROUNDS,
        "lgbm_seed": rank.LGBM_SEED,
        "train_rows": train_df.height,
        **_candidate_generation_params(),
    }

    with mlflow.start_run(run_name="stage4_1_training") as parent_run:
        mlflow.log_params({f"shared.{k}": v for k, v in shared_params.items() if not isinstance(v, list)})
        mlflow.log_dict(rank.FEATURE_COLUMNS, "feature_columns.json")
        mlflow.log_dict(categorical_mapping, "categorical_mapping.json")
        mlflow.log_dict(shared_params, "shared_params.json")

        ranker_report, ranker_model = _train_and_log_model(
            "ranker", rank.train_ranker, train_df, categorical_mapping, shared_params
        )
        classifier_report, classifier_model = _train_and_log_model(
            "classifier", rank.train_classifier, train_df, categorical_mapping, shared_params
        )

    # distributions for the notebooks to read, computed with the models just trained
    report_score_distributions.write(
        ranker_model, classifier_model, train_df, categorical_mapping,
        {"ranker_run_id": ranker_report["run_id"], "classifier_run_id": classifier_report["run_id"]},
    )

    wall_time_s = time.time() - t0
    stop_flag.set()
    sampler.join(timeout=2)

    manifest = {
        "train_rows": train_df.height,
        "wall_time_s": round(wall_time_s, 1),
        "peak_rss_gb": round(peak_rss["bytes"] / 1e9, 2),
        "mlflow_run_id": parent_run.info.run_id,
        # pinned production model identity, captured as each run is created; consumers read it, never "latest run"
        # (the backtest ranker shares this experiment)
        "ranker_run_id": ranker_report["run_id"],
        "classifier_run_id": classifier_report["run_id"],
        "calibrator_path": str(CALIBRATOR_PATH),
        "ranker": ranker_report,
        "classifier": classifier_report,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _train_and_log_model(name: str, train_fn, train_df: pl.DataFrame, categorical_mapping: dict, shared_params: dict) -> tuple[dict, object]:
    import mlflow.lightgbm

    t0 = time.time()
    model, meta = train_fn(train_df, categorical_mapping)
    wall_time_s = time.time() - t0

    with mlflow.start_run(run_name=f"stage4_1_{name}", nested=True) as run:
        mlflow.log_params(model.get_params())
        mlflow.log_param("best_iteration", meta["best_iteration"])
        mlflow.log_metric("wall_time_s", round(wall_time_s, 1))
        # skops refuses untrusted classes by default; trust the specific LightGBM sklearn class explicitly
        # rather than broadening trust or falling back to pickle
        mlflow.lightgbm.log_model(
            model, name=name, skops_trusted_types=[f"lightgbm.sklearn.{type(model).__name__}"]
        )
        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "best_iteration": meta["best_iteration"],
        "wall_time_s": round(wall_time_s, 1),
    }, model


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
