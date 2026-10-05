# Fits isotonic calibration on the fit subset, evaluates on the eval subset, persists the calibrator.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import time
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

import src.calibrate as calibrate
import src.phase_peaks as phase_peaks
import src.tracking as tracking

SCORED_DIR = Path("data/processed/calibration/scored")
SCORE_MANIFEST_PATH = Path("data/processed/calibration/_score_manifest.json")
MODELS_MANIFEST_PATH = Path("data/processed/models/_manifest.json")
CALIBRATOR_PATH = Path("data/processed/models/calibrator.joblib")
MANIFEST_PATH = Path("data/processed/calibration/_calibration_manifest.json")


def run() -> dict:
    pl.enable_string_cache()
    sampler = phase_peaks.PhasePeaks(track_uss=False)
    t0 = time.time()

    score_manifest = json.loads(SCORE_MANIFEST_PATH.read_text(encoding="utf-8"))
    scored = pl.concat([pl.read_parquet(p) for p in sorted(SCORED_DIR.glob("part-*.parquet"))])

    fit_df = scored.filter(calibrate.fit_bucket_mask(scored["bucket"]))
    eval_df = scored.filter(calibrate.eval_bucket_mask(scored["bucket"]))

    fit_customers = set(fit_df["customer_id"].unique().to_list())
    eval_customers = set(eval_df["customer_id"].unique().to_list())
    overlap = fit_customers & eval_customers
    if overlap:
        raise ValueError(f"{len(overlap)} customers appear in both the fit and eval subsets")

    calibrator = calibrate.fit_calibrator(fit_df["label"].to_numpy(), fit_df["raw_prob"].to_numpy())

    eval_y = eval_df["label"].to_numpy()
    eval_raw = eval_df["raw_prob"].to_numpy()
    eval_calibrated = calibrate.apply_calibration(calibrator, eval_raw)

    base_rate = float(eval_y.mean())
    raw_brier = calibrate.brier_score(eval_y, eval_raw)
    calibrated_brier = calibrate.brier_score(eval_y, eval_calibrated)
    constant_brier = calibrate.brier_score(eval_y, np.full_like(eval_y, base_rate, dtype=float))
    raw_skill = calibrate.brier_skill_score(raw_brier, base_rate)
    calibrated_skill = calibrate.brier_skill_score(calibrated_brier, base_rate)

    reliability = calibrate.quantile_reliability_table(
        eval_y, eval_calibrated, eval_df.select("customer_id", "article_id"), n_bins=calibrate.N_QUANTILE_BINS
    )

    calibrate.save_calibrator(calibrator, CALIBRATOR_PATH)

    tracking.configure_mlflow()
    models_manifest = json.loads(MODELS_MANIFEST_PATH.read_text(encoding="utf-8"))
    client = mlflow.tracking.MlflowClient()
    classifier_run_id = tracking.find_child_run_id(
        client, models_manifest["mlflow_run_id"], "stage4_1_classifier"
    )

    with mlflow.start_run(run_id=classifier_run_id):
        mlflow.log_param("calibration_fit_bucket_range", [calibrate.FIT_BUCKET_LOWER, calibrate.FIT_BUCKET_UPPER])
        mlflow.log_param("calibration_eval_bucket_range", [calibrate.EVAL_BUCKET_LOWER, calibrate.EVAL_BUCKET_UPPER])
        mlflow.log_param("calibration_split_seed", calibrate.CALIBRATION_SPLIT_SEED)
        mlflow.log_param("calibration_fit_rows", fit_df.height)
        mlflow.log_param("calibration_eval_rows", eval_df.height)
        mlflow.log_metric("mean_raw_probability_full_pool", score_manifest["mean_raw_probability"])
        mlflow.log_metric("observed_positive_rate_full_pool", score_manifest["observed_positive_rate"])
        mlflow.log_metric("eval_base_rate", base_rate)
        mlflow.log_metric("raw_brier", raw_brier)
        mlflow.log_metric("calibrated_brier", calibrated_brier)
        mlflow.log_metric("constant_brier", constant_brier)
        mlflow.log_metric("raw_brier_skill_score", raw_skill)
        mlflow.log_metric("calibrated_brier_skill_score", calibrated_skill)
        mlflow.log_artifact(str(CALIBRATOR_PATH))
        mlflow.log_dict(reliability.to_dicts(), "reliability_table.json")

    manifest = {
        "fit_rows": fit_df.height,
        "eval_rows": eval_df.height,
        "mean_raw_probability_full_pool": score_manifest["mean_raw_probability"],
        "observed_positive_rate_full_pool": score_manifest["observed_positive_rate"],
        "eval_base_rate": base_rate,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "constant_brier": constant_brier,
        "raw_brier_skill_score": raw_skill,
        "calibrated_brier_skill_score": calibrated_skill,
        "reliability_table": reliability.to_dicts(),
        "classifier_run_id": classifier_run_id,
    }
    sampler.stop()
    manifest["wall_time_s"] = round(time.time() - t0, 1)
    manifest["peak_rss_gb"] = sampler.peak_rss_gb
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
