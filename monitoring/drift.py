"""Drift check: Evidently PSI of the training reference against the scoring week, gated on the sampling-matched comparison."""
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import math
import sys
import time
from pathlib import Path

import mlflow.lightgbm
import pandas as pd
import polars as pl
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

import src.calibrate as calibrate
import src.dataset as dataset
import src.phase_peaks as phase_peaks
import src.production_models as production_models
import src.rank as rank
import src.tracking as tracking

PSI_FEATURE_THRESHOLD = 0.25
MAX_DRIFTED_FEATURE_SHARE = 0.25
PSI_PROBABILITY_THRESHOLD = 0.25
MIN_SAMPLE_ROWS = 50_000

SAMPLE_ROWS = 100_000
SAMPLE_SEED = 20200916
CURRENT_HASH_MODULUS = 1_000
MATCHED_CUSTOMER_MODULUS = 4

PROBABILITY_COLUMN = "calibrated_probability"
DATA_DIR = Path("data/processed")
TRAIN_FEATURES_DIR = DATA_DIR / "features" / "train"
VALIDATION_FEATURES_GLOB = DATA_DIR / "features" / "validation" / "part-*.parquet"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = Path("reports")

FEATURE_COLUMNS = rank.FEATURE_COLUMNS
CATEGORICAL_FEATURES = list(rank.CATEGORICAL_COLUMNS) + [c for c in FEATURE_COLUMNS if c.startswith("from_")]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def column_psi(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path | None = None) -> dict[str, float]:
    """PSI per column from an Evidently drift report, optionally saved as HTML."""
    columns = [c for c in reference.columns if c in current.columns]
    categorical = [c for c in columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in columns if c not in categorical]
    ref, cur = reference[columns].copy(), current[columns].copy()
    for c in categorical:
        ref[c], cur[c] = ref[c].astype(str), cur[c].astype(str)
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)
    report = Report(
        [DataDriftPreset(method="psi", threshold=PSI_FEATURE_THRESHOLD, drift_share=MAX_DRIFTED_FEATURE_SHARE)]
    )
    snapshot = report.run(
        current_data=Dataset.from_pandas(cur, data_definition=definition),
        reference_data=Dataset.from_pandas(ref, data_definition=definition),
    )
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    psi = {}
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith(":ValueDrift"):
            psi[config["column"]] = float(metric["value"])
    return psi


def decide(psi: dict[str, float], reference_rows: int, current_rows: int) -> dict:
    """Applies the four threshold rules to per-column PSI."""
    if min(reference_rows, current_rows) < MIN_SAMPLE_ROWS:
        raise ValueError(f"need at least {MIN_SAMPLE_ROWS} rows per side, got {reference_rows} and {current_rows}")
    features = {c: psi[c] for c in FEATURE_COLUMNS if c in psi}
    drifted = sorted(c for c, v in features.items() if math.isnan(v) or v >= PSI_FEATURE_THRESHOLD)
    share = len(drifted) / len(features)
    probability_psi = psi[PROBABILITY_COLUMN]
    probability_drifted = math.isnan(probability_psi) or probability_psi >= PSI_PROBABILITY_THRESHOLD
    share_failed = share > MAX_DRIFTED_FEATURE_SHARE
    return {
        "passed": not share_failed and not probability_drifted,
        "feature_count": len(features),
        "drifted_feature_count": len(drifted),
        "drifted_feature_share": share,
        "drifted_features": drifted,
        "share_rule_failed": share_failed,
        "probability_psi": probability_psi,
        "probability_rule_failed": probability_drifted,
        "feature_psi": features,
    }


def evaluate_drift(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path | None = None) -> dict:
    result = decide(column_psi(reference, current, html_path), len(reference), len(current))
    result["reference_rows"], result["current_rows"] = len(reference), len(current)
    return result


def _score(df: pl.DataFrame, classifier, mapping: dict, calibrator) -> pd.DataFrame:
    # raw feature values for the report, plus the calibrated probability from the pinned models
    raw = classifier.predict_proba(rank.to_pandas_X(rank.apply_categorical_mapping(df, mapping)))[:, 1]
    frame = df.select(FEATURE_COLUMNS).to_pandas()
    frame[PROBABILITY_COLUMN] = calibrate.apply_calibration(calibrator, raw.astype("float32")).astype("float64")
    return frame


def load_reference() -> pl.DataFrame:
    # classifier-in-sample for the probability column
    frame = pl.concat([pl.read_parquet(p, columns=FEATURE_COLUMNS) for p in sorted(TRAIN_FEATURES_DIR.glob("window-*.parquet"))])
    return frame.sample(n=SAMPLE_ROWS, seed=SAMPLE_SEED)


def _pair_hash() -> pl.Expr:
    return pl.struct(pl.col("customer_id").cast(pl.String), pl.col("article_id").cast(pl.String)).hash(seed=SAMPLE_SEED)


def load_current_unmatched() -> pl.DataFrame:
    # diagnostic: a fixed hash slice of the full scoring-week candidate pool, unlabeled
    frame = (
        pl.scan_parquet(VALIDATION_FEATURES_GLOB)
        .filter(_pair_hash() % CURRENT_HASH_MODULUS == 0)
        .select(FEATURE_COLUMNS)
        .collect(engine="streaming")
    )
    return frame.sample(n=min(SAMPLE_ROWS, frame.height), seed=SAMPLE_SEED)


def load_current_matched() -> pl.DataFrame:
    # gated: the training-set construction applied to the scoring week, customers with a purchase and 10 negatives per positive
    window = next(w for w in dataset.label_windows() if w["split"] == "validation")
    keys = (
        pl.scan_parquet(VALIDATION_FEATURES_GLOB)
        .filter(pl.col("customer_id").cast(pl.String).hash(seed=SAMPLE_SEED) % MATCHED_CUSTOMER_MODULUS == 0)
        .select("customer_id", "article_id")
        .collect(engine="streaming")
    )
    transactions = pl.scan_parquet(DATA_DIR / "transactions_train.parquet").with_columns(
        pl.col("customer_id", "article_id").cast(pl.Categorical)
    )
    sampled = dataset.assemble_training_labels(keys, transactions, window["window_start"], window["window_end"])
    del keys
    frame = (
        pl.scan_parquet(VALIDATION_FEATURES_GLOB)
        .join(sampled.select("customer_id", "article_id").lazy(), on=["customer_id", "article_id"], how="semi")
        .select(FEATURE_COLUMNS)
        .collect(engine="streaming")
    )
    if frame.height != sampled.height:
        raise RuntimeError(f"matched sample has {frame.height} feature rows for {sampled.height} sampled pairs")
    return frame.sample(n=min(SAMPLE_ROWS, frame.height), seed=SAMPLE_SEED)


def run() -> dict:
    pl.enable_string_cache()
    sampler = phase_peaks.PhasePeaks(track_uss=False)
    t0 = time.time()

    tracking.configure_mlflow()
    production = production_models.load_production_models()
    classifier = mlflow.lightgbm.load_model(f"runs:/{production.classifier_run_id}/classifier")
    mapping = rank.load_categorical_mapping(MODELS_DIR / "categorical_mapping.json")
    calibrator = calibrate.load_calibrator(production.calibrator_path)
    scoring_date = str(next(w for w in dataset.label_windows() if w["split"] == "validation")["window_start"])

    reference = _score(load_reference(), classifier, mapping, calibrator)
    unmatched = _score(load_current_unmatched(), classifier, mapping, calibrator)
    unmatched_result = evaluate_drift(reference, unmatched, REPORTS_DIR / f"drift_report_{scoring_date}_unmatched.html")
    del unmatched
    matched = _score(load_current_matched(), classifier, mapping, calibrator)
    matched_result = evaluate_drift(reference, matched, REPORTS_DIR / f"drift_report_{scoring_date}.html")
    sampler.stop()

    result = {
        "scoring_date": scoring_date,
        "thresholds": {
            "psi_feature": PSI_FEATURE_THRESHOLD,
            "max_drifted_feature_share": MAX_DRIFTED_FEATURE_SHARE,
            "psi_probability": PSI_PROBABILITY_THRESHOLD,
            "min_sample_rows": MIN_SAMPLE_ROWS,
        },
        "sampling": {"rows_per_side": SAMPLE_ROWS, "seed": SAMPLE_SEED, "current_hash_modulus": CURRENT_HASH_MODULUS,
                     "matched_customer_modulus": MATCHED_CUSTOMER_MODULUS},
        "classifier_run_id": production.classifier_run_id,
        "gated_matched": {"gate": True, **matched_result},
        "diagnostic_unmatched": {"gate": False, **unmatched_result},
        "wall_time_s": round(time.time() - t0, 1),
        "peak_rss_gb": sampler.peak_rss_gb,
    }
    out = REPORTS_DIR / f"drift_result_{scoring_date}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def gate_passed(result: dict) -> bool:
    # only the sampling-matched comparison gates, the unmatched one is a diagnostic
    return result["gated_matched"]["passed"]


def main() -> int:
    result = run()
    summary = {k: result["gated_matched"][k] for k in ("passed", "drifted_feature_count", "drifted_feature_share", "probability_psi")}
    print(json.dumps({"verdict": "PASS" if summary["passed"] else "FAIL", **summary, "peak_rss_gb": result["peak_rss_gb"],
                      "wall_time_s": result["wall_time_s"]}, indent=2))
    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    sys.exit(main())
