# Reporting only: re-bins the top decile of calibrated probability into 10 sub-bins and reports top 1% lift.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
from pathlib import Path

import polars as pl

import src.calibrate as calibrate

DATA_DIR = Path("data/processed")
SCORED_DIR = DATA_DIR / "calibration" / "scored"
CALIBRATOR_PATH = DATA_DIR / "models" / "calibrator.joblib"
OUTPUT_PATH = DATA_DIR / "calibration" / "_top_decile_reliability.json"

N_BINS = 10


def run() -> dict:
    pl.enable_string_cache()

    calibrator = calibrate.load_calibrator(CALIBRATOR_PATH)
    scored = pl.concat([pl.read_parquet(p) for p in sorted(SCORED_DIR.glob("part-*.parquet"))])
    eval_df = scored.filter(calibrate.eval_bucket_mask(scored["bucket"]))
    calibrated = calibrate.apply_calibration(calibrator, eval_df["raw_prob"].to_numpy())

    y = eval_df["label"].to_numpy()
    base_rate = float(y.mean())

    # same binning as quantile_reliability_table, at n_bins=10, then take
    # exactly the rows that fell in the top bin (bin 9) there
    keys = eval_df.select("customer_id", "article_id")
    full_table = calibrate.quantile_reliability_table(y, calibrated, keys, n_bins=N_BINS)

    top_decile_mask = calibrate.quantile_bin_index(calibrated, keys, n_bins=N_BINS) == N_BINS - 1

    top_y = y[top_decile_mask]
    top_prob = calibrated[top_decile_mask]

    sub_table = calibrate.quantile_reliability_table(top_y, top_prob, keys.filter(pl.Series(top_decile_mask)), n_bins=N_BINS)
    top_subbin = sub_table.filter(pl.col("bin") == N_BINS - 1).row(0, named=True)
    lift = top_subbin["observed_rate"] / base_rate

    result = {
        "eval_base_rate": base_rate,
        "full_table_top_bin": full_table.filter(pl.col("bin") == N_BINS - 1).to_dicts()[0],
        "top_decile_rows": int(top_decile_mask.sum()),
        "top_decile_sub_bins": sub_table.to_dicts(),
        "top_subbin_observed_rate": top_subbin["observed_rate"],
        "top_subbin_lift_vs_base_rate": lift,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
