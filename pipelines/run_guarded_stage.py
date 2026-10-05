# runs one pipeline stage's run() in this process under the same 5.5 GB guard as the Prefect flows
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import src.memory_guard as memory_guard
import src.phase_peaks as phase_peaks

# "module" calls module.run(); "module:function" calls that function
STAGES = {
    "validate_processed_tables": "src.dataset:validate_processed_tables",
    "generate_candidates": "pipelines.generate_candidates",
    "evaluate_candidates": "pipelines.evaluate_candidates",
    "build_features": "pipelines.build_features",
    "train_ranking_models": "pipelines.train_ranking_models",
    "report_score_distributions": "pipelines.report_score_distributions",
    "score_validation_candidates": "pipelines.score_validation_candidates",
    "calibrate_classifier": "pipelines.calibrate_classifier",
    "score_validation_ranker": "pipelines.score_validation_ranker",
    "evaluate_map12": "pipelines.evaluate_map12",
    "score_nbo_probabilities": "pipelines.score_nbo_probabilities",
    "run_nbo": "pipelines.run_nbo",
    "build_offer_store": "pipelines.build_offer_store",
    "build_offer_store_sample": "pipelines.build_offer_store_sample",
    "drift_report": "monitoring.drift",
    "report_nbo_divergence": "pipelines.report_nbo_divergence",
    "report_offer_threshold_analysis": "pipelines.report_offer_threshold_analysis",
    "report_top_decile_reliability": "pipelines.report_top_decile_reliability",
    "report_cap_reach": "pipelines.report_cap_reach",
}


def _write_report(path: Path, report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()
    stage = args.stage

    module_name, _, function_name = STAGES[stage].partition(":")
    fn = getattr(importlib.import_module(module_name), function_name or "run")
    sampler = phase_peaks.PhasePeaks(track_uss=False)
    t0 = time.time()
    report = fn()
    sampler.stop()

    # the stage's own peak (children included where it reports one) and this
    # process's peak both count, so a stage without a peak_rss_gb is still guarded
    if isinstance(report, dict):
        report["peak_rss_gb"] = max(report.get("peak_rss_gb") or 0, sampler.peak_rss_gb)
        report.setdefault("wall_time_s", round(time.time() - t0, 1))
    # written before the guard check so a calling flow can read and judge it
    if args.report_path is not None:
        _write_report(args.report_path, report)
    print(json.dumps(report, indent=2, default=str))
    memory_guard.check_peak_rss(stage, report)
    print(f"{stage}: peak_rss_gb={report['peak_rss_gb']} within guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
