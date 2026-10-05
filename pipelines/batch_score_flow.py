# Prefect flow: score a date and write the offer store, one subprocess per task.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from prefect import flow, get_run_logger, task

import src.memory_guard as memory_guard
import src.nbo as nbo


def _log_report(name: str, report) -> None:
    logger = get_run_logger()
    if isinstance(report, dict) and "wall_time_s" in report:
        logger.info(f"{name}: wall_time_s={report['wall_time_s']} peak_rss_gb={report.get('peak_rss_gb')}")
    else:
        logger.info(f"{name}: {report}")
    memory_guard.check_peak_rss(name, report)


def _run_stage(name: str, stage: str) -> dict:
    """Runs one stage in its own subprocess and checks its report against the memory guard."""
    with tempfile.TemporaryDirectory(prefix="batch_flow_") as tmp:
        report_path = Path(tmp) / f"{stage}.json"
        proc = subprocess.run(
            [sys.executable, "-m", "pipelines.run_guarded_stage", stage, "--report-path", str(report_path)],
            check=False,
        )
        if not report_path.exists():
            raise RuntimeError(f"{name}: stage process exited {proc.returncode} without writing a report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    _log_report(name, report)
    if proc.returncode != 0:
        raise RuntimeError(f"{name}: stage process exited {proc.returncode}")
    return report


@task(name="validate_processed_tables")
def validate_processed_tables_task() -> dict:
    """Pandera-validates the four converted tables before any transform runs."""
    return _run_stage("validate_processed_tables", "validate_processed_tables")


@task(name="generate_candidates")
def generate_candidates_task() -> dict:
    return _run_stage("generate_candidates", "generate_candidates")


@task(name="build_features")
def build_features_task() -> dict:
    return _run_stage("build_features", "build_features")


@task(name="score_ranker_and_top12")
def score_validation_ranker_task() -> dict:
    return _run_stage("score_ranker_and_top12", "score_validation_ranker")


@task(name="score_validation_candidates_for_calibration_reuse")
def score_validation_candidates_task() -> dict:
    """Rebuilds the calibration scoring partitions so they match the rebuilt features."""
    return _run_stage("score_validation_candidates_for_calibration_reuse", "score_validation_candidates")


@task(name="score_calibrated_probabilities")
def score_nbo_probabilities_task() -> dict:
    return _run_stage("score_calibrated_probabilities", "score_nbo_probabilities")


@task(name="decision_layer")
def run_nbo_task() -> dict:
    return _run_stage("decision_layer", "run_nbo")


@task(name="build_offer_store")
def build_offer_store_task() -> dict:
    return _run_stage("build_offer_store", "build_offer_store")


@flow(name="batch_score_flow")
def batch_score_flow(scoring_date: date) -> dict:
    if scoring_date != nbo.SCORING_DATE:
        raise NotImplementedError(
            f"batch_score_flow only supports scoring_date={nbo.SCORING_DATE} today: every task it "
            "calls (candidates, features, ranker/probability scoring, the decision layer, the offer "
            "store build) is hardcoded to that date, not yet parameterized. See this file's header."
        )

    validation_report = validate_processed_tables_task()

    candidates_report = generate_candidates_task()
    features_report = build_features_task()
    ranker_report = score_validation_ranker_task()
    calibration_scoring_report = score_validation_candidates_task()
    probabilities_report = score_nbo_probabilities_task()
    decision_report = run_nbo_task()
    offer_store_report = build_offer_store_task()

    return {
        "scoring_date": str(scoring_date),
        "validation": validation_report,
        "candidates": candidates_report,
        "calibration_scoring": calibration_scoring_report,
        "features": features_report,
        "ranker_top12": ranker_report,
        "calibrated_probabilities": probabilities_report,
        "decision_layer": decision_report,
        "offer_store": offer_store_report,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-date", type=date.fromisoformat, default=nbo.SCORING_DATE)
    args = parser.parse_args()

    result = batch_score_flow(args.scoring_date)
    print(json.dumps(result, indent=2, default=str))
