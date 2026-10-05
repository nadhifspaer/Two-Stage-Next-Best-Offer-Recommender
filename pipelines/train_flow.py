# Prefect flow: candidates, features, train, evaluate, log, orchestrating the existing scripts.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json

from prefect import flow, get_run_logger, task

import pipelines.build_features as build_features
import pipelines.build_training_features as build_training_features
import pipelines.calibrate_classifier as calibrate_classifier
import pipelines.evaluate_candidates as evaluate_candidates
import pipelines.evaluate_map12 as evaluate_map12
import pipelines.evaluate_training_candidates as evaluate_training_candidates
import pipelines.generate_candidates as generate_candidates
import pipelines.generate_training_candidates as generate_training_candidates
import pipelines.score_validation_candidates as score_validation_candidates
import pipelines.score_validation_ranker as score_validation_ranker
import pipelines.train_ranking_models as train_ranking_models
import src.dataset as dataset
import src.memory_guard as memory_guard


def _log_report(name: str, report) -> None:
    logger = get_run_logger()
    if isinstance(report, dict) and "wall_time_s" in report:
        logger.info(f"{name}: wall_time_s={report['wall_time_s']} peak_rss_gb={report.get('peak_rss_gb')}")
    else:
        logger.info(f"{name}: {report}")
    memory_guard.check_peak_rss(name, report)


@task(name="validate_and_convert_raw")
def validate_and_convert_raw_task() -> dict:
    """Pandera-validates each raw table before any is written to data/processed/."""
    report = memory_guard.measure(dataset.convert_raw_to_parquet)
    _log_report("convert_raw_to_parquet", report)
    return report


@task(name="generate_training_candidates")
def generate_training_candidates_task() -> dict:
    report = generate_training_candidates.run()
    _log_report("generate_training_candidates", report)
    return report


@task(name="generate_validation_candidates")
def generate_validation_candidates_task() -> dict:
    report = generate_candidates.run()
    _log_report("generate_validation_candidates", report)
    return report


@task(name="evaluate_training_candidates_recall")
def evaluate_training_candidates_task() -> list:
    report = evaluate_training_candidates.run()
    _log_report("evaluate_training_candidates_recall", report)
    return report


@task(name="evaluate_validation_candidates_recall")
def evaluate_validation_candidates_task() -> dict:
    report = evaluate_candidates.run()
    _log_report("evaluate_validation_candidates_recall", report)
    return report


@task(name="build_training_labels_and_features")
def build_training_features_task() -> dict:
    report = build_training_features.run()
    _log_report("build_training_labels_and_features", report)
    return report


@task(name="build_validation_features")
def build_validation_features_task() -> dict:
    report = build_features.run()
    _log_report("build_validation_features", report)
    return report


@task(name="train_ranker_and_classifier")
def train_ranking_models_task() -> dict:
    """Trains both models: fit on the first three windows, evaluate on the fourth, refit at that iteration count."""
    report = train_ranking_models.run()
    _log_report("train_ranker_and_classifier", report)
    return report


@task(name="score_validation_candidates_for_calibration")
def score_validation_candidates_task() -> dict:
    report = score_validation_candidates.run()
    _log_report("score_validation_candidates_for_calibration", report)
    return report


@task(name="calibrate_classifier")
def calibrate_classifier_task() -> dict:
    report = calibrate_classifier.run()
    _log_report("calibrate_classifier", report)
    return report


@task(name="score_validation_ranker")
def score_validation_ranker_task() -> dict:
    report = score_validation_ranker.run()
    _log_report("score_validation_ranker", report)
    return report


@task(name="evaluate_validation_map12")
def evaluate_map12_task() -> dict:
    report = evaluate_map12.run()
    _log_report("evaluate_validation_map12", report)
    return report


@flow(name="train_flow")
def train_flow() -> dict:
    conversion_report = validate_and_convert_raw_task()

    training_candidates_report = generate_training_candidates_task()
    validation_candidates_report = generate_validation_candidates_task()

    training_recall_report = evaluate_training_candidates_task()
    validation_recall_report = evaluate_validation_candidates_task()

    training_features_report = build_training_features_task()
    validation_features_report = build_validation_features_task()

    training_report = train_ranking_models_task()

    scored_candidates_report = score_validation_candidates_task()
    calibration_report = calibrate_classifier_task()

    ranker_scores_report = score_validation_ranker_task()
    map12_report = evaluate_map12_task()

    return {
        "conversion": conversion_report,
        "training_candidates": training_candidates_report,
        "validation_candidates": validation_candidates_report,
        "training_recall": training_recall_report,
        "validation_recall": validation_recall_report,
        "training_features": training_features_report,
        "validation_features": validation_features_report,
        "training": training_report,
        "scored_validation_candidates": scored_candidates_report,
        "calibration": calibration_report,
        "ranker_scores": ranker_scores_report,
        "map12": map12_report,
    }


if __name__ == "__main__":
    result = train_flow()
    print(json.dumps(result, indent=2, default=str))
