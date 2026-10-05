# Recall manifest: atomic write and publish predicate
import json
from pathlib import Path

import pipelines.evaluate_candidates as evaluate_candidates


def _result() -> dict:
    return {
        "window_start": "2020-09-16",
        "window_end": "2020-09-22",
        "partitions": 2,
        "true_pairs": 10,
        "pool_hits": 4,
        "pool_recall_at_100": 0.4,
        "per_strategy": [{"strategy": "als", "recall_at_100": 0.1, "marginal_contribution": 0.05}],
        "per_recency_bucket": [{"recency_bucket": "never", "true_pairs": 10, "recall_at_100": 0.4}],
        "wall_time_s": 1.0,
        "peak_rss_gb": 1.0,
    }


def test_write_recall_manifest_round_trips_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "candidates" / "_recall_manifest.json"
    evaluate_candidates.write_recall_manifest(_result(), path)
    assert json.loads(path.read_text(encoding="utf-8")) == _result()
    assert [p.name for p in path.parent.iterdir()] == ["_recall_manifest.json"]


def test_write_recall_manifest_replaces_an_existing_file(tmp_path):
    path = tmp_path / "_recall_manifest.json"
    path.write_text("{}", encoding="utf-8")
    evaluate_candidates.write_recall_manifest(_result(), path)
    assert json.loads(path.read_text(encoding="utf-8"))["pool_hits"] == 4


def test_only_the_committed_validation_run_publishes():
    assert evaluate_candidates.publishes_manifest(evaluate_candidates.CANDIDATES_DIR, None)
    assert not evaluate_candidates.publishes_manifest(Path("scratch/candidates"), None)
    assert not evaluate_candidates.publishes_manifest(
        evaluate_candidates.CANDIDATES_DIR, {"window_start": "2020-09-09"}
    )
