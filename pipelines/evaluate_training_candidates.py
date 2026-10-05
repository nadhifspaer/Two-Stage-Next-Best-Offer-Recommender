# Recall@100 for each training window, a date-bound sanity check against the validation window.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import json
from pathlib import Path

import src.dataset as dataset
import pipelines.evaluate_candidates as evaluate_candidates

CANDIDATES_ROOT = Path("data/processed/candidates/train")
VALIDATION_RECALL_AT_100 = 0.1011


def run() -> list[dict]:
    train_windows = [w for w in dataset.label_windows() if w["split"] == "train"]
    results = []
    for window in train_windows:
        candidates_dir = CANDIDATES_ROOT / str(window["window_start"])
        result = evaluate_candidates.run(candidates_dir=candidates_dir, window=window)
        results.append(
            {
                "window_start": result["window_start"],
                "window_end": result["window_end"],
                "true_pairs": result["true_pairs"],
                "pool_recall_at_100": result["pool_recall_at_100"],
                "wall_time_s": result["wall_time_s"],
                "peak_rss_gb": result["peak_rss_gb"],
            }
        )
    return results


if __name__ == "__main__":
    results = run()
    print(json.dumps(results, indent=2))
    print()
    print(f"validation Recall@100 for comparison: {VALIDATION_RECALL_AT_100}")
    for r in results:
        print(f"  {r['window_start']}  pool_recall_at_100={r['pool_recall_at_100']:.4f}")
