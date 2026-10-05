# peak RSS guard behaviour for src/memory_guard.py
import pytest

import src.memory_guard as memory_guard


def test_peak_under_guard_passes():
    memory_guard.check_peak_rss("stage", {"peak_rss_gb": 4.9})


def test_peak_over_guard_raises():
    with pytest.raises(memory_guard.MemoryBudgetExceeded):
        memory_guard.check_peak_rss("stage", {"peak_rss_gb": 6.01})


def test_missing_peak_raises_instead_of_passing():
    with pytest.raises(memory_guard.PeakNotReported):
        memory_guard.check_peak_rss("stage", {"rows": 10})


def test_non_dict_report_raises():
    for report in (None, [], "ok", 3):
        with pytest.raises(memory_guard.PeakNotReported):
            memory_guard.check_peak_rss("stage", report)


def test_per_window_list_checks_every_window():
    memory_guard.check_peak_rss("stage", [{"peak_rss_gb": 3.0}, {"peak_rss_gb": 4.0}])
    with pytest.raises(memory_guard.MemoryBudgetExceeded):
        memory_guard.check_peak_rss("stage", [{"peak_rss_gb": 3.0}, {"peak_rss_gb": 5.6}])
    with pytest.raises(memory_guard.PeakNotReported):
        memory_guard.check_peak_rss("stage", [{"peak_rss_gb": 3.0}, {"wall_time_s": 1.0}])


def test_nested_windows_report_checks_every_window():
    # generate_training_candidates reports per-window peaks under "windows" and none at the top level
    memory_guard.check_peak_rss("stage", {"windows": [{"peak_rss_gb": 4.5}, {"peak_rss_gb": 4.4}]})
    with pytest.raises(memory_guard.MemoryBudgetExceeded):
        memory_guard.check_peak_rss("stage", {"windows": [{"peak_rss_gb": 4.5}, {"peak_rss_gb": 5.7}]})


def test_measure_adds_wall_time_and_peak():
    report = memory_guard.measure(lambda: {"tables": 4})
    assert report["tables"] == 4
    assert report["peak_rss_gb"] > 0
    assert report["wall_time_s"] >= 0
    memory_guard.check_peak_rss("stage", report)
