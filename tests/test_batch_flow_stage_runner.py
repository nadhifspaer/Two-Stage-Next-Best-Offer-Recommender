# subprocess stage runner in pipelines/batch_score_flow.py
import json
import logging
import subprocess
from pathlib import Path

import pytest

import pipelines.batch_score_flow as flow
import src.memory_guard as memory_guard


class _Completed:
    def __init__(self, returncode):
        self.returncode = returncode


def _fake_run(report, returncode=0):
    def run(cmd, check=False):
        if report is not None:
            Path(cmd[cmd.index("--report-path") + 1]).write_text(json.dumps(report), encoding="utf-8")
        return _Completed(returncode)

    return run


@pytest.fixture(autouse=True)
def _logger(monkeypatch):
    monkeypatch.setattr(flow, "get_run_logger", lambda: logging.getLogger("test"))


def test_stage_report_is_returned(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run({"wall_time_s": 1.0, "peak_rss_gb": 3.0}))
    assert flow._run_stage("name", "stage")["peak_rss_gb"] == 3.0


def test_guard_breach_in_stage_raises_budget_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run({"wall_time_s": 1.0, "peak_rss_gb": 6.01}, returncode=1))
    with pytest.raises(memory_guard.MemoryBudgetExceeded):
        flow._run_stage("name", "stage")


def test_report_without_peak_raises(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run({"wall_time_s": 1.0}))
    with pytest.raises(memory_guard.PeakNotReported):
        flow._run_stage("name", "stage")


def test_crash_without_report_raises(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(None, returncode=3))
    with pytest.raises(RuntimeError, match="without writing a report"):
        flow._run_stage("name", "stage")


def test_nonzero_exit_with_clean_report_raises(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run({"wall_time_s": 1.0, "peak_rss_gb": 3.0}, returncode=2))
    with pytest.raises(RuntimeError, match="exited 2"):
        flow._run_stage("name", "stage")
