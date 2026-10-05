# Per-stage memory ceiling checked right after each flow task returns, so a flow stops at the first stage that breaches it;
# a report with no peak fails, never passes: no stage is exempt
import time

import src.phase_peaks as phase_peaks

MEMORY_GUARD_GB = 5.5  # above the budget so a stage landing on budget does not trip the guard


class MemoryBudgetExceeded(RuntimeError):
    pass


class PeakNotReported(RuntimeError):
    pass


def measure(fn) -> dict:
    """Runs fn() and returns its dict report with wall_time_s and the
    process peak_rss_gb added, for stages that do not sample their own."""
    sampler = phase_peaks.PhasePeaks(track_uss=False)
    t0 = time.time()
    report = fn()
    sampler.stop()
    report["wall_time_s"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = max(report.get("peak_rss_gb") or 0, sampler.peak_rss_gb)
    return report


def _reports(stage_name: str, report) -> list[dict]:
    # a list of per-window reports, or a dict that nests them under "windows"
    if isinstance(report, dict) and isinstance(report.get("windows"), list):
        items = report["windows"]
    elif isinstance(report, list):
        items = report
    else:
        items = [report]
    if not items or not all(isinstance(item, dict) for item in items):
        raise PeakNotReported(f"{stage_name}: report has no dict to read a peak_rss_gb from")
    return items


def check_peak_rss(stage_name: str, report) -> None:
    for item in _reports(stage_name, report):
        peak_rss_gb = item.get("peak_rss_gb")
        if peak_rss_gb is None:
            raise PeakNotReported(f"{stage_name}: report has no peak_rss_gb, so the guard cannot evaluate it")
        if peak_rss_gb > MEMORY_GUARD_GB:
            # parent and partition phases have different causes; name both when reported
            breakdown = ""
            if "parent_peak_rss_gb" in item:
                breakdown = (
                    f" (parent_peak_rss_gb={item['parent_peak_rss_gb']},"
                    f" partition_peak_rss_gb={item.get('partition_peak_rss_gb')})"
                )
            raise MemoryBudgetExceeded(
                f"{stage_name}: peak_rss_gb={peak_rss_gb} exceeds the {MEMORY_GUARD_GB} GB guard{breakdown}"
            )
