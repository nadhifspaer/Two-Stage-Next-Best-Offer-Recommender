# per-phase RSS sampler for a single process, recorded in stage manifests
import threading
import time
from contextlib import contextmanager

import psutil


class PhasePeaks:
    def __init__(self, interval_s: float = 0.1, track_uss: bool = True):
        self._proc = psutil.Process()
        self._interval_s = interval_s
        self._track_uss = track_uss
        self._stop = threading.Event()
        self._current: str | None = None
        self._peaks: dict[str, dict] = {}
        self._overall_rss = 0
        self._overall_uss = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            # memory_full_info computes USS and is slow on Windows; rss only when USS is not tracked
            if self._track_uss:
                full = self._proc.memory_full_info()
                self._overall_uss = max(self._overall_uss, full.uss)
                rss = full.rss
            else:
                rss = self._proc.memory_info().rss
            self._overall_rss = max(self._overall_rss, rss)
            phase = self._peaks.get(self._current)
            if phase is not None:
                phase["peak_rss_bytes"] = max(phase["peak_rss_bytes"], rss)
            time.sleep(self._interval_s)

    @contextmanager
    def phase(self, name: str):
        start_rss = self._proc.memory_info().rss
        self._peaks[name] = {"start_rss_bytes": start_rss, "peak_rss_bytes": start_rss, "t0": time.time()}
        self._current = name
        try:
            yield
        finally:
            end = self._proc.memory_info().rss
            entry = self._peaks[name]
            entry["peak_rss_bytes"] = max(entry["peak_rss_bytes"], end)
            entry["end_rss_bytes"] = end
            entry["wall_time_s"] = time.time() - entry["t0"]
            self._current = None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak_rss_gb(self) -> float:
        return round(self._overall_rss / 1e9, 2)

    @property
    def peak_uss_gb(self) -> float:
        return round(self._overall_uss / 1e9, 2)

    def report(self) -> dict:
        return {
            name: {
                "start_rss_gb": round(v["start_rss_bytes"] / 1e9, 2),
                "peak_rss_gb": round(v["peak_rss_bytes"] / 1e9, 2),
                "end_rss_gb": round(v.get("end_rss_bytes", 0) / 1e9, 2),
                "wall_time_s": round(v.get("wall_time_s", 0.0), 1),
            }
            for name, v in self._peaks.items()
        }
