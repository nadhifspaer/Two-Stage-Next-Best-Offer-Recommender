import src.env_config  # noqa: F401
import json, sys
import src.memory_guard as mg
import src.phase_peaks as pp
import pipelines.build_features as bf

s = pp.PhasePeaks()
r = bf.run()
s.stop()
r["peak_rss_gb"] = max(r.get("peak_rss_gb") or 0, s.peak_rss_gb)
print(json.dumps({k: v for k, v in r.items() if k != "partition_reports"}, indent=2, default=str))
mg.check_peak_rss("build_validation_features", r)
print("within guard", r["peak_rss_gb"])
