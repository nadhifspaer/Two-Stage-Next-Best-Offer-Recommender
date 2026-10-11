import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LOCK = Path("requirements.lock.txt")
PLATFORM_VARS = ("sys_platform", "platform_system", "os_name")


# pip freeze drops environment markers; restore platform markers from installed metadata
def platform_markers(locked):
    edges = {}
    for dist in metadata.distributions():
        if canonicalize_name(dist.metadata["Name"]) not in locked:
            continue
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker and "extra" in str(req.marker):
                continue
            edges.setdefault(canonicalize_name(req.name), []).append(req.marker)
    out = {}
    for name, markers in edges.items():
        if name not in locked or any(m is None for m in markers):
            continue
        if all(any(v in str(m) for v in PLATFORM_VARS) for m in markers):
            out[name] = " or ".join(sorted({str(m) for m in markers}))
    return out


# metadata must come from the environment the lockfile describes
def env_mismatches(pins):
    out = []
    for name, ver in pins.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        if have != ver:
            out.append(f"{name}: lock {ver}, installed {have}")
    return out


def main(write):
    lines = LOCK.read_text().splitlines()
    locked = {
        canonicalize_name(l.split("==")[0].strip())
        for l in lines
        if "==" in l and not l.lstrip().startswith("#")
    }
    pins = {
        canonicalize_name(l.split("==")[0].strip()): l.split("==")[1].split(";")[0].strip()
        for l in lines
        if "==" in l and not l.lstrip().startswith("#")
    }
    mismatches = env_mismatches(pins)
    if mismatches:
        print("installed environment does not match the lockfile:")
        for m in mismatches:
            print("  " + m)
        return 2
    markers = platform_markers(locked)
    new, bad = [], []
    for l in lines:
        if "==" in l and not l.lstrip().startswith("#"):
            pin = l.split(";")[0].strip()
            name = canonicalize_name(pin.split("==")[0])
            if name in markers:
                want = f"{pin} ; {markers[name]}"
                if l.strip() != want:
                    bad.append(want)
                l = want
        new.append(l)
    for b in bad:
        print("platform marker:", b)
    if write:
        LOCK.write_text("\n".join(new) + "\n")
    return 1 if bad and not write else 0


if __name__ == "__main__":
    sys.exit(main("--write" in sys.argv))
