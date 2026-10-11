"""Environment & data-provenance capture for experiment logging.

All values returned are JSON-serializable (native Python types only).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

#: Packages whose versions are recorded in every run's environment block.
TRACKED_PACKAGES = (
    "numpy",
    "faiss-cpu",
    "faiss",
    "scipy",
    "matplotlib",
    "seaborn",
    "h5py",
    "tqdm",
    "pyarrow",
    "huggingface-hub",
)

#: Files larger than this are fingerprinted partially (fast) instead of full SHA-256.
_PARTIAL_HASH_THRESHOLD = 256 * 1024 * 1024
_PARTIAL_CHUNK = 8 * 1024 * 1024


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _total_ram_bytes() -> int | None:
    """Parse MemTotal from /proc/meminfo (Linux); None elsewhere."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except OSError:
        pass
    return None


def _gpu_info() -> str | None:
    """Report GPU presence via nvidia-smi if available (expected: None on CPU boxes)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or nvidia_smi
    except (OSError, subprocess.SubprocessError):
        return nvidia_smi


def _git_state(repo_root: Path | None) -> dict[str, str | bool] | None:
    """Capture git commit + dirty flag; None if not a git repo."""
    try:
        root = repo_root or Path.cwd()
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if commit.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return {
            "commit": commit.stdout.strip(),
            "dirty": bool(dirty.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def capture_environment(repo_root: Path | str | None = None) -> dict:
    """Capture a JSON-serializable snapshot of the runtime environment."""
    root = Path(repo_root) if repo_root else None
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "cpu_cores": os.cpu_count(),
        "ram_bytes": _total_ram_bytes(),
        "gpu": _gpu_info(),
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "packages": _package_versions(),
        "git": _git_state(root),
        "cwd": str(Path.cwd()),
    }


def fingerprint_file(path: Path | str) -> dict:
    """Fingerprint a data file for provenance tracking.

    Small files (<= 256 MB) get a full SHA-256. Large files get a fast
    partial fingerprint: hash(first 8 MB) + hash(last 8 MB) + size + mtime,
    which is collision-resistant enough for provenance checks without
    re-reading multi-GB files on every run.
    """
    p = Path(path)
    stat = p.stat()
    size = stat.st_size
    mtime = stat.st_mtime_ns

    if size <= _PARTIAL_HASH_THRESHOLD:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        mode = "sha256"
    else:
        head = hashlib.sha256()
        tail = hashlib.sha256()
        with p.open("rb") as fh:
            head.update(fh.read(_PARTIAL_CHUNK))
            fh.seek(max(0, size - _PARTIAL_CHUNK))
            tail.update(fh.read(_PARTIAL_CHUNK))
        digest = hashlib.sha256(
            head.digest() + tail.digest() + str(size).encode() + str(mtime).encode()
        ).hexdigest()
        mode = "partial"

    return {
        "path": str(p.resolve()),
        "size_bytes": size,
        "mtime_ns": mtime,
        "digest": digest,
        "mode": mode,
    }
