"""Structured, disk-persisted experiment logging for NoraIVF rebuttal runs.

Every run writes a self-contained directory::

    results/logs/<run_id>/
        meta.json       # env + config + data provenance (written/updated)
        run.jsonl       # append-only JSONL event stream (flushed per event)
        errors.jsonl    # error events (subset of run.jsonl, for triage)
        events.summary.json  # produced by summarize_run()

Design goals (see rebuttal-experiments-plan.md §4):
- crash-safe: every event is appended + flushed immediately;
  fsync on close and on every error event.
- reproducible: meta.json pins environment, git commit and data fingerprints.
- bottleneck-friendly: phase_start/phase_end record durations; the summary
  tool aggregates them and reports the dominant phases.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from noraivf.logging.env import capture_environment, fingerprint_file

#: Event types understood by the summary tool.
EVENT_TYPES = {
    "run_start",
    "run_end",
    "phase_start",
    "phase_end",
    "checkpoint",
    "metric",
    "info",
    "warn",
    "error",
}

_DEFAULT_OUT_DIR = Path("results/logs")


def _sanitize(value: Any) -> Any:
    """Make a payload JSON-safe: numpy scalars/arrays, Paths, non-finite floats."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    # numpy types (avoid importing numpy at module top; lazy check by module name)
    mod = type(value).__module__
    if mod.startswith("numpy"):
        import numpy as np  # local import

        if isinstance(value, np.ndarray):
            return [_sanitize(v) for v in value.tolist()]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return None if not math.isfinite(float(value)) else float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    return str(value)


class ExperimentLogger:
    """Writes one experiment run's structured logs to ``out_dir/<run_id>/``.

    Thread-safe: all mutations and writes are serialized by an internal lock.
    """

    def __init__(
        self,
        experiment_id: str,
        config: dict | None = None,
        out_dir: str | Path = _DEFAULT_OUT_DIR,
        run_id: str | None = None,
        repo_root: str | Path | None = None,
        fsync_on_error: bool = True,
    ) -> None:
        self.experiment_id = experiment_id
        self.run_id = run_id or uuid.uuid4().hex[:16]
        self.run_dir = Path(out_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self._config = _sanitize(config or {})
        self._env = capture_environment(repo_root)
        self._provenance: list[dict] = []
        self._started_at = _now_iso()
        self._closed = False
        self._lock = threading.Lock()
        self._phase_stack: list[tuple[str, float]] = []  # (phase, start_ts_epoch)

        self._run_fh = (self.run_dir / "run.jsonl").open("a", encoding="utf-8")
        self._error_fh = (self.run_dir / "errors.jsonl").open("a", encoding="utf-8")
        self._fsync_on_error = fsync_on_error

        self._write_meta()
        self._event("run_start", phase=None, message="run started")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def event(self, event_type: str, phase: str | None = None, **fields: Any) -> None:
        """Low-level event emission. Unknown event types are recorded as-is."""
        if event_type not in EVENT_TYPES:
            event_type = f"custom:{event_type}"
        self._event(event_type, phase=phase, **fields)

    def phase_start(self, phase: str, **fields: Any) -> None:
        self._event("phase_start", phase=phase, **fields)

    def phase_end(self, phase: str, metrics: dict | None = None, **fields: Any) -> None:
        self._event("phase_end", phase=phase, metrics=metrics, **fields)

    @contextmanager
    def phase(self, phase: str, **start_fields: Any) -> Iterator["ExperimentLogger"]:
        """Context manager: emit phase_start on entry, phase_end (with duration) on exit."""
        self.phase_start(phase, **start_fields)
        t0 = time.monotonic()
        try:
            yield self
        finally:
            self.phase_end(phase, metrics={"phase_duration_s": time.monotonic() - t0})

    def checkpoint(self, phase: str, **fields: Any) -> None:
        self._event("checkpoint", phase=phase, **fields)

    def metric(self, phase: str | None, **values: Any) -> None:
        """Record one metric event; pass scalar values as keyword arguments."""
        self._event("metric", phase=phase, metrics=_sanitize(values))

    def info(self, message: str, phase: str | None = None, **fields: Any) -> None:
        self._event("info", phase=phase, message=message, **fields)

    def warn(self, message: str, phase: str | None = None, **fields: Any) -> None:
        self._event("warn", phase=phase, message=message, **fields)

    def error(self, exc: BaseException | str, phase: str | None = None, **fields: Any) -> None:
        """Log an error (stack trace if an exception is given) and fsync errors.jsonl."""
        if isinstance(exc, BaseException):
            payload = {
                "type": type(exc).__name__,
                "message": str(exc) or type(exc).__name__,
                "stack": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }
        else:
            payload = {"type": "Error", "message": str(exc), "stack": None}
        payload.update(fields)
        self._event("error", phase=phase, error=payload)
        if self._fsync_on_error:
            with self._lock:
                os.fsync(self._error_fh.fileno())

    def add_data_provenance(self, path: str | Path) -> dict:
        """Fingerprint a data file and record it in meta.json. Returns the fingerprint."""
        fp = fingerprint_file(path)
        with self._lock:
            self._provenance.append(fp)
            self._write_meta_locked()
        return fp

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._event_locked(
                "run_end", phase=None, message="run ended",
                ended_at=_now_iso(),
            )
            self._run_fh.flush()
            os.fsync(self._run_fh.fileno())
            self._error_fh.flush()
            self._run_fh.close()
            self._error_fh.close()
            self._write_meta_locked(ended_at=_now_iso())
            self._closed = True

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.error(exc)
        self.close()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _event(self, event_type: str, phase: str | None, **fields: Any) -> None:
        with self._lock:
            self._event_locked(event_type, phase=phase, **fields)

    def _event_locked(self, event_type: str, phase: str | None, **fields: Any) -> None:
        if self._closed:
            raise RuntimeError(f"logger {self.run_id} is closed")
        record = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "ts": _now_iso(),
            "event_type": event_type,
            "phase": phase,
        }
        record.update(_sanitize(fields))
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self._run_fh.write(line)
        self._run_fh.flush()
        if event_type == "error":
            self._error_fh.write(line)
            self._error_fh.flush()

    def _write_meta(self) -> None:
        with self._lock:
            self._write_meta_locked()

    def _write_meta_locked(self, ended_at: str | None = None) -> None:
        meta = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "env": self._env,
            "config": self._config,
            "data_provenance": self._provenance,
            "files": {
                "run_jsonl": str(self.run_dir / "run.jsonl"),
                "errors_jsonl": str(self.run_dir / "errors.jsonl"),
            },
        }
        tmp = self.run_dir / "meta.json.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.run_dir / "meta.json")


class NullLogger:
    """No-op logger implementing the ExperimentLogger hook API.

    Used as the default for experiment hooks so existing code paths run
    unchanged when no logger is configured (see experiments hooks).
    """

    run_id = None
    experiment_id = None

    def event(self, event_type, phase=None, **fields):
        pass

    def phase_start(self, phase, **fields):
        pass

    def phase_end(self, phase, metrics=None, **fields):
        pass

    @contextmanager
    def phase(self, phase, **start_fields):
        yield self

    def checkpoint(self, phase, **fields):
        pass

    def metric(self, phase, **values):
        pass

    def info(self, message, phase=None, **fields):
        pass

    def warn(self, message, phase=None, **fields):
        pass

    def error(self, exc, phase=None, **fields):
        pass

    def add_data_provenance(self, path):
        return None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
