"""Unit tests for the logging module (run: python tests/test_logging.py).

Covers: run-dir layout, event stream integrity, phase durations,
metric aggregation, error triage, data provenance, crash-safe flush,
thread safety, and the summary pipeline.
"""

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from noraivf.logging import (
    ExperimentLogger,
    format_summary,
    fingerprint_file,
    summarize_run,
    write_summary,
)


def _make_logger(out_dir, **kwargs):
    return ExperimentLogger(
        experiment_id="test-suite",
        config={"dataset": "synthetic", "n": 1000, "nlist": 128, "seed": 42},
        out_dir=out_dir,
        **kwargs,
    )


def test_run_dir_layout_and_meta():
    with tempfile.TemporaryDirectory() as tmp:
        with _make_logger(tmp) as logger:
            pass
        run_dir = Path(tmp) / logger.run_id
        assert run_dir.is_dir()
        assert (run_dir / "run.jsonl").exists()
        assert (run_dir / "errors.jsonl").exists()
        assert (run_dir / "meta.json").exists()

        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["run_id"] == logger.run_id
        assert meta["experiment_id"] == "test-suite"
        assert meta["config"]["nlist"] == 128
        assert meta["ended_at"] is not None
        assert meta["env"]["hostname"]
        assert meta["env"]["cpu_cores"] > 0
        assert "python" in meta["env"]


def test_event_stream_and_phase_durations():
    with tempfile.TemporaryDirectory() as tmp:
        with _make_logger(tmp) as logger:
            with logger.phase("build"):
                logger.checkpoint("build", batch=1)
                logger.checkpoint("build", batch=2)
            with logger.phase("stream"):
                logger.metric("stream", throughput=100.0)
                logger.metric("stream", throughput=200.0)

        run_dir = Path(tmp) / logger.run_id
        events = [
            json.loads(line)
            for line in (run_dir / "run.jsonl").read_text().splitlines()
            if line.strip()
        ]
        types = [ev["event_type"] for ev in events]
        assert types[0] == "run_start"
        assert types[-1] == "run_end"
        assert "phase_start" in types and "phase_end" in types
        assert sum(t == "checkpoint" for t in types) == 2
        assert sum(t == "metric" for t in types) == 2

        # each event carries identity + timestamp
        for ev in events:
            assert ev["run_id"] == logger.run_id
            assert ev["experiment_id"] == "test-suite"
            assert ev["ts"]


def test_summary_aggregation_and_bottleneck():
    import time

    with tempfile.TemporaryDirectory() as tmp:
        with _make_logger(tmp) as logger:
            with logger.phase("slow_phase"):
                time.sleep(0.05)
            with logger.phase("fast_phase"):
                pass
            logger.metric("search", recall=0.9, qps=100.0)
            logger.metric("search", recall=0.95, qps=150.0)

        summary = summarize_run(Path(tmp) / logger.run_id)
        assert summary["counts"]["by_type"]["run_start"] == 1
        assert "slow_phase" in summary["phases"]
        assert summary["metrics"]["qps"]["count"] == 2
        assert 100.0 <= summary["metrics"]["qps"]["mean"] <= 150.0
        # bottleneck list ranks phases by total time
        assert summary["bottleneck"][0]["phase"] == "slow_phase"
        assert summary["bottleneck"][0]["total_s"] >= summary["bottleneck"][1]["total_s"]


def test_errors_land_in_both_streams_with_stack():
    with tempfile.TemporaryDirectory() as tmp:
        with _make_logger(tmp) as logger:
            try:
                raise ValueError("boom")
            except ValueError as exc:
                logger.error(exc, phase="stream", batch_idx=7)

        run_dir = Path(tmp) / logger.run_id
        err_events = [
            json.loads(line)
            for line in (run_dir / "errors.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(err_events) == 1
        err = err_events[0]["error"]
        assert err["type"] == "ValueError"
        assert err["message"] == "boom"
        assert "ValueError" in err["stack"]
        assert err_events[0]["phase"] == "stream"
        assert err_events[0]["error"]["batch_idx"] == 7

        # error also present in run.jsonl and surfaced in summary
        summary = summarize_run(run_dir)
        assert len(summary["errors"]) == 1
        assert summary["errors"][0]["error"]["type"] == "ValueError"
        rendered = format_summary(summary)
        assert "ValueError" in rendered


def test_data_provenance_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "train.npy"
        data_path.write_bytes(b"\x00" * 1024 + b"\xff" * 1024)

        with _make_logger(tmp) as logger:
            fp = logger.add_data_provenance(data_path)
            assert fp["mode"] == "sha256"
            assert fp["size_bytes"] == 2048

        meta = json.loads((Path(tmp) / logger.run_id / "meta.json").read_text())
        assert len(meta["data_provenance"]) == 1
        assert meta["data_provenance"][0]["path"].endswith("train.npy")

        # fingerprint is deterministic
        fp2 = fingerprint_file(data_path)
        assert fp["digest"] == fp2["digest"]


def test_numpy_values_serialize_and_nan_sanitized():
    with tempfile.TemporaryDirectory() as tmp:
        with _make_logger(tmp) as logger:
            logger.metric("build", n_splits=np.int64(3), score=np.float32(0.5),
                          recall_nan=float("nan"))
        events = [
            json.loads(line)
            for line in (Path(tmp) / logger.run_id / "run.jsonl").read_text().splitlines()
            if line.strip()
        ]
        metric_events = [ev for ev in events if ev["event_type"] == "metric"]
        assert metric_events[0]["metrics"]["n_splits"] == 3
        assert metric_events[0]["metrics"]["score"] == 0.5
        assert metric_events[0]["metrics"]["recall_nan"] is None


def test_thread_safety_append():
    with tempfile.TemporaryDirectory() as tmp:
        with _make_logger(tmp) as logger:
            def worker(i):
                for j in range(200):
                    logger.checkpoint("stream", thread=i, step=j)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        run_dir = Path(tmp) / logger.run_id
        lines = [l for l in (run_dir / "run.jsonl").read_text().splitlines() if l.strip()]
        assert sum(1 for l in lines if '"checkpoint"' in l) == 800
        # every line must parse and be complete (no interleaved partial writes)
        for line in lines:
            ev = json.loads(line)
            assert ev["run_id"] == logger.run_id


def test_write_summary_json():
    with tempfile.TemporaryDirectory() as tmp:
        with _make_logger(tmp) as logger:
            with logger.phase("build"):
                pass
        run_dir = Path(tmp) / logger.run_id
        out = write_summary(run_dir)
        assert out.name == "events.summary.json"
        assert out.exists()
        summary = json.loads(out.read_text())
        assert "phases" in summary and "bottleneck" in summary


def test_write_after_close_raises():
    with tempfile.TemporaryDirectory() as tmp:
        logger = _make_logger(tmp)
        logger.close()
        try:
            logger.checkpoint("build", batch=1)
            raised = False
        except RuntimeError:
            raised = True
        assert raised


if __name__ == "__main__":
    test_run_dir_layout_and_meta()
    test_event_stream_and_phase_durations()
    test_summary_aggregation_and_bottleneck()
    test_errors_land_in_both_streams_with_stack()
    test_data_provenance_fingerprint()
    test_numpy_values_serialize_and_nan_sanitized()
    test_thread_safety_append()
    test_write_summary_json()
    test_write_after_close_raises()
    print("logging tests passed")
