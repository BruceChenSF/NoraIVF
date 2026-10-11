#!/usr/bin/env python
"""Data pipeline: convert the AudioSet VGGish embeddings feature pack
(``data/raw/features.tar.gz``) into the canonical NoraIVF dataset layout
``data/annbench/audioset-vggish-128/``.

Input (two mutually exclusive modes):
  * --tar: the official AudioSet "TensorFlow Feature Vectors (VGGish)"
    download tar.gz; the tar is streamed member-by-member and only
    ``audioset_v1_embeddings/bal_train/*.tfrecord`` members are converted.
  * --dir: a directory of *.tfrecord shards (recursively found, e.g. an
    extracted ``unbal_train/`` or ``eval/`` split); every shard is converted.

Output ``data/annbench/audioset-vggish-128/``::

    train.npy        # (N, 128) float32, one vector per bal_train segment
    queries.npy      # (1000, 128) float32, seed=42 random sample from train
    gt.npy           # (1000, 100) int32, brute-force IP top-100
    distances.npy    # (1000, 100) float32, brute-force IP values (sorted desc)
    meta.json        # provenance: source URL + sha256, dequantization, shapes

Pipeline (per TFRecord segment):
  1. parse the TFRecord container ([uint64 len][uint32 len-crc][payload]
     [uint32 payload-crc]; CRC check optional, skipped by default)
  2. hand-rolled protobuf wire parsing of the tf.train.SequenceExample:
     context (field 1) -> video_id (bytes); feature_lists (field 2) ->
     'audio_embedding' -> one uint8[128] frame per entry (frame = 1s VGGish)
  3. dequantize each frame: x = v/127.5 - 1.0  (see DEQUANTIZATION note)
  4. aggregate the segment's frames by mean -> 1 x 128 vector
  5. L2-normalize (all NoraIVF runs use IP on normalized vectors; squared-L2
     ordering on normalized vectors == IP ordering, so GT is computed by IP)

The tar is streamed member-by-member (tarfile ``r|gz``); nothing is
extracted to disk and non-bal_train members are drained without buffering.

DEQUANTIZATION note (choice documented; no ANN-benchmark impact):
  The official postprocessor (vggish/vggish_postprocess.py) applies PCA with
  whitening, clips to [-2.0, +2.0], then maps to uint8 via
  ``(x + 2.0) * 63.75``.  Its exact inverse (up to uint8 rounding) is
  ``v/63.75 - 2.0``.  We instead use the symmetric affine
  ``x = v/127.5 - 1.0``, which maps [0, 255] -> [-1, 1].  Note that
  ``v/127.5 - 1.0 == 0.5 * (v/63.75 - 2.0)``: the two formulas differ by a
  pure global rescale, which L2-normalization cancels exactly, so the
  normalized segment vectors are identical either way.  The symmetric
  choice is retained for clarity and documented here.

Usage::

    python scripts/prepare_audioset_vggish.py --tar data/raw/features.tar.gz
    python scripts/prepare_audioset_vggish.py \\
        --dir /path/to/audioset_v1_embeddings/unbal_train \\
        --name audioset-unbal-128

    # debug: stop after 100k segments, 50 queries, k=10
    python scripts/prepare_audioset_vggish.py --tar data/raw/features.tar.gz \\
        --max-segments 100000 --queries 50 --gt-k 10

    # inspect the first 3 records of the first shard (tar or dir mode)
    python scripts/prepare_audioset_vggish.py --tar data/raw/features.tar.gz --inspect
    python scripts/prepare_audioset_vggish.py --dir <unbal_train> --inspect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

SOURCE_URL = (
    "https://research.google.com/audioset/download.html "
    "(TensorFlow Feature Vectors (VGGish) section)"
)

# ---------------------------------------------------------------------------
# TFRecord container parsing
# ---------------------------------------------------------------------------
# TFRecord: uint64 length (little-endian) | uint32 masked-CRC32C(length) |
#           payload | uint32 masked-CRC32C(payload)
# CRC is CRC-32C (Castagnoli, reflected, poly 0x1EDC6F41) with TensorFlow's
# mask: masked = ((crc >> 15) | (crc << 17)) + 0xa282ead8 (mod 2^32).

_CRC32C_TABLE: list[int] = []


def _build_crc32c_table() -> list[int]:
    poly = 0x82F63B78  # reflected CRC-32C polynomial (Castagnoli)
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ poly if crc & 1 else crc >> 1
        table.append(crc)
    return table


def crc32c(data: bytes, crc: int = 0) -> int:
    """CRC-32C (Castagnoli) of *data*, little-endian per TFRecord spec."""
    global _CRC32C_TABLE
    if not _CRC32C_TABLE:
        _CRC32C_TABLE = _build_crc32c_table()
    crc ^= 0xFFFFFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC32C_TABLE[(crc ^ b) & 0xFF]
    return crc ^ 0xFFFFFFFF


def _mask_crc(crc: int) -> int:
    """TensorFlow masked CRC: rotate right 15, add 0xa282ead8 (mod 2^32)."""
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecord_records(payload_stream, path_name: str, verify_crc: bool = False):
    """Yield (payload, length_crc_ok, payload_crc_ok) per TFRecord.

    *payload_stream* is a file-like object positioned at the first record.
    CRC verification is off by default (the source is a trusted official
    download; skipping it avoids hashing ~40M frames); pass ``verify_crc=True``
    to check every record against its CRC-32C.
    """
    while True:
        header = payload_stream.read(8)
        if not header:
            return
        if len(header) < 8:
            raise ValueError(f"{path_name}: truncated TFRecord header")
        length = int.from_bytes(header, "little")
        len_crc = payload_stream.read(4)
        payload = payload_stream.read(length)
        pay_crc = payload_stream.read(4)
        if len(payload) < length:
            raise ValueError(f"{path_name}: truncated TFRecord payload")
        if verify_crc:
            len_crc_ok = _mask_crc(crc32c(header)) == int.from_bytes(len_crc, "little")
            pay_crc_ok = _mask_crc(crc32c(payload)) == int.from_bytes(pay_crc, "little")
            if not (len_crc_ok and pay_crc_ok):
                raise ValueError(
                    f"{path_name}: CRC mismatch (len_ok={len_crc_ok}, payload_ok={pay_crc_ok})"
                )
        else:
            len_crc_ok = pay_crc_ok = None
        yield payload, len_crc_ok, pay_crc_ok


# ---------------------------------------------------------------------------
# Minimal protobuf wire parser (tf.train.SequenceExample subset)
# ---------------------------------------------------------------------------
# SequenceExample { Features context = 1; FeatureLists feature_lists = 2; }
# Features        { map<string, Feature> feature = 1; }
# FeatureLists    { map<string, FeatureList> feature_list = 1; }
#   map entry     { key = 1 (string); value = 2 (value message) }
# FeatureList     { repeated Feature feature = 1; }
# Feature         { oneof: bytes_list = 1 | float_list = 2 | int64_list = 3 }
# BytesList       { repeated bytes value = 1; }
# FloatList       { repeated float value = 1 [packed]; }
# Int64List       { repeated int64 value = 1 [packed]; }

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    result = 0
    while True:
        if pos >= len(buf):
            raise ValueError("varint past end of buffer")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def iter_proto_fields(buf: bytes):
    """Yield (field_number, wire_type, value) for a protobuf message.
    wire_type 0 -> int; 1/5 -> raw bytes; 2 -> length-delimited bytes."""
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_no, wt = key >> 3, key & 7
        if wt == 0:
            val, pos = _read_varint(buf, pos)
        elif wt == 1:
            if pos + 8 > end:
                raise ValueError("truncated 64-bit field")
            val, pos = buf[pos:pos + 8], pos + 8
        elif wt == 2:
            ln, pos = _read_varint(buf, pos)
            if pos + ln > end:
                raise ValueError("truncated length-delimited field")
            val, pos = buf[pos:pos + ln], pos + ln
        elif wt == 5:
            if pos + 4 > end:
                raise ValueError("truncated 32-bit field")
            val, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield field_no, wt, val


def _map_value(features_msg: bytes, wanted_key: str):
    """Inside a Features/FeatureLists map message, return the value message
    whose key == *wanted_key*, or None."""
    for fn, wt, val in iter_proto_fields(features_msg):
        if fn == 1 and wt == 2:  # map entry
            key = None
            value = None
            for fn2, wt2, val2 in iter_proto_fields(val):
                if fn2 == 1 and wt2 == 2:
                    key = val2.decode("utf-8", "replace")
                elif fn2 == 2 and wt2 == 2:
                    value = val2
            if key == wanted_key and value is not None:
                return value
    return None


def extract_video_id(payload: bytes) -> str | None:
    """SequenceExample context -> Features map -> 'video_id' -> first bytes
    value (the YouTube video id of the segment)."""
    for fn, wt, val in iter_proto_fields(payload):
        if fn == 1 and wt == 2:  # context Features
            feat = _map_value(val, "video_id")
            if feat is None:
                continue
            for fn2, wt2, val2 in iter_proto_fields(feat):
                if fn2 == 1 and wt2 == 2:  # bytes_list
                    for fn3, wt3, val3 in iter_proto_fields(val2):
                        if fn3 == 1 and wt3 == 2:  # bytes value
                            return val3.decode("utf-8", "replace")
    return None


def parse_audio_embedding(payload: bytes):
    """Extract ``audio_embedding`` frames from a SequenceExample payload.

    Returns a list of uint8 ndarrays, one per frame (128 dims = 128 bytes
    in this 8-bit quantized pack).
    """
    frames: list[np.ndarray] = []
    for fn, wt, val in iter_proto_fields(payload):
        if fn == 2 and wt == 2:  # feature_lists
            feature_list = _map_value(val, "audio_embedding")
            if feature_list is None:
                continue
            for fn2, wt2, val2 in iter_proto_fields(feature_list):
                if fn2 == 1 and wt2 == 2:  # Feature
                    bytes_list = None
                    for fn3, wt3, val3 in iter_proto_fields(val2):
                        if fn3 == 1 and wt3 == 2:  # bytes_list
                            bytes_list = val3
                    if bytes_list is not None:
                        for fn4, wt4, val4 in iter_proto_fields(bytes_list):
                            if fn4 == 1 and wt4 == 2:  # bytes value
                                frames.append(np.frombuffer(val4, dtype=np.uint8))
    return frames


def dequantize_frame(frame_uint8: np.ndarray) -> np.ndarray:
    """uint8 quantized VGGish embedding -> float32, symmetric affine
    ``x = v/127.5 - 1.0`` (maps [0, 255] -> [-1, 1]; see module docstring)."""
    return frame_uint8.astype(np.float32) / 127.5 - 1.0


def aggregate_segment(frames: list[np.ndarray]) -> np.ndarray | None:
    """Mean-aggregate a segment's dequantized frames, then L2-normalize.

    Returns a (128,) float32 vector, or None when the segment has no frames
    or a zero-norm aggregate.
    """
    if not frames:
        return None
    stacked = np.stack([dequantize_frame(f) for f in frames], axis=0)  # (n, 128)
    vec = stacked.mean(axis=0)
    nrm = np.linalg.norm(vec)
    if nrm == 0.0:
        return None
    return (vec / nrm).astype(np.float32)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _sha256(path: Path, log=print) -> str:
    log(f"  sha256({path.name}) ...", flush=True)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_bal_train_members(tar_path: Path):
    """Stream the tar; yield (member_name, stream) for every
    bal_train/*.tfrecord member.  Non-matching members are drained (the tar
    is read once, sequentially)."""
    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            if member.isdir() or not member.name.endswith(".tfrecord"):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            if "/bal_train/" in member.name:
                yield member.name, fobj
            else:
                while fobj.read(1 << 20):
                    pass


def iter_dir_tfrecords(dir_path: Path):
    """Recursively find *.tfrecord under *dir_path* (sorted); yield
    (relative_name, open binary stream) per file.  The stream stays open
    while the caller consumes it (one shard at a time)."""
    files = sorted(dir_path.rglob("*.tfrecord"))
    if not files:
        raise SystemExit(f"no *.tfrecord files found under {dir_path}")

    def _gen():
        for f in files:
            with f.open("rb") as fh:
                yield f.relative_to(dir_path).as_posix(), fh

    return _gen()


def inspect_tar(tar_path: Path, n_records: int = 3, member_override: Path | None = None) -> None:
    """Print the first records of one shard: video_id and the first 8 dims
    of the aggregated (dequantized, L2-normalized) vector."""
    if member_override is not None:
        print(f"inspect file: {member_override}")
        with member_override.open("rb") as fh:
            _inspect_stream(fh, str(member_override), n_records)
        return
    print(f"inspect tar: {tar_path}")
    shown = 0
    for member_name, stream in iter_bal_train_members(tar_path):
        print(f"  shard: {member_name}")
        shown = _inspect_stream(stream, member_name, n_records)
        if shown >= n_records:
            return
    if shown == 0:
        raise SystemExit(f"no bal_train/*.tfrecord members found in {tar_path}")


def inspect_dir(dir_path: Path, n_records: int = 3) -> None:
    """Print the first records of the first *.tfrecord under *dir_path*."""
    files = sorted(dir_path.rglob("*.tfrecord"))
    if not files:
        raise SystemExit(f"no *.tfrecord files found under {dir_path}")
    print(f"inspect dir: {dir_path} ({len(files)} shards)")
    shown = 0
    for f in files:
        print(f"  shard: {f.relative_to(dir_path)} ({f.stat().st_size} bytes)")
        with f.open("rb") as fh:
            shown = _inspect_stream(fh, str(f), n_records)
        if shown >= n_records:
            return


def _inspect_stream(stream, name: str, n_records: int) -> int:
    shown = 0
    for i, (payload, _lo, _po) in enumerate(iter_tfrecord_records(stream, name)):
        frames = parse_audio_embedding(payload)
        vec = aggregate_segment(frames)
        vid = extract_video_id(payload)
        dims = "n/a" if vec is None else np.round(vec[:8], 6).tolist()
        print(f"  record {i}: video_id={vid!r} frames={len(frames)} "
              f"aggregated_vec[:8]={dims}")
        shown += 1
        if shown >= n_records:
            return shown
    return shown


def compute_ip_ground_truth(train: np.ndarray, queries: np.ndarray, k: int,
                            batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Blocked brute-force inner-product top-k on L2-normalized vectors.

    Returns (gt int32 (Q, k), distances float32 (Q, k) sorted descending).
    """
    nq = len(queries)
    gt = np.empty((nq, k), dtype=np.int32)
    dists = np.empty((nq, k), dtype=np.float32)
    train_t = train.T.astype(np.float32)  # (128, n); transposed once
    for i in range(0, nq, batch):
        sim = queries[i:i + batch].astype(np.float32) @ train_t  # (b, n)
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sim, top, axis=1)
        order = np.argsort(-vals, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        gt[i:i + batch] = top.astype(np.int32)
        dists[i:i + batch] = vals.astype(np.float32)
    return gt, dists


def convert(shards, out_dir: Path, source: dict, hash_path: Path | None = None,
            n_queries: int = 1000, seed: int = 42, gt_k: int = 100,
            max_segments: int | None = None, log=print) -> dict:
    """Convert *shards* (iterator of (name, stream) TFRecord shards) into
    the canonical dataset layout under *out_dir*.

    *source* becomes the ``meta.json`` "source" object (sha256/n_shards are
    filled in here); *hash_path* is the file to sha256 for provenance (the
    tar in tar mode, None in dir mode).
    """
    n_shards = 0
    n_records = 0
    n_frames_total = 0
    n_segments = 0
    vecs_list: list[np.ndarray] = []

    for member_name, stream in shards:
        n_shards += 1
        shard_vecs = []
        for payload, _lo, _po in iter_tfrecord_records(stream, member_name):
            n_records += 1
            frames = parse_audio_embedding(payload)
            n_frames_total += len(frames)
            vec = aggregate_segment(frames)
            if vec is None:
                continue
            shard_vecs.append(vec)
        vecs_list.extend(shard_vecs)
        n_segments += len(shard_vecs)
        if n_shards % 500 == 0 or max_segments and n_segments >= max_segments:
            log(f"  {member_name}: {len(shard_vecs)} vectors "
                f"(cumulative {n_segments})", flush=True)
        if max_segments is not None and n_segments >= max_segments:
            log(f"  --max-segments reached at {n_segments} (after {n_shards} shards)")
            break

    if n_segments == 0:
        raise SystemExit("no segment vectors produced")

    train = np.asarray(vecs_list, dtype=np.float32)
    del vecs_list
    n, d = train.shape
    log(f"converted: {n} segments x {d} dims "
        f"({n_shards} shards, {n_records} records, {n_frames_total} frames)")

    n_queries = min(n_queries, n)
    rng = np.random.default_rng(seed)
    qidx = rng.choice(n, size=n_queries, replace=False)
    queries = train[qidx]

    log(f"exact IP top-{gt_k} for {n_queries} queries (blocked matmul) ...", flush=True)
    gt, distances = compute_ip_ground_truth(train, queries, gt_k)
    log(f"gt done: {gt.shape}, distances range [{distances.min():.4f}, {distances.max():.4f}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train)
    np.save(out_dir / "queries.npy", queries)
    np.save(out_dir / "gt.npy", gt)
    np.save(out_dir / "distances.npy", distances)

    digest = _sha256(hash_path, log=log) if hash_path is not None else None

    source = dict(source)
    source["sha256"] = digest
    source["n_shards"] = n_shards

    meta = {
        "name": out_dir.name,
        "description": (
            "AudioSet V1 VGGish audio embeddings, one 128-d vector per "
            "segment (mean over the segment's 1s frames, uint8-dequantized, "
            "L2-normalized). IP metric on normalized vectors "
            "(== squared-L2 ordering)."
        ),
        "source": source,
        "dequantization": {
            "quantized_dtype": "uint8",
            "official_quantizer": (
                "vggish/vggish_postprocess.py: PCA(+whitening) -> clip "
                "[-2.0, +2.0] -> (x + 2.0) * 63.75 cast to uint8"
            ),
            "inverse_used": "x = v/127.5 - 1.0  (symmetric affine, maps [0,255] -> [-1,1])",
            "note": (
                "x = v/127.5 - 1.0 == 0.5 * (v/63.75 - 2.0): exactly half the "
                "official inverse; the difference is a pure global rescale that "
                "L2-normalization cancels, so normalized output is identical."
            ),
        },
        "aggregation": {
            "frames_per_segment": "variable (10 for full 10s segments)",
            "method": "mean over dequantized frames, then L2-normalize",
        },
        "metric": "ip",
        "gt_basis": "inner_product",
        "normalized": True,
        "shape": {
            "train": list(train.shape),
            "queries": list(queries.shape),
            "gt": list(gt.shape),
            "distances": list(distances.shape),
        },
        "dtype": "float32",
        "gt_k": gt_k,
        "query_sample": {
            "n": int(n_queries),
            "seed": int(seed),
            "replace": False,
            "source": "train (sampled from the full segment set)",
        },
        "gt_method": "blocked numpy inner-product top-k (argpartition) on L2-normalized vectors",
        "files": {
            "train": "train.npy",
            "queries": "queries.npy",
            "gt": "gt.npy",
            "distances": "distances.npy",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/prepare_audioset_vggish.py",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # spot-check round-trip through the written npy files
    for arr_name, arr in (("train", train), ("queries", queries), ("gt", gt)):
        reloaded = np.load(out_dir / f"{arr_name}.npy")
        assert reloaded.shape == arr.shape
        assert np.array_equal(reloaded[:3, :8], arr[:3, :8]), f"{arr_name} mismatch"
    # sanity: GT indices in range, IP values descending per row
    assert int(gt.max()) < n
    assert np.all(distances[:, :-1] >= distances[:, 1:] - 1e-6)

    log(f"wrote {out_dir} (train {n}x{d}, queries {n_queries}x{d}, gt top-{gt_k})")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--tar", type=Path, default=Path("data/raw/features.tar.gz"),
                     help="AudioSet VGGish features pack tar.gz (default: data/raw/features.tar.gz)")
    src.add_argument("--dir", type=Path, default=None,
                     help="directory of *.tfrecord shards (recursive); mutually "
                          "exclusive with --tar; e.g. an extracted unbal_train/ split")
    parser.add_argument("--out-root", type=Path, default=Path("data/annbench"),
                        help="output root (default: data/annbench); dataset dir = <out-root>/<name>")
    parser.add_argument("--name", type=str, default="audioset-vggish-128",
                        help="dataset dir name under --out-root (default: audioset-vggish-128)")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="stop after this many segment vectors (debug)")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--inspect", action="store_true",
                        help="print the first 3 records of the first shard "
                             "(video_id + aggregated vector[:8]) and exit")
    parser.add_argument("--inspect-file", type=Path, default=None,
                        help="with --inspect (tar mode): use this .tfrecord file instead of the tar")
    parser.add_argument("--verify-crc", action="store_true",
                        help="verify every TFRecord CRC-32C (default: skip)")
    args = parser.parse_args()

    if args.inspect:
        if args.dir is not None:
            inspect_dir(args.dir, n_records=3)
        else:
            inspect_tar(args.tar, n_records=3, member_override=args.inspect_file)
        return

    out_dir = args.out_root / args.name

    if args.dir is not None:
        shards = iter_dir_tfrecords(args.dir)
        source = {
            "kind": "dir",
            "url": SOURCE_URL,
            "file": str(args.dir.resolve()),
            "size_bytes": None,
            "split": args.dir.name,
        }
        hash_path = None
    else:
        tar_path = args.tar
        if not tar_path.exists():
            raise SystemExit(f"tar not found: {tar_path}")
        shards = iter_bal_train_members(tar_path)
        source = {
            "kind": "tar",
            "url": SOURCE_URL,
            "file": str(tar_path.resolve()),
            "size_bytes": tar_path.stat().st_size,
            "split": "bal_train",
        }
        hash_path = tar_path

    convert(
        shards,
        out_dir,
        source,
        hash_path=hash_path,
        n_queries=args.queries,
        seed=args.seed,
        gt_k=args.gt_k,
        max_segments=args.max_segments,
    )


if __name__ == "__main__":
    main()
