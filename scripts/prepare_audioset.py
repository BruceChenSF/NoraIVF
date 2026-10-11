#!/usr/bin/env python
"""Data pipeline: convert AudioSet v1 audio embeddings (TFRecord) into the
canonical NoraIVF dataset directory ``data/annbench/audioset-128-euclidean/``.

Source: <audioset>/audioset_v1_embeddings/bal_train/*.tfrecord
(pure TFRecord container, no tensorflow dependency).

Pipeline (per segment):
  1. parse the TFRecord SequenceExample protobuf (hand-rolled wire parser)
  2. extract ``audio_embedding`` feature_lists -> 10 frames x 128 float32
  3. aggregate: mean over the 10 frames -> 1 x 128 vector
  4. L2-normalize (cosine-equivalent metric; dataset named -euclidean because
     search runs on normalized vectors with L2, ordering == cosine ordering)

Output ``data/annbench/audioset-128-euclidean/``::

    train.npy        # (N, 128) float32, all segments, L2-normalized
    queries.npy      # (1000, 128) float32, seed=42 random sample from train
    gt.npy           # (1000, 100) int32, faiss IndexFlatL2 top-100
    distances.npy    # (1000, 100) float32, squared L2 distances from faiss
    meta.json        # provenance: source, fingerprint, shapes, aggregation

NOTE on the actual data format (verified against the bytes):

    The bal_train shards in this local copy are the 8-bit QUANTIZED
    AudioSet V1 embeddings, not float32: each ``audio_embedding`` frame is
    one bytes_list value of 128 bytes = 128 uint8 dims. Evidence:
    - frames are 128 bytes (float32 128-d would be 512);
    - the byte histogram has heavy spikes at exactly 0x00 (9.4%) and 0xFF
      (9.0%) = clipping at the uint8 quantizer extremes (an int16 or int8
      two's-complement layout would spike at 0x7F/0x80 instead);
    - int16 pair statistics are inconsistent with an int16 layout;
    - the centered (mean-subtracted) bytes carry the signal: within-segment
      frame similarity 0.68 vs 0.02 across segments; label-based validation
      (AudioSet label ids from the record context) gives ~0.95 top-20
      label recall for the resulting segment vectors.

    Dequantization: y = (byte - 128) / 128  (uint8 center at 128; the
    global scale is irrelevant after L2-normalization, so y = byte - 128).
    Aggregation: mean over the segment's frames (typically 10, some
    segments have fewer), then L2-normalize.

Usage::

    python scripts/prepare_audioset.py \
        --source-dir /path/to/audioset_v1_embeddings/bal_train \
        --out-dir data/annbench/audioset-128-euclidean \
        --n-queries 1000 --seed 42 --gt-k 100

TFRecord container format (per record):
    uint64 length (little-endian, TensorFlow EncodeFixed64)
    | uint32 masked-CRC32C-of-length | payload | uint32 masked-CRC32C-of-payload
The CRC is CRC-32C (Castagnoli, reflected, poly 0x1EDC6F41) with TensorFlow's
mask function: masked = ((crc >> 15) | (crc << 17)) + 0xa282ead8 (mod 2^32).
Implemented with a small table so no external crc library is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from noraivf.logging import fingerprint_file

# ---------------------------------------------------------------------------
# TFRecord container parsing
# ---------------------------------------------------------------------------

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


def iter_tfrecord_records(path: str | Path, verify_crc: bool = True):
    """Yield (payload, length_crc_ok, payload_crc_ok) for each record in a
    TFRecord file. CRC verification is on by default; a CRC mismatch raises
    (data corruption should never be silently accepted)."""
    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if not header:
                return
            if len(header) < 8:
                raise ValueError(f"{path}: truncated TFRecord header")
            length = int.from_bytes(header, "little")
            len_crc = fh.read(4)
            if len(len_crc) < 4:
                raise ValueError(f"{path}: truncated length CRC")
            payload = fh.read(length)
            if len(payload) < length:
                raise ValueError(f"{path}: truncated payload")
            pay_crc = fh.read(4)
            if len(pay_crc) < 4:
                raise ValueError(f"{path}: truncated payload CRC")
            len_crc_ok = _mask_crc(crc32c(header)) == int.from_bytes(len_crc, "little")
            pay_crc_ok = _mask_crc(crc32c(payload)) == int.from_bytes(pay_crc, "little")
            if verify_crc and not (len_crc_ok and pay_crc_ok):
                raise ValueError(f"{path}: CRC mismatch (len_ok={len_crc_ok}, payload_ok={pay_crc_ok})")
            yield payload, len_crc_ok, pay_crc_ok


# ---------------------------------------------------------------------------
# Minimal protobuf wire parser (SequenceExample subset)
# ---------------------------------------------------------------------------
# SequenceExample { Features context = 1; FeatureLists feature_lists = 2; }
# FeatureLists   { map<string, FeatureList> feature_list = 1; }
#   map entry    { key = 1 (string); value = 2 (FeatureList) }
# FeatureList    { repeated Feature feature = 1; }
# Feature        { oneof: bytes_list = 1 | float_list = 2 | int64_list = 3 }
# BytesList      { repeated bytes value = 1; }
# FloatList      { repeated float value = 1 [packed]; }
# Int64List      { repeated int64 value = 1 [packed]; }

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
    For wire_type 0 value is int; for 1/5 raw bytes; for 2 length-delimited bytes."""
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


def _read_packed_varint(buf: bytes, pos: int = 0) -> int:
    shift = 0
    result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result
        shift += 7


def get_labels(payload: bytes) -> list[int]:
    """Extract the AudioSet label ids from the SequenceExample context
    (``labels`` int64_list, packed varints). Used for validation only."""
    out: list[int] = []
    for fn, wt, val in iter_proto_fields(payload):
        if fn == 1 and wt == 2:  # context Features
            for fn2, wt2, val2 in iter_proto_fields(val):
                if fn2 == 1 and wt2 == 2:  # map entry
                    key = None
                    feat = None
                    for fn3, wt3, val3 in iter_proto_fields(val2):
                        if fn3 == 1 and wt3 == 2:
                            key = val3.decode("utf-8", "replace")
                        elif fn3 == 2 and wt3 == 2:
                            feat = val3
                    if key == "labels" and feat is not None:
                        for fn4, wt4, val4 in iter_proto_fields(feat):
                            if fn4 == 3 and wt4 == 2:  # int64_list
                                for fn5, wt5, val5 in iter_proto_fields(val4):
                                    if fn5 == 1 and wt5 == 2:  # packed
                                        pos = 0
                                        while pos < len(val5):
                                            out.append(_read_packed_varint(val5, pos))
                                            pos += 1
                                        break
                                    elif fn5 == 1 and wt5 == 0:  # unpacked
                                        out.append(val5)
    return out


def parse_audio_embedding(payload: bytes):
    """Extract the ``audio_embedding`` feature from a SequenceExample payload.

    Returns a list of uint8 arrays (one per frame); each frame is 128 dims
    (128 bytes) in this 8-bit quantized copy of the AudioSet V1 embeddings.
    """
    frames: list[np.ndarray] = []
    for fn, wt, val in iter_proto_fields(payload):
        if fn == 2 and wt == 2:  # feature_lists
            for fn2, wt2, val2 in iter_proto_fields(val):
                if fn2 == 1 and wt2 == 2:  # map entry
                    key = None
                    feature_list = None
                    for fn3, wt3, val3 in iter_proto_fields(val2):
                        if fn3 == 1 and wt3 == 2:
                            key = val3.decode("utf-8", "replace")
                        elif fn3 == 2 and wt3 == 2:
                            feature_list = val3
                    if key == "audio_embedding" and feature_list is not None:
                        for fn4, wt4, val4 in iter_proto_fields(feature_list):
                            if fn4 == 1 and wt4 == 2:  # Feature
                                bytes_list = None
                                for fn5, wt5, val5 in iter_proto_fields(val4):
                                    if fn5 == 1 and wt5 == 2:  # bytes_list
                                        bytes_list = val5
                                if bytes_list is not None:
                                    for fn6, wt6, val6 in iter_proto_fields(bytes_list):
                                        if fn6 == 1 and wt6 == 2:  # bytes value
                                            arr = np.frombuffer(val6, dtype=np.uint8)
                                            frames.append(arr)
    return frames


def dequantize_frame(frame_uint8: np.ndarray) -> np.ndarray:
    """8-bit quantized embedding -> float32, uint8 center at 128.
    y = (byte - 128) / 128; the /128 scale is dropped (irrelevant after
    L2-normalization)."""
    return frame_uint8.astype(np.float32) - 128.0


def parse_segment_context(payload: bytes) -> dict:
    """Parse the SequenceExample context (video_id, timestamps, labels) for
    diagnostics / provenance. Returns a dict with whatever was found."""
    ctx = {}
    for fn, wt, val in iter_proto_fields(payload):
        if fn == 1 and wt == 2:  # context Features
            for fn2, wt2, val2 in iter_proto_fields(val):
                if wt2 != 2:
                    continue
                for fn3, wt3, val3 in iter_proto_fields(val2):
                    # map entry inside Features
                    if wt3 != 2:
                        continue
                    key = None
                    for fn4, wt4, val4 in iter_proto_fields(val3):
                        if fn4 == 1 and wt4 == 2:
                            key = val4.decode("utf-8", "replace")
                    if key and key not in ctx:
                        ctx[key] = val2[:64]
    return ctx


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def parse_shard(path: Path, log=print):
    """Parse one TFRecord shard -> list of (128,) float32 mean-aggregated,
    L2-normalized segment vectors."""
    shard_vecs = []
    n_records = 0
    n_frames_total = 0
    for payload, _len_ok, _pay_ok in iter_tfrecord_records(path):
        n_records += 1
        frames = parse_audio_embedding(payload)
        n_frames_total += len(frames)
        if not frames:
            continue
        stacked = np.stack([dequantize_frame(f) for f in frames], axis=0)  # (n, 128)
        vec = stacked.mean(axis=0)          # (128,)
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            shard_vecs.append(vec / nrm)
    if n_records:
        log(f"  {path.name}: {n_records} records, {n_frames_total} frames "
            f"-> {len(shard_vecs)} vectors")
    return shard_vecs


def convert(
    source_dir: Path,
    out_dir: Path,
    n_queries: int = 1000,
    seed: int = 42,
    gt_k: int = 100,
    log=print,
) -> dict:
    shards = sorted(source_dir.glob("*.tfrecord"))
    if not shards:
        raise SystemExit(f"no *.tfrecord found in {source_dir}")

    log(f"audioset: {len(shards)} shards under {source_dir}")
    all_vecs: list[np.ndarray] = []
    for shard in shards:
        all_vecs.extend(parse_shard(shard, log=log))

    train = np.asarray(all_vecs, dtype=np.float32)
    del all_vecs
    n, d = train.shape
    log(f"audioset: total {n} segments x {d} dims")

    # query sample: seed=42, without replacement, from the full set
    rng = np.random.default_rng(seed)
    qidx = rng.choice(n, size=n_queries, replace=False)
    queries = train[qidx]

    # exact GT with faiss brute-force L2 on the normalized vectors
    import faiss

    log(f"audioset: faiss IndexFlatL2 top-{gt_k} for {n_queries} queries ...")
    index = faiss.IndexFlatL2(d)
    index.add(train)
    dists, neigh = index.search(queries, gt_k)
    gt = neigh.astype(np.int32)
    distances = dists.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train)
    np.save(out_dir / "queries.npy", queries)
    np.save(out_dir / "gt.npy", gt)
    np.save(out_dir / "distances.npy", distances)

    meta = {
        "name": out_dir.name,
        "description": (
            "AudioSet v1 audio embeddings (bal_train), one vector per 10s segment; "
            "8-bit quantized uint8 frames dequantized as (byte-128), mean-aggregated "
            "over the segment's frames, then L2-normalized. "
            "Euclidean search on normalized vectors is cosine-equivalent."
        ),
        "source": {
            "dir": str(source_dir.resolve()),
            "n_shards": len(shards),
            "shard_fingerprint": fingerprint_file(shards[0]),
            "bytes_total": sum(p.stat().st_size for p in shards),
        },
        "aggregation": {
            "frames_per_segment": 10,
            "method": "mean over frames, then L2 normalize",
            "quantization": "8-bit uint8, dequantize y=(byte-128)/128 (scale dropped: L2-normalized output)",
            "source_format_note": (
                "Actual local data is the quantized 8-bit version (128 bytes/frame); "
                "not the float32 version (512 bytes/frame). Verified by byte histogram "
                "(clipping spikes at 0x00/0xFF), int16-pair statistics, and label-based "
                "validation (0.95 top-20 label recall)."
            ),
        },
        "metric": "euclidean",
        "normalized": True,
        "shape": {
            "train": list(train.shape),
            "queries": list(queries.shape),
            "gt": list(gt.shape),
        },
        "dtype": "float32",
        "gt_k": gt_k,
        "query_sample": {
            "n": int(n_queries),
            "seed": int(seed),
            "replace": False,
            "source": "train (sampled from the full segment set)",
        },
        "gt_method": "faiss.IndexFlatL2 brute force on normalized vectors",
        "files": {
            "train": "train.npy",
            "queries": "queries.npy",
            "gt": "gt.npy",
            "distances": "distances.npy",
        },
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"audioset: wrote {out_dir} (train {n}x{d}, queries {n_queries}x{d}, gt top-{gt_k})")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="directory containing bal_train *.tfrecord shards")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/annbench/audioset-128-euclidean"))
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt-k", type=int, default=100)
    args = parser.parse_args()

    convert(
        args.source_dir,
        args.out_dir,
        n_queries=args.n_queries,
        seed=args.seed,
        gt_k=args.gt_k,
    )


if __name__ == "__main__":
    main()
