#!/usr/bin/env python
"""Standalone probe: verify TFRecord + SequenceExample parsing on one shard.

Prints the first record's structure (context fields, audio_embedding frame
shape/count), CRC checks, and per-shard totals.

Usage::

    python scripts/probe_audioset_tfrecord.py /path/to/bal_train/00.tfrecord
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_audioset import (  # noqa: E402
    dequantize_frame,
    get_labels,
    iter_tfrecord_records,
    parse_audio_embedding,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: probe_audioset_tfrecord.py <file.tfrecord>")
    path = Path(sys.argv[1])
    print(f"probe: {path} ({path.stat().st_size} bytes)")

    n_records = 0
    n_frames = 0
    for i, (payload, len_ok, pay_ok) in enumerate(iter_tfrecord_records(path)):
        n_records += 1
        frames = parse_audio_embedding(payload)
        n_frames += len(frames)
        if i == 0:
            print(f"  record 0: payload {len(payload)} bytes, "
                  f"length_CRC_ok={len_ok}, payload_CRC_ok={pay_ok}")
            if frames:
                shapes = {f.shape for f in frames}
                print(f"  record 0: {len(frames)} frames, frame shapes: {shapes} "
                      f"(uint8; dequantized dim = byte - 128)")
                y = dequantize_frame(frames[0])
                print(f"  record 0: frame 0 dequantized[:8] = {y[:8]}")
                print(f"  record 0: frame 0 byte min/max = {int(frames[0].min())}/{int(frames[0].max())}")
            else:
                print("  record 0: NO audio_embedding frames found")
            print(f"  record 0: labels = {get_labels(payload)}")

    print(f"  total: {n_records} records, {n_frames} frames "
          f"({n_frames / max(n_records, 1):.1f} frames/record)")
    assert n_records > 0, "no records parsed — check format"
    assert n_frames > 0, "no audio_embedding frames — check protobuf parser"
    print("probe OK")


if __name__ == "__main__":
    main()
