#!/usr/bin/env python3
"""Convert the bigann .bin files into the npy layout NoraIVF's loader expects.

Formats (little endian, per the VDB_Benchmark/Nanvivi SIFT1B cards):
  base.bin : [uint32 n][uint32 d][uint8  vectors]   n*d bytes of payload
  query.bin: [uint32 q][uint32 d][uint8  vectors]
  gt.bin   : [uint32 q][uint32 k][uint32 ids]

Writes <out>/train.npy, queries.npy, gt.npy, meta.json.

train.npy is stored as float16.  SIFT components are integers in 0..255 and
float16 represents every integer up to 2048 exactly, so this conversion is
lossless; it halves the on-disk footprint against float32 (256 GB vs 512 GB),
and disk -- not memory -- is the binding constraint on this machine.
NoraIVF's loader does np.asarray(np.load(...), dtype=np.float32), so it reads
this back as float32 with no change on its side.

Streams in chunks so peak RSS stays small.
"""
import argparse
import json
import os

import numpy as np

CHUNK = 5_000_000          # vectors per chunk


def read_header(f):
    n, d = np.fromfile(f, dtype=np.uint32, count=2)
    return int(n), int(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--query', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='float16',
                    choices=['float16', 'float32'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.base, 'rb') as f:
        n, d = read_header(f)
        print(f'base: n={n:,} d={d} -> {a.dtype}', flush=True)
        tr = np.lib.format.open_memmap(
            os.path.join(a.out, 'train.npy'), mode='w+',
            dtype=np.dtype(a.dtype), shape=(n, d))
        done = 0
        while done < n:
            m = min(CHUNK, n - done)
            buf = np.fromfile(f, dtype=np.uint8, count=m * d)
            if buf.size != m * d:
                raise RuntimeError(f'truncated base at {done}')
            tr[done:done + m] = buf.reshape(m, d)
            done += m
            print(f'  train {done:,}/{n:,} ({100.0 * done / n:.1f}%)',
                  flush=True)
        tr.flush()
        del tr
        del buf

    with open(a.query, 'rb') as f:
        q, dq = read_header(f)
        assert dq == d, (dq, d)
        arr = np.fromfile(f, dtype=np.uint8, count=q * dq).reshape(q, dq)
    np.save(os.path.join(a.out, 'queries.npy'), arr.astype(np.float32))

    with open(a.gt, 'rb') as f:
        gq, k = read_header(f)
        g = np.fromfile(f, dtype=np.uint32, count=gq * k).reshape(gq, k)
    np.save(os.path.join(a.out, 'gt.npy'), g.astype(np.int32))

    with open(os.path.join(a.out, 'meta.json'), 'w') as f:
        json.dump(dict(name='sift1b', n=n, dim=d, n_queries=q, gt_k=k,
                       train_dtype=a.dtype,
                       source='VDB_Benchmark/sift1b (hf-mirror mirror)',
                       note='float16 is lossless: SIFT components are 0..255'),
                  f, indent=2)
    print(f'done: train({n}x{d}) queries({q}) gt({gq}x{k})', flush=True)


if __name__ == '__main__':
    main()
