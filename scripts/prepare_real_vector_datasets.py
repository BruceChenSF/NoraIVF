"""Prepare real/public-derived vector datasets for the L2 experiments.

Outputs one directory per dataset with train.npy, queries.npy, gt.npy, and
metadata.json. The script intentionally avoids the synthetic generator used for
early layout experiments.
"""
import sys
import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from torchvision.models import ResNet18_Weights, resnet18

from noraivf.datasets import compute_l2_ground_truth


CORE50_SHARDS = [
    "data/train-00000-of-00010-d2371d2f7481a2fc.parquet",
    "data/train-00001-of-00010-a833ce8269661791.parquet",
    "data/train-00002-of-00010-43bab836fc009709.parquet",
    "data/train-00003-of-00010-086b468b2a91e159.parquet",
    "data/train-00004-of-00010-786300e57a4667e1.parquet",
]

SCANNET_SHARDS = [
    "data/train-00000-of-00006-24970ba3dec6449d.parquet",
    "data/train-00001-of-00006-ea00006002706ec8.parquet",
    "data/train-00002-of-00006-4dd1f6f4eb9f6c1a.parquet",
    "data/train-00003-of-00006-164db95bd41bec20.parquet",
    "data/train-00004-of-00006-94061acfd36e7fb2.parquet",
    "data/train-00005-of-00006-af7dd4f9e5e9c867.parquet",
]

SCANNET_IMAGE_REPOS = [
    "fjd/scannet-processed-test",
    "YWjimmy/PeRFception-ScanNet",
]

EGO4D_CHUNKS = [
    "chunk_0.tar.gz",
    "chunk_1.tar.gz",
    "chunk_2.tar.gz",
    "chunk_3.tar.gz",
]

SCANNET_POINTS_TAR = Path("data/raw/scannet_yang/points.head.tar")


def local_hf_file(repo_id, filename):
    local = Path("data") / "raw" / "hf_parquet" / repo_id.replace("/", "__") / Path(filename).name
    if local.exists() and local.stat().st_size > 0:
        return str(local)
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def image_from_value(value):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    return None


def find_image_column(row):
    for key, value in row.items():
        img = image_from_value(value)
        if img is not None:
            return key, img
    return None, None


def load_image_encoder(device):
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return transform, model


def encode_images(images, transform, model, device):
    batch = torch.stack([transform(img) for img in images], dim=0).to(device)
    with torch.inference_mode():
        feats = model(batch)
    return normalize(feats.detach().cpu().numpy())


def parquet_image_features(repo_id, shards, n_total, batch_size, device,
                           transform=None, model=None, allow_less=False):
    if transform is None or model is None:
        transform, model = load_image_encoder(device)
    feats = []
    collected = 0
    for shard in shards:
        path = local_hf_file(repo_id, shard)
        pf = pq.ParquetFile(path)
        print(f"[{repo_id}] reading {shard}: {pf.metadata.num_rows} rows", flush=True)
        batch = []
        for record_batch in pf.iter_batches(batch_size=batch_size):
            rows = record_batch.to_pylist()
            for row in rows:
                _, img = find_image_column(row)
                if img is None:
                    continue
                batch.append(img)
                if len(batch) >= batch_size:
                    encoded = encode_images(batch, transform, model, device)
                    feats.append(encoded)
                    collected += len(encoded)
                    if collected % (batch_size * 10) == 0:
                        print(f"[{repo_id}] encoded {collected}/{n_total}", flush=True)
                    batch = []
                    if collected >= n_total:
                        return np.vstack(feats)[:n_total]
        if batch:
            encoded = encode_images(batch, transform, model, device)
            feats.append(encoded)
            collected += len(encoded)
        if collected >= n_total:
            return np.vstack(feats)[:n_total]
    out = np.vstack(feats) if feats else np.empty((0, 512), dtype=np.float32)
    if len(out) < n_total and allow_less:
        print(f"[{repo_id}] collected {len(out)}/{n_total}; continuing with another source", flush=True)
        return out
    if len(out) < n_total:
        raise RuntimeError(f"Only collected {len(out)} vectors from {repo_id}, need {n_total}")
    return out[:n_total]


def image_file_features(repo_id, n_total, batch_size, device, transform, model):
    api = HfApi()
    info = api.dataset_info(repo_id)
    files = sorted(
        s.rfilename for s in info.siblings
        if s.rfilename.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    feats = []
    batch = []
    collected = 0
    for filename in files:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        batch.append(img)
        if len(batch) >= batch_size:
            encoded = encode_images(batch, transform, model, device)
            feats.append(encoded)
            collected += len(encoded)
            if collected % (batch_size * 10) == 0:
                print(f"[{repo_id}] encoded {collected}/{n_total}", flush=True)
            batch = []
            if collected >= n_total:
                return np.vstack(feats)[:n_total]
    if batch:
        encoded = encode_images(batch, transform, model, device)
        feats.append(encoded)
        collected += len(encoded)
    out = np.vstack(feats) if feats else np.empty((0, 512), dtype=np.float32)
    if len(out) < n_total:
        print(f"[{repo_id}] only collected {len(out)} image vectors", flush=True)
    return out[:n_total]


def scannet_features(n_total, batch_size, device):
    if SCANNET_POINTS_TAR.exists():
        return scannet_point_features_from_tar(SCANNET_POINTS_TAR, n_total)

    transform, model = load_image_encoder(device)
    parts = []
    first = parquet_image_features(
        "ZiAngGu/scannet_box3d", SCANNET_SHARDS, n_total, batch_size, device,
        transform, model, allow_less=True
    )
    parts.append(first)
    collected = len(first)
    for repo_id in SCANNET_IMAGE_REPOS:
        if collected >= n_total:
            break
        need = n_total - collected
        part = image_file_features(repo_id, need, batch_size, device, transform, model)
        if len(part):
            parts.append(part)
            collected += len(part)
    out = np.vstack(parts) if parts else np.empty((0, 512), dtype=np.float32)
    if len(out) < n_total:
        raise RuntimeError(f"Only collected {len(out)} ScanNet vectors, need {n_total}")
    return normalize(out[:n_total])


def scannet_point_features_from_tar(path, n_total):
    """Load real ScanNet point records stored as 50k x 6 float32 scene bins."""
    parts = []
    collected = 0
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bin"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            if len(data) != member.size:
                continue
            arr = np.frombuffer(data, dtype=np.float32)
            if arr.size % 6 != 0:
                continue
            points = arr.reshape(-1, 6)
            parts.append(points)
            collected += len(points)
            print(f"[ScanNet points] loaded {member.name}: {collected}/{n_total}", flush=True)
            if collected >= n_total:
                break
    if collected < n_total:
        raise RuntimeError(f"Only collected {collected} ScanNet point vectors, need {n_total}")
    out = np.vstack(parts)[:n_total].astype(np.float32)
    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return normalize((out - mean) / std)


def find_tensor_features(obj):
    if torch.is_tensor(obj) and obj.ndim == 2:
        return obj.detach().cpu().float().numpy()
    if isinstance(obj, dict):
        for value in obj.values():
            found = find_tensor_features(value)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_tensor_features(value)
            if found is not None:
                return found
    return None


def ego4d_features(n_total):
    feats = []
    repo_id = "Jazzcharles/ego4d_videomae_L14_feature_fps8"
    for chunk in EGO4D_CHUNKS:
        path = hf_hub_download(repo_id=repo_id, filename=chunk, repo_type="dataset")
        print(f"[Ego4D] reading {chunk}", flush=True)
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    obj = torch.load(io.BytesIO(f.read()), map_location="cpu")
                except Exception:
                    continue
                arr = find_tensor_features(obj)
                if arr is None or arr.size == 0:
                    continue
                feats.append(arr)
                if sum(len(x) for x in feats) >= n_total:
                    return normalize(np.vstack(feats)[:n_total])
    out = np.vstack(feats) if feats else np.empty((0, 768), dtype=np.float32)
    if len(out) < n_total:
        raise RuntimeError(f"Only collected {len(out)} Ego4D vectors, need {n_total}")
    return normalize(out[:n_total])


def write_split(name, vectors, out_dir, train_count, query_count, source):
    os.makedirs(out_dir, exist_ok=True)
    vectors = normalize(vectors)
    train = vectors[:train_count]
    queries = vectors[train_count : train_count + query_count]
    gt = compute_l2_ground_truth(train, queries, k=100)
    np.save(os.path.join(out_dir, "train.npy"), train.astype(np.float32))
    np.save(os.path.join(out_dir, "queries.npy"), queries.astype(np.float32))
    np.save(os.path.join(out_dir, "gt.npy"), gt.astype(np.int32))
    metadata = {
        "name": name,
        "kind": "real_or_public_derived_features",
        "source": source,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": int(train.shape[0]),
        "n_queries": int(queries.shape[0]),
        "dimension": int(train.shape[1]),
        "distance": "squared Euclidean on unit-normalized vectors",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["core50", "scannet", "ego4d"])
    parser.add_argument("--out-root", default="data")
    parser.add_argument("--train-count", type=int, default=30000)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    n_total = args.train_count + args.query_count

    if "core50" in args.datasets:
        vecs = parquet_image_features("adrake17/core50", CORE50_SHARDS, n_total, args.batch_size, args.device)
        write_split("CoRE50", vecs, os.path.join(args.out_root, "core50"), args.train_count, args.query_count,
                    {"repo": "adrake17/core50", "shards": CORE50_SHARDS, "feature": "ResNet18 ImageNet feature"})

    if "scannet" in args.datasets:
        vecs = scannet_features(n_total, args.batch_size, args.device)
        write_split("ScanNet", vecs, os.path.join(args.out_root, "ScanNetCLIP"), args.train_count, args.query_count,
                    {"repo": "YangCaoCS/ScanNet_processed",
                     "file": str(SCANNET_POINTS_TAR),
                     "feature": "standardized ScanNet point XYZRGB vectors"})

    if "ego4d" in args.datasets:
        vecs = ego4d_features(n_total)
        write_split("Ego4D", vecs, os.path.join(args.out_root, "Ego4D"), args.train_count, args.query_count,
                    {"repo": "Jazzcharles/ego4d_videomae_L14_feature_fps8", "chunks": EGO4D_CHUNKS,
                     "feature": "VideoMAE-L14 8fps frame/video features"})


if __name__ == "__main__":
    main()
