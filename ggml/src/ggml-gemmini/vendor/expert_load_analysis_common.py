#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances


def resolve_cache(path: Path) -> Path:
    path = path.expanduser().resolve()

    if path.is_file():
        return path

    candidates = [
        path / "sorted_loads.npz",
        path / "analysis_cache" / "sorted_loads.npz",
        path / "expert_load_analysis" / "analysis_cache" / "sorted_loads.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = list(path.rglob("sorted_loads.npz"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find sorted_loads.npz under {path}")
    raise RuntimeError(
        f"Found {len(matches)} sorted_loads.npz files under {path}; "
        "pass the specific cache file."
    )


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    cache_path = resolve_cache(path)
    with np.load(cache_path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    result["_cache_path"] = np.asarray(str(cache_path))
    return result


def subset_cache(cache: Dict[str, np.ndarray], length: str):
    labels = cache["length_labels"].astype(str)

    if length == "all":
        mask = np.ones(len(labels), dtype=bool)
    else:
        mask = labels == length

    if not np.any(mask):
        raise ValueError(f'No samples with length_label="{length}" in cache')

    subset = {}
    for key, value in cache.items():
        if key.startswith("_"):
            subset[key] = value
            continue

        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == len(labels)
        ):
            subset[key] = value[mask]
        else:
            subset[key] = value

    subset["_mask"] = mask
    return subset


def get_representation(cache: Dict[str, np.ndarray], representation: str):
    if representation == "raw":
        return cache["sorted_raw"].astype(np.float64, copy=False)
    if representation == "layer_fraction":
        return cache["sorted_fraction"].astype(np.float64, copy=False)
    raise ValueError(f"Unsupported representation: {representation}")


def pairwise_map_distance(X: np.ndarray, metric: str) -> np.ndarray:
    """
    X shape: [samples, layers, ranks].

    wasserstein:
        Each layer is already sorted, so equal-mass empirical W1 is the mean
        absolute rank-wise difference. Averaging over layers is therefore the
        mean absolute difference over the whole canonical map.

    l1/l2/cosine:
        Distances on the flattened canonical map.
    """
    flat = X.reshape(X.shape[0], -1)

    if metric == "wasserstein":
        return pairwise_distances(flat, metric="manhattan") / flat.shape[1]
    if metric == "l1":
        return pairwise_distances(flat, metric="manhattan")
    if metric == "l2":
        return pairwise_distances(flat, metric="euclidean")
    if metric == "cosine":
        return pairwise_distances(flat, metric="cosine")

    raise ValueError(f"Unsupported metric: {metric}")


def per_layer_distance(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """
    a,b shape: [layers, ranks].
    Returns one distance per layer.
    """
    if metric == "wasserstein":
        return np.mean(np.abs(a - b), axis=1)
    if metric == "l1":
        return np.sum(np.abs(a - b), axis=1)
    if metric == "l2":
        return np.sqrt(np.sum((a - b) ** 2, axis=1))
    if metric == "cosine":
        num = np.sum(a * b, axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        result = np.ones(a.shape[0], dtype=np.float64)
        valid = den > 0
        result[valid] = 1.0 - num[valid] / den[valid]
        both_zero = (np.linalg.norm(a, axis=1) == 0) & (
            np.linalg.norm(b, axis=1) == 0
        )
        result[both_zero] = 0.0
        return result

    raise ValueError(f"Unsupported metric: {metric}")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def finite_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
