#!/usr/bin/env python3
"""Classical MDS / PCoA on pairwise distances between sorted expert-load maps."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from expert_load_analysis_common import get_representation, load_cache, pairwise_map_distance, save_json, subset_cache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("--length", choices=["short", "long", "all"], default="short")
    p.add_argument("--representation", choices=["raw", "layer_fraction"], default="layer_fraction")
    p.add_argument("--metric", choices=["wasserstein", "l1", "l2", "cosine"], default="wasserstein")
    p.add_argument("--dimensions", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def classical_mds(D, dimensions):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values = values[order]; vectors = vectors[:, order]
    positive = values > 1e-12
    pvals = values[positive]; pvecs = vectors[:, positive]
    dims = min(dimensions, len(pvals))
    if dims < 1:
        raise ValueError("No positive MDS eigenvalues")
    coords = pvecs[:, :dims] * np.sqrt(pvals[:dims])[None, :]
    return coords, values


def upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def stress1(D, Dhat):
    d, dh = upper(D), upper(Dhat)
    denom = np.sum(d*d)
    return float(np.sqrt(np.sum((d-dh)**2)/denom)) if denom > 0 else 0.0


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0


def main():
    args = parse_args()
    if args.dimensions <= 0: raise ValueError("--dimensions must be positive")
    cache = subset_cache(load_cache(args.cache), args.length)
    X = get_representation(cache, args.representation)
    D = pairwise_map_distance(X, args.metric)
    coords, eigenvalues = classical_mds(D, args.dimensions)

    cache_path = Path(str(cache["_cache_path"]))
    out = args.output_dir.expanduser().resolve() if args.output_dir else (
        cache_path.parent.parent / "mds" / args.length / args.representation / args.metric
    )
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out / "mds_pairwise_distances.npz", distance_matrix=D, labels=cache["labels"], run_ids=cache["run_ids"])

    df = pd.DataFrame({
        "run_id": cache["run_ids"].astype(str),
        "class_label": cache["labels"].astype(str),
        "length_label": cache["length_labels"].astype(str),
        "repetition": cache["repetitions"].astype(int),
        "selection_index": cache["selection_indices"].astype(int),
        "source_row": cache["source_rows"].astype(int),
    })
    for i in range(coords.shape[1]): df[f"mds_{i+1:02d}"] = coords[:, i]
    df.to_csv(out / "mds_coordinates.csv", index=False)

    pos_sum = float(eigenvalues[eigenvalues > 0].sum())
    neg_sum = float(np.abs(eigenvalues[eigenvalues < 0]).sum())
    rows, cumulative = [], 0.0
    for i, value in enumerate(eigenvalues, start=1):
        frac = float(value / pos_sum) if value > 0 and pos_sum > 0 else 0.0
        cumulative += frac
        rows.append({"dimension": i, "eigenvalue": float(value), "positive_eigenvalue_fraction": frac, "cumulative_positive_eigenvalue_fraction": cumulative})
    pd.DataFrame(rows).to_csv(out / "mds_eigenvalues.csv", index=False)

    original = upper(D)
    embedded = pairwise_distances(coords)
    summary = {
        "samples": int(len(X)), "representation": args.representation, "distance_metric": args.metric,
        "retained_dimensions": int(coords.shape[1]),
        "stress_1": stress1(D, embedded),
        "pairwise_distance_pearson_r": corr(original, upper(embedded)),
        "positive_eigenvalue_sum": pos_sum,
        "negative_eigenvalue_absolute_sum": neg_sum,
        "negative_to_positive_eigenvalue_ratio": neg_sum / pos_sum if pos_sum else None,
        "dimension_1_positive_variance_fraction": float(max(eigenvalues[0], 0)/pos_sum) if pos_sum else 0.0,
        "dimension_1_2_positive_variance_fraction": float(np.maximum(eigenvalues[:2], 0).sum()/pos_sum) if pos_sum else 0.0,
    }

    if coords.shape[1] >= 2:
        D2 = pairwise_distances(coords[:, :2])
        summary["stress_1_2d"] = stress1(D, D2)
        summary["pairwise_distance_pearson_r_2d"] = corr(original, upper(D2))

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
        ax.scatter(coords[:, 0], coords[:, 1], s=18)
        ax.set_xlabel("MDS dimension 1"); ax.set_ylabel("MDS dimension 2")
        ax.set_title(f"Classical MDS of {args.metric} prompt distances")
        fig.tight_layout(); fig.savefig(out / "mds_2d_scatter.png", dpi=180); plt.close(fig)

        emb2 = upper(D2)
        fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111)
        ax.scatter(original, emb2, s=8)
        lo = min(float(original.min()), float(emb2.min())); hi = max(float(original.max()), float(emb2.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(f"Original {args.metric} distance"); ax.set_ylabel("2-D MDS Euclidean distance")
        ax.set_title(f"Distance preservation: r={summary['pairwise_distance_pearson_r_2d']:.3f}, stress={summary['stress_1_2d']:.3f}")
        fig.tight_layout(); fig.savefig(out / "mds_distance_preservation.png", dpi=180); plt.close(fig)

    save_json(out / "mds_summary.json", summary)
    print(f"MDS analysis: {out}")
    print(f"2-D positive-eigenvalue share={100*summary['dimension_1_2_positive_variance_fraction']:.2f}%")
    if "stress_1_2d" in summary:
        print(f"2-D stress={summary['stress_1_2d']:.4f} | distance r={summary['pairwise_distance_pearson_r_2d']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
