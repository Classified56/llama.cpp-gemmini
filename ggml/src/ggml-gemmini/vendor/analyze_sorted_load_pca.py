#!/usr/bin/env python3
"""Global PCA across canonical sorted expert-load maps."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from expert_load_analysis_common import get_representation, load_cache, save_json, subset_cache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("--length", choices=["short", "long", "all"], default="short")
    p.add_argument("--representation", choices=["raw", "layer_fraction"], default="layer_fraction")
    p.add_argument("--components", type=int, default=20)
    p.add_argument("--plot-components", type=int, default=4)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def n_for(cumulative, threshold):
    if cumulative.size == 0:
        return 0
    if cumulative[-1] < threshold:
        return None
    return int(np.searchsorted(cumulative, threshold) + 1)


def main():
    args = parse_args()
    cache = subset_cache(load_cache(args.cache), args.length)
    maps = get_representation(cache, args.representation)
    n_samples, n_layers, n_ranks = maps.shape
    X = maps.reshape(n_samples, -1)

    if args.standardize:
        X_fit = StandardScaler().fit_transform(X)
        preprocessing = "standardized"
    else:
        X_fit = X
        preprocessing = "centered_only"

    n_components = min(args.components, X_fit.shape[0], X_fit.shape[1])
    if n_components < 1:
        raise ValueError("No PCA components can be fit")

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_fit)
    loadings = pca.components_.reshape(n_components, n_layers, n_ranks)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    cache_path = Path(str(cache["_cache_path"]))
    suffix = "_standardized" if args.standardize else ""
    out = args.output_dir.expanduser().resolve() if args.output_dir else (
        cache_path.parent.parent / "pca" / args.length / f"{args.representation}{suffix}"
    )
    out.mkdir(parents=True, exist_ok=True)

    score_df = pd.DataFrame({
        "run_id": cache["run_ids"].astype(str),
        "class_label": cache["labels"].astype(str),
        "length_label": cache["length_labels"].astype(str),
        "repetition": cache["repetitions"].astype(int),
        "selection_index": cache["selection_indices"].astype(int),
        "source_row": cache["source_rows"].astype(int),
    })
    for i in range(scores.shape[1]):
        score_df[f"pc{i+1:02d}"] = scores[:, i]
    score_df.to_csv(out / "pca_scores.csv", index=False)

    ev = pd.DataFrame({
        "component": np.arange(1, n_components + 1),
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": cumulative,
        "singular_value": pca.singular_values_,
    })
    ev.to_csv(out / "pca_explained_variance.csv", index=False)

    np.savez_compressed(
        out / "pca_loadings.npz",
        components=loadings,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        layer_ids=cache["layer_ids"],
        load_ranks=cache["load_ranks"],
        mean=pca.mean_,
    )

    layer_rows, rank_rows = [], []
    for ci in range(n_components):
        sq = loadings[ci] ** 2
        total = sq.sum()
        lc = sq.sum(axis=1) / total if total else np.zeros(n_layers)
        rc = sq.sum(axis=0) / total if total else np.zeros(n_ranks)
        for li, frac in enumerate(lc):
            layer_rows.append({"component": ci+1, "layer": int(cache["layer_ids"][li]), "contribution_fraction": float(frac)})
        for ri, frac in enumerate(rc):
            rank_rows.append({"component": ci+1, "load_rank": int(cache["load_ranks"][ri]), "contribution_fraction": float(frac)})
    pd.DataFrame(layer_rows).to_csv(out / "pca_component_layer_contributions.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(out / "pca_component_rank_contributions.csv", index=False)

    recon_rows = []
    centered_ss = np.sum((X_fit - pca.mean_) ** 2)
    for k in range(1, n_components + 1):
        reconstructed = scores[:, :k] @ pca.components_[:k] + pca.mean_
        err = np.sum((X_fit - reconstructed) ** 2)
        recon_rows.append({
            "components": k,
            "mse": float(np.mean((X_fit - reconstructed) ** 2)),
            "relative_centered_squared_error": float(err / centered_ss) if centered_ss > 0 else 0.0,
        })
    pd.DataFrame(recon_rows).to_csv(out / "pca_reconstruction_curve.csv", index=False)

    fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
    xs = np.arange(1, n_components + 1)
    ax.plot(xs, pca.explained_variance_ratio_, marker="o", label="Individual")
    ax.plot(xs, cumulative, marker="o", label="Cumulative")
    for y in (0.90, 0.95, 0.99): ax.axhline(y, linestyle="--")
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance fraction"); ax.set_ylim(0, 1.01)
    ax.set_title("Global PCA explained variance"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "pca_explained_variance.png", dpi=180); plt.close(fig)

    if scores.shape[1] >= 2:
        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
        ax.scatter(scores[:, 0], scores[:, 1], s=18)
        ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.2f}% variance)")
        ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.2f}% variance)")
        ax.set_title("Global PCA prompt embedding"); fig.tight_layout()
        fig.savefig(out / "pca_pc1_pc2_scatter.png", dpi=180); plt.close(fig)

    for ci in range(min(args.plot_components, n_components)):
        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
        im = ax.imshow(loadings[ci], aspect="auto")
        ax.set_xlabel("Sorted expert load rank"); ax.set_ylabel("Layer")
        ax.set_title(f"PC{ci+1} loading map ({100*pca.explained_variance_ratio_[ci]:.2f}% variance)")
        fig.colorbar(im, ax=ax, label="PCA loading"); fig.tight_layout()
        fig.savefig(out / f"pc_{ci+1:02d}_loading_heatmap.png", dpi=180); plt.close(fig)

    summary = {
        "samples": int(n_samples), "layers": int(n_layers), "load_ranks": int(n_ranks),
        "features": int(X.shape[1]), "retained_components": int(n_components),
        "representation": args.representation, "preprocessing": preprocessing,
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc1_pc2_explained_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "components_90_percent": n_for(cumulative, 0.90),
        "components_95_percent": n_for(cumulative, 0.95),
        "components_99_percent": n_for(cumulative, 0.99),
    }
    save_json(out / "pca_summary.json", summary)
    print(f"Global PCA: {out}")
    print(f"PC1={100*summary['pc1_explained_variance']:.2f}% | PC1+PC2={100*summary['pc1_pc2_explained_variance']:.2f}% | 90%={summary['components_90_percent']} PCs | 95%={summary['components_95_percent']} PCs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
