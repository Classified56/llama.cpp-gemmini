#!/usr/bin/env python3
"""Layer-wise PCA across sorted expert-load distributions."""
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
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--scatter-layers", type=int, default=6)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def n_for(cumulative, threshold):
    if cumulative.size == 0 or cumulative[-1] < threshold: return None
    return int(np.searchsorted(cumulative, threshold) + 1)


def participation_ratio(values):
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0: return 0.0
    p = values / total
    den = np.sum(p*p)
    return float(1.0 / den) if den > 0 else 0.0


def main():
    args = parse_args()
    cache = subset_cache(load_cache(args.cache), args.length)
    maps = get_representation(cache, args.representation)
    n_samples, n_layers, n_ranks = maps.shape
    n_components = min(args.components, n_samples, n_ranks)
    if n_components < 1: raise ValueError("No PCA components can be fit")
    layer_ids = cache["layer_ids"].astype(int)

    loadings = np.zeros((n_layers, n_components, n_ranks), dtype=float)
    scores = np.zeros((n_samples, n_layers, n_components), dtype=float)
    summary_rows, variance_rows = [], []

    for lp in range(n_layers):
        X = maps[:, lp, :]
        X_fit = StandardScaler().fit_transform(X) if args.standardize else X
        total_var = float(np.var(X_fit, axis=0, ddof=1).sum()) if n_samples > 1 else 0.0
        if total_var <= 0:
            summary_rows.append({
                "layer": int(layer_ids[lp]), "total_feature_variance": 0.0,
                "pc1_explained_variance_ratio": 0.0, "pc1_pc2_explained_variance_ratio": 0.0,
                "components_90_percent": 0, "components_95_percent": 0, "components_99_percent": 0,
                "effective_dimension": 0.0, "constant_layer": 1,
            })
            for ci in range(n_components):
                variance_rows.append({"layer": int(layer_ids[lp]), "component": ci+1, "explained_variance": 0.0, "explained_variance_ratio": 0.0, "cumulative_explained_variance": 0.0})
            continue

        pca = PCA(n_components=n_components)
        s = pca.fit_transform(X_fit)
        ratios = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0)
        cumulative = np.cumsum(ratios)
        scores[:, lp, :] = s
        loadings[lp, :, :] = pca.components_
        summary_rows.append({
            "layer": int(layer_ids[lp]), "total_feature_variance": total_var,
            "pc1_explained_variance_ratio": float(ratios[0]),
            "pc1_pc2_explained_variance_ratio": float(ratios[:2].sum()),
            "components_90_percent": n_for(cumulative, 0.90),
            "components_95_percent": n_for(cumulative, 0.95),
            "components_99_percent": n_for(cumulative, 0.99),
            "effective_dimension": participation_ratio(pca.explained_variance_),
            "constant_layer": 0,
        })
        for ci in range(n_components):
            variance_rows.append({
                "layer": int(layer_ids[lp]), "component": ci+1,
                "explained_variance": float(pca.explained_variance_[ci]),
                "explained_variance_ratio": float(ratios[ci]),
                "cumulative_explained_variance": float(cumulative[ci]),
            })

    cache_path = Path(str(cache["_cache_path"]))
    suffix = "_standardized" if args.standardize else ""
    out = args.output_dir.expanduser().resolve() if args.output_dir else (
        cache_path.parent.parent / "layer_pca" / args.length / f"{args.representation}{suffix}"
    )
    out.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "layer_pca_summary.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(out / "layer_pca_explained_variance.csv", index=False)

    score_rows = []
    for si in range(n_samples):
        for lp in range(n_layers):
            row = {"run_id": str(cache["run_ids"][si]), "class_label": str(cache["labels"][si]), "length_label": str(cache["length_labels"][si]), "layer": int(layer_ids[lp])}
            for ci in range(n_components): row[f"pc{ci+1:02d}"] = float(scores[si, lp, ci])
            score_rows.append(row)
    pd.DataFrame(score_rows).to_csv(out / "layer_pca_scores.csv", index=False)
    np.savez_compressed(out / "layer_pca_loadings.npz", loadings=loadings, scores=scores, layer_ids=layer_ids, load_ranks=cache["load_ranks"])

    fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(111)
    ax.plot(summary_df["layer"], summary_df["pc1_explained_variance_ratio"], marker="o", label="PC1")
    ax.plot(summary_df["layer"], summary_df["pc1_pc2_explained_variance_ratio"], marker="o", label="PC1 + PC2")
    ax.set_xlabel("Layer"); ax.set_ylabel("Explained variance fraction"); ax.set_ylim(0, 1.01)
    ax.set_title("Layer-wise PCA variance concentration"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "layer_pc1_variance_profile.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(111)
    ax.plot(summary_df["layer"], summary_df["effective_dimension"], marker="o")
    ax.set_xlabel("Layer"); ax.set_ylabel("Participation-ratio dimension")
    ax.set_title("Effective dimensionality of routing variation by layer"); fig.tight_layout()
    fig.savefig(out / "layer_effective_dimension_profile.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(111)
    ax.plot(summary_df["layer"], summary_df["components_95_percent"], marker="o")
    ax.set_xlabel("Layer"); ax.set_ylabel("Components for 95% variance")
    ax.set_title("Layer-wise dimensions required for 95% variance"); fig.tight_layout()
    fig.savefig(out / "layer_components_95_profile.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
    im = ax.imshow(loadings[:, 0, :], aspect="auto")
    ax.set_xlabel("Sorted expert load rank"); ax.set_ylabel("Layer"); ax.set_title("Layer-wise PC1 loading patterns")
    fig.colorbar(im, ax=ax, label="PC1 loading"); fig.tight_layout()
    fig.savefig(out / "layer_pc1_loading_heatmap.png", dpi=180); plt.close(fig)

    if n_components >= 2 and args.scatter_layers > 0:
        variable = summary_df[summary_df["constant_layer"] == 0].sort_values("total_feature_variance", ascending=False)
        for layer_id in variable["layer"].head(args.scatter_layers):
            lp = int(np.where(layer_ids == int(layer_id))[0][0])
            fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111)
            ax.scatter(scores[:, lp, 0], scores[:, lp, 1], s=18)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"Layer {layer_id} PCA prompt geometry")
            fig.tight_layout(); fig.savefig(out / f"layer_{int(layer_id):03d}_pc1_pc2.png", dpi=180); plt.close(fig)

    nonconst = summary_df[summary_df["constant_layer"] == 0]
    overall = {
        "samples": int(n_samples), "layers": int(n_layers), "load_ranks": int(n_ranks),
        "retained_components_per_layer": int(n_components), "representation": args.representation,
        "preprocessing": "standardized" if args.standardize else "centered_only",
        "constant_layers": [int(v) for v in summary_df.loc[summary_df["constant_layer"] == 1, "layer"]],
        "mean_pc1_explained_variance": float(nonconst["pc1_explained_variance_ratio"].mean()) if len(nonconst) else 0.0,
        "mean_pc1_pc2_explained_variance": float(nonconst["pc1_pc2_explained_variance_ratio"].mean()) if len(nonconst) else 0.0,
        "mean_effective_dimension": float(nonconst["effective_dimension"].mean()) if len(nonconst) else 0.0,
    }
    save_json(out / "layer_pca_overall_summary.json", overall)
    print(f"Layer-wise PCA: {out}")
    print(f"Mean PC1={100*overall['mean_pc1_explained_variance']:.2f}% | Mean PC1+PC2={100*overall['mean_pc1_pc2_explained_variance']:.2f}% | Mean effective dimension={overall['mean_effective_dimension']:.2f}")
    if overall["constant_layers"]: print(f"Constant layers: {overall['constant_layers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
