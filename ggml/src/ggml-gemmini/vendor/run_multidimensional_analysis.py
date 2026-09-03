#!/usr/bin/env python3
"""Run global PCA, Wasserstein MDS, and layer-wise PCA from one sorted-load cache."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def csv_list(value, allowed):
    result = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in result if x not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"Invalid values {bad}; allowed={sorted(allowed)}")
    return result


def run(cmd):
    print("\n+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path, help="sorted_loads.npz or directory containing it")
    p.add_argument("--lengths", type=lambda x: csv_list(x, {"short", "long", "all"}), default=["short", "long"])
    p.add_argument("--representations", type=lambda x: csv_list(x, {"raw", "layer_fraction"}), default=["layer_fraction"])
    p.add_argument("--pca-components", type=int, default=20)
    p.add_argument("--layer-pca-components", type=int, default=8)
    p.add_argument("--mds-metric", choices=["wasserstein", "l1", "l2", "cosine"], default="wasserstein")
    p.add_argument("--mds-dimensions", type=int, default=3)
    p.add_argument("--standardize-pca", action="store_true")
    p.add_argument("--standardize-layer-pca", action="store_true")
    p.add_argument("--skip", default="", help="Comma separated: pca,mds,layer_pca")
    p.add_argument("--output-root", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cache = args.cache.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}

    for length in args.lengths:
        for rep in args.representations:
            if "pca" not in skip:
                cmd = [sys.executable, str(HERE / "analyze_sorted_load_pca.py"), str(cache), "--length", length, "--representation", rep, "--components", str(args.pca_components)]
                if args.standardize_pca: cmd.append("--standardize")
                if output_root: cmd += ["--output-dir", str(output_root / "pca" / length / rep)]
                run(cmd)

            if "mds" not in skip:
                cmd = [sys.executable, str(HERE / "analyze_sorted_load_mds.py"), str(cache), "--length", length, "--representation", rep, "--metric", args.mds_metric, "--dimensions", str(args.mds_dimensions)]
                if output_root: cmd += ["--output-dir", str(output_root / "mds" / length / rep / args.mds_metric)]
                run(cmd)

            if "layer_pca" not in skip:
                cmd = [sys.executable, str(HERE / "analyze_layerwise_pca.py"), str(cache), "--length", length, "--representation", rep, "--components", str(args.layer_pca_components)]
                if args.standardize_layer_pca: cmd.append("--standardize")
                if output_root: cmd += ["--output-dir", str(output_root / "layer_pca" / length / rep)]
                run(cmd)

    print("\nAll requested multidimensional analyses complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
