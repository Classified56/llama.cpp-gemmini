#!/usr/bin/env python3
"""Plot llama.cpp MoE expert load and compute measurements.

The parser understands the key=value records emitted by llama-expert-trace:

    expert_compute ...
    expert_compute_total ...

Pass multiple log files to measure run-to-run consistency. Each input file is
treated as one independent run.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise SystemExit(
        "Missing plotting dependencies. Install them with:\n"
        "  python3 -m pip install numpy pandas matplotlib seaborn\n"
        f"Original error: {exc}"
    ) from exc


KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
INTEGER_FIELDS = {
    "layer",
    "expert",
    "load",
    "matrix_ops",
    "cycle_before",
    "cycle_after",
    "cycle_delta",
}

PLOT_DESCRIPTIONS = {
    "overview": "overall expert load versus total measured cycles",
    "by-layer": "load versus total cycles, faceted by MoE layer",
    "normalized-distribution": "histogram and ECDF of total cycles per routed token",
    "by-load": "cycles-per-token distributions grouped by equal load",
    "by-layer-distribution": "cycles-per-token distributions grouped by layer",
    "expert-heatmap": "layer/expert normalized-time or repeated-run CV heatmap",
    "operation-distribution": "cycles-per-token distributions for gate/up/down operations",
    "ffn-stage-scatter": "load versus measured cycles separately for each FFN stage",
    "ffn-stage-load-heatmap": "per-layer/expert load heatmaps for each FFN stage",
}

PREFERRED_OPERATION_ORDER = (
    "ffn_moe_gate",
    "ffn_gate",
    "ffn_moe_up",
    "ffn_up",
    "ffn_moe_down",
    "ffn_down",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Graph MoE expert load versus measured cycles and generate "
            "measurement-consistency plots."
        )
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Log files, directories, or quoted glob patterns (for example 'logs/*.log').",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="expert-trace-plots",
        help="Output directory (default: expert-trace-plots).",
    )
    parser.add_argument(
        "--phase",
        default="prefill",
        help="Phase to plot, usually prefill or decode. Use 'all' for no filter (default: prefill).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180).",
    )
    parser.add_argument(
        "--max-load-groups",
        type=int,
        default=20,
        help="Maximum load values in the grouped distribution plot (default: 20).",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Do not label expert numbers in the per-layer plots.",
    )
    parser.add_argument(
        "--plots",
        nargs="*",
        choices=[*PLOT_DESCRIPTIONS, "all"],
        metavar="PLOT",
        help=(
            "Graphs to generate. Omit this option, or pass it without names, to generate all. "
            "Use --list-plots to show the available names."
        ),
    )
    parser.add_argument(
        "--list-plots",
        action="store_true",
        help="List the graph names accepted by --plots and exit.",
    )
    return parser.parse_args()


def selected_plots(requested: list[str] | None) -> set[str]:
    if not requested or "all" in requested:
        return set(PLOT_DESCRIPTIONS)
    return set(requested)


def print_plot_names() -> None:
    print("Available graphs:")
    width = max(len(name) for name in PLOT_DESCRIPTIONS)
    for name, description in PLOT_DESCRIPTIONS.items():
        print(f"  {name:<{width}}  {description}")


def resolve_inputs(items: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for item in items:
        matches = [Path(p) for p in glob.glob(item)]
        if not matches:
            matches = [Path(item)]
        for path in matches:
            if path.is_dir():
                resolved.extend(p for p in path.rglob("*") if p.is_file())
            elif path.is_file():
                resolved.append(path)
            else:
                print(f"warning: input does not exist: {path}", file=sys.stderr)

    # Preserve command-line order while removing duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        canonical = path.resolve()
        if canonical not in seen:
            unique.append(canonical)
            seen.add(canonical)
    return unique


def parse_record(line: str) -> tuple[str, dict[str, object]] | None:
    if line.startswith("expert_compute_total "):
        kind = "total"
    elif line.startswith("expert_compute "):
        kind = "operation"
    else:
        return None

    fields: dict[str, object] = dict(KEY_VALUE_RE.findall(line))
    for key in INTEGER_FIELDS:
        if key in fields:
            try:
                fields[key] = int(str(fields[key]), 0)
            except ValueError:
                fields[key] = np.nan
    return kind, fields


def load_logs(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_rows: list[dict[str, object]] = []
    operation_rows: list[dict[str, object]] = []

    for run_number, path in enumerate(paths, start=1):
        run = f"run_{run_number:03d}:{path.name}"
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                parsed = parse_record(line.rstrip("\n"))
                if parsed is None:
                    continue
                kind, fields = parsed
                fields.update(
                    {
                        "run": run,
                        "source_file": str(path),
                        "line_number": line_number,
                    }
                )
                if kind == "total":
                    total_rows.append(fields)
                else:
                    operation_rows.append(fields)

    totals = pd.DataFrame(total_rows)
    operations = pd.DataFrame(operation_rows)
    for frame in (totals, operations):
        if not frame.empty and {"cycle_delta", "load"}.issubset(frame.columns):
            frame["cycles_per_token"] = np.where(
                frame["load"] > 0,
                frame["cycle_delta"] / frame["load"],
                np.nan,
            )
    return totals, operations


def active_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["load"] > 0) & (frame["cycle_delta"] > 0)].copy()


def ordered_operations(operations: pd.DataFrame) -> list[str]:
    if operations.empty or "operation" not in operations.columns:
        return []
    available = [str(value) for value in operations["operation"].dropna().unique()]
    preferred = [name for name in PREFERRED_OPERATION_ORDER if name in available]
    return preferred + sorted(name for name in available if name not in preferred)


def operation_label(operation: str) -> str:
    short = operation.removeprefix("ffn_moe_").removeprefix("ffn_")
    return f"FFN {short.replace('_', ' ')}"


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def regression_stats(active: pd.DataFrame) -> dict[str, float]:
    x = active["load"].to_numpy(dtype=float)
    y = active["cycle_delta"].to_numpy(dtype=float)
    if len(x) < 2 or np.all(x == x[0]):
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "pearson_r": float("nan"),
        }
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum else float("nan")
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "pearson_r": pearson_r,
    }


def plot_overview(active: pd.DataFrame, output: Path, dpi: int) -> dict[str, float]:
    stats = regression_stats(active)
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    points = ax.scatter(
        active["load"],
        active["cycle_delta"],
        c=active["layer"],
        cmap="viridis",
        s=28,
        alpha=0.72,
        linewidths=0,
    )
    # if math.isfinite(stats["slope"]):
    #     x_line = np.linspace(0, float(active["load"].max()), 200)
    #     y_line = stats["slope"] * x_line + stats["intercept"]
    #     ax.plot(
    #         x_line,
    #         y_line,
    #         color="#cf3f3f",
    #         linewidth=2,
    #         label=(
    #             f"linear fit: {stats['slope']:.3g} ticks/load + "
    #             f"{stats['intercept']:.3g}; $R^2$={stats['r_squared']:.4f}"
    #         ),
    #     )
    #     ax.legend(frameon=False, loc="upper left")
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Layer")
    ax.set(
        title="MoE expert load versus measured CPU compute time",
        xlabel="Expert load (routed token assignments)",
        ylabel="Measured cycle delta",
    )
    ax.grid(alpha=0.22)
    save_figure(fig, output / "01_load_vs_cycles_overview.png", dpi)
    return stats


def plot_per_layer(
    active: pd.DataFrame,
    output: Path,
    dpi: int,
    annotate: bool,
) -> None:
    layers = sorted(active["layer"].dropna().astype(int).unique())
    columns = 4
    rows = math.ceil(len(layers) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, max(4.0, rows * 3.25)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    scatter = None
    one_run = active["run"].nunique() == 1
    for ax, layer in zip(axes_array, layers):
        part = active[active["layer"] == layer]
        scatter = ax.scatter(
            part["load"],
            part["cycle_delta"],
            c=part["expert"],
            cmap="turbo",
            vmin=float(active["expert"].min()),
            vmax=float(active["expert"].max()),
            s=24,
            alpha=0.75,
            linewidths=0,
        )
        if annotate and one_run and len(part) <= 64:
            for row in part.itertuples():
                ax.annotate(
                    str(int(row.expert)),
                    (row.load, row.cycle_delta),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=5.5,
                    alpha=0.8,
                )
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.grid(alpha=0.18)

    for ax in axes_array[len(layers) :]:
        ax.set_visible(False)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=list(axes_array[: len(layers)]), shrink=0.55, pad=0.01)
        colorbar.set_label("Expert position")
    fig.supxlabel("Expert load (routed token assignments)")
    fig.supylabel("Measured cycle delta")
    fig.suptitle("Expert load versus compute time by MoE layer", fontsize=16)
    save_figure(fig, output / "02_load_vs_cycles_by_layer.png", dpi)


def plot_normalized_distribution(active: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    sns.histplot(
        data=active,
        x="cycles_per_token",
        bins="auto",
        kde=True,
        ax=axes[0],
        color="#326c8e",
    )
    axes[0].set(
        title="Normalized compute-time distribution",
        xlabel="Cycle delta per routed token assignment",
        ylabel="Measurements",
    )
    sns.ecdfplot(
        data=active,
        x="cycles_per_token",
        ax=axes[1],
        color="#cf3f3f",
        linewidth=2,
    )
    axes[1].set(
        title="Empirical cumulative distribution",
        xlabel="Cycle delta per routed token assignment",
        ylabel="Fraction of measurements",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    save_figure(fig, output / "03_cycles_per_token_distribution.png", dpi)


def plot_by_load(
    active: pd.DataFrame,
    output: Path,
    dpi: int,
    max_groups: int,
) -> None:
    most_common = active["load"].value_counts().head(max_groups).index
    selected = active[active["load"].isin(most_common)].copy()
    order = sorted(int(value) for value in most_common)
    fig, ax = plt.subplots(figsize=(max(11, len(order) * 0.6), 6), constrained_layout=True)
    sns.boxplot(
        data=selected,
        x="load",
        y="cycles_per_token",
        order=order,
        color="#78a6c8",
        fliersize=2.5,
        linewidth=1,
        ax=ax,
    )
    ax.set(
        title="Measurement consistency at equal expert loads",
        xlabel="Expert load",
        ylabel="Cycle delta per routed token assignment",
    )
    ax.grid(axis="y", alpha=0.22)
    save_figure(fig, output / "04_cycles_per_token_by_load.png", dpi)


def plot_by_layer(active: pd.DataFrame, output: Path, dpi: int) -> None:
    order = sorted(active["layer"].dropna().astype(int).unique())
    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.55), 6), constrained_layout=True)
    sns.boxplot(
        data=active,
        x="layer",
        y="cycles_per_token",
        order=order,
        color="#80b88a",
        fliersize=2.5,
        linewidth=1,
        ax=ax,
    )
    ax.set(
        title="Normalized expert compute time by layer",
        xlabel="Layer",
        ylabel="Cycle delta per routed token assignment",
    )
    ax.grid(axis="y", alpha=0.22)
    save_figure(fig, output / "05_cycles_per_token_by_layer.png", dpi)


def plot_consistency_heatmap(
    active: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    dpi: int,
) -> str:
    repeated = summary["samples"].max() >= 2
    if repeated:
        value = "cycles_per_token_cv"
        title = "Run-to-run coefficient of variation by layer and expert"
        label = "Coefficient of variation"
        filename = "06_run_to_run_cv_heatmap.png"
    else:
        value = "cycles_per_token_mean"
        title = "Normalized compute time by layer and expert"
        label = "Mean cycle delta per token"
        filename = "06_layer_expert_normalized_heatmap.png"

    matrix = summary.pivot(index="layer", columns="expert", values=value)
    fig, ax = plt.subplots(
        figsize=(max(12, matrix.shape[1] * 0.42), max(7, matrix.shape[0] * 0.34)),
        constrained_layout=True,
    )
    sns.heatmap(
        matrix,
        cmap="mako" if repeated else "viridis",
        ax=ax,
        cbar_kws={"label": label},
    )
    ax.set(title=title, xlabel="Expert position", ylabel="Layer")
    save_figure(fig, output / filename, dpi)
    return "cv" if repeated else "mean"


def plot_operation_distribution(
    operations: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    if operations.empty or "operation" not in operations.columns:
        return
    active = active_rows(operations)
    if active.empty:
        return
    order = ordered_operations(active)
    active["operation_label"] = active["operation"].map(operation_label)
    display_order = [operation_label(name) for name in order]
    fig, ax = plt.subplots(figsize=(max(9, len(order) * 2.1), 6), constrained_layout=True)
    sns.boxplot(
        data=active,
        x="operation_label",
        y="cycles_per_token",
        order=display_order,
        color="#c29bca",
        fliersize=2.5,
        ax=ax,
    )
    ax.set(
        title="Per-operation normalized compute-time distribution",
        xlabel="Expert matrix operation",
        ylabel="Cycle delta per routed token assignment",
    )
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.22)
    save_figure(fig, output / "07_cycles_per_token_by_operation.png", dpi)


def plot_ffn_stage_scatter(
    operations: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    if operations.empty or "operation" not in operations.columns:
        print("warning: skipping ffn-stage-scatter; no expert_compute records found", file=sys.stderr)
        return
    active = active_rows(operations)
    order = ordered_operations(active)
    if not order:
        print("warning: skipping ffn-stage-scatter; no FFN operations found", file=sys.stderr)
        return

    fig, axes = plt.subplots(
        1,
        len(order),
        figsize=(max(7, len(order) * 5.2), 5.5),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    scatter = None
    for ax, operation in zip(axes_array, order):
        part = active[active["operation"] == operation]
        scatter = ax.scatter(
            part["load"],
            part["cycle_delta"],
            c=part["layer"],
            cmap="viridis",
            s=24,
            alpha=0.72,
            linewidths=0,
        )
        stats = regression_stats(part)
        if math.isfinite(stats["slope"]):
            x_line = np.linspace(0, float(part["load"].max()), 150)
            ax.plot(
                x_line,
                stats["slope"] * x_line + stats["intercept"],
                color="#cf3f3f",
                linewidth=1.8,
            )
        ax.set_title(
            f"{operation_label(operation)}\n"
            f"$R^2$={stats['r_squared']:.4f}, slope={stats['slope']:.3g}"
        )
        ax.grid(alpha=0.2)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=list(axes_array), pad=0.02)
        colorbar.set_label("Layer")
    fig.supxlabel("Expert load (routed token assignments)")
    fig.supylabel("Measured cycle delta")
    fig.suptitle("Expert load versus measured compute time by FFN stage", fontsize=15)
    save_figure(fig, output / "08_ffn_stage_load_vs_cycles.png", dpi)


def complete_stage_loads(
    totals: pd.DataFrame,
    operations: pd.DataFrame,
) -> pd.DataFrame:
    """Replicate routed loads across the FFN stages observed in each run.

    Zero-load experts do not emit expert_compute records, so the complete load
    grid comes from expert_compute_total. Operations are only replicated for a
    run/phase where that operation was actually observed.
    """
    if operations.empty or "operation" not in operations.columns:
        return pd.DataFrame()
    keys = ["run", "phase", "layer", "expert", "load"]
    totals_grid = totals[keys].drop_duplicates()
    stages = operations[["run", "phase", "operation"]].drop_duplicates()
    return totals_grid.merge(stages, on=["run", "phase"], how="inner")


def plot_ffn_stage_load_heatmaps(
    totals: pd.DataFrame,
    operations: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    stage_loads = complete_stage_loads(totals, operations)
    order = ordered_operations(stage_loads)
    if stage_loads.empty or not order:
        print(
            "warning: skipping ffn-stage-load-heatmap; no stage load records found",
            file=sys.stderr,
        )
        return

    layers = sorted(totals["layer"].dropna().astype(int).unique())
    experts = sorted(totals["expert"].dropna().astype(int).unique())
    maximum_load = float(stage_loads["load"].max())
    fig, axes = plt.subplots(
        len(order),
        1,
        figsize=(max(13, len(experts) * 0.42), max(4.2, len(order) * 4.0)),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    for index, (ax, operation) in enumerate(zip(axes_array, order)):
        part = stage_loads[stage_loads["operation"] == operation]
        matrix = (
            part.groupby(["layer", "expert"])["load"]
            .mean()
            .unstack("expert")
            .reindex(index=layers, columns=experts, fill_value=0)
        )
        sns.heatmap(
            matrix,
            cmap="rocket_r",
            vmin=0,
            vmax=maximum_load,
            ax=ax,
            cbar=index == 0,
            cbar_kws={"label": "Mean expert load"} if index == 0 else None,
        )
        ax.set(title=operation_label(operation), xlabel="", ylabel="Layer")
    axes_array[-1].set_xlabel("Expert position")
    fig.suptitle(
        "Per-expert routed load at each FFN stage\n"
        "Matching panels are expected: gate, up, and down consume the same routed rows",
        fontsize=15,
    )
    save_figure(fig, output / "09_expert_load_by_ffn_stage.png", dpi)


def build_summary(active: pd.DataFrame) -> pd.DataFrame:
    summary = (
        active.groupby(["phase", "layer", "expert"], as_index=False)
        .agg(
            samples=("cycle_delta", "size"),
            load_mean=("load", "mean"),
            load_std=("load", "std"),
            cycles_mean=("cycle_delta", "mean"),
            cycles_std=("cycle_delta", "std"),
            cycles_per_token_mean=("cycles_per_token", "mean"),
            cycles_per_token_std=("cycles_per_token", "std"),
            cycles_per_token_min=("cycles_per_token", "min"),
            cycles_per_token_max=("cycles_per_token", "max"),
        )
    )
    summary["cycles_per_token_cv"] = (
        summary["cycles_per_token_std"] / summary["cycles_per_token_mean"]
    )
    return summary


def write_text_summary(
    totals: pd.DataFrame,
    active: pd.DataFrame,
    paths: list[Path],
    stats: dict[str, float],
    heatmap_mode: str,
    output: Path,
) -> None:
    sources = ", ".join(sorted(str(value) for value in totals["cycle_source"].dropna().unique()))
    phases = ", ".join(sorted(str(value) for value in totals["phase"].dropna().unique()))
    normalized = active["cycles_per_token"]
    lines = [
        "MoE expert trace summary",
        "========================",
        f"Input logs: {len(paths)}",
        f"Phases: {phases}",
        f"Cycle sources: {sources}",
        f"Total expert records: {len(totals)}",
        f"Active expert records: {len(active)}",
        f"Layers: {active['layer'].nunique()}",
        f"Expert positions: {active['expert'].nunique()}",
        f"Pearson(load, cycles): {stats['pearson_r']:.8f}",
        f"Linear-fit R-squared: {stats['r_squared']:.8f}",
        f"Linear-fit slope: {stats['slope']:.6f}",
        f"Median cycles/load: {normalized.median():.6f}",
        f"Mean cycles/load: {normalized.mean():.6f}",
        f"Std. dev. cycles/load: {normalized.std():.6f}",
        f"Heatmap metric: {'run-to-run CV' if heatmap_mode == 'cv' else 'mean cycles/load (one sample per expert)'}",
        "",
        "A run-to-run CV heatmap requires at least two input logs containing",
        "measurements for the same layer/expert positions.",
    ]
    path = output / "summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    args = parse_args()
    if args.list_plots:
        print_plot_names()
        return 0

    paths = resolve_inputs(args.logs)
    if not paths:
        print("error: no readable input files; use --list-plots to see graph names", file=sys.stderr)
        return 2
    plots = selected_plots(args.plots)

    totals, operations = load_logs(paths)
    if totals.empty:
        print(
            "error: no expert_compute_total records found in the supplied files",
            file=sys.stderr,
        )
        return 2

    if args.phase.lower() != "all":
        totals = totals[totals["phase"] == args.phase].copy()
        if not operations.empty:
            operations = operations[operations["phase"] == args.phase].copy()
    if totals.empty:
        print(f"error: no total records found for phase={args.phase!r}", file=sys.stderr)
        return 2

    active = active_rows(totals)
    if active.empty:
        print("error: all matching expert measurements have zero load or zero cycles", file=sys.stderr)
        return 2

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    totals.to_csv(output / "expert_measurements.csv", index=False)
    print(f"wrote {output / 'expert_measurements.csv'}")
    summary = build_summary(active)
    summary.to_csv(output / "summary_by_layer_expert.csv", index=False)
    print(f"wrote {output / 'summary_by_layer_expert.csv'}")

    stats = regression_stats(active)
    heatmap_mode = "cv" if summary["samples"].max() >= 2 else "mean"
    if "overview" in plots:
        plot_overview(active, output, args.dpi)
    if "by-layer" in plots:
        plot_per_layer(active, output, args.dpi, not args.no_annotations)
    if "normalized-distribution" in plots:
        plot_normalized_distribution(active, output, args.dpi)
    if "by-load" in plots:
        plot_by_load(active, output, args.dpi, args.max_load_groups)
    if "by-layer-distribution" in plots:
        plot_by_layer(active, output, args.dpi)
    if "expert-heatmap" in plots:
        heatmap_mode = plot_consistency_heatmap(active, summary, output, args.dpi)
    if "operation-distribution" in plots:
        plot_operation_distribution(operations, output, args.dpi)
    if "ffn-stage-scatter" in plots:
        plot_ffn_stage_scatter(operations, output, args.dpi)
    if "ffn-stage-load-heatmap" in plots:
        plot_ffn_stage_load_heatmaps(totals, operations, output, args.dpi)
    write_text_summary(totals, active, paths, stats, heatmap_mode, output)

    print(
        f"\nParsed {len(totals)} totals ({len(active)} active) from "
        f"{len(paths)} log(s). Pearson r={stats['pearson_r']:.6f}; "
        f"R^2={stats['r_squared']:.6f}."
    )
    print(f"Selected graphs: {', '.join(name for name in PLOT_DESCRIPTIONS if name in plots)}")
    if heatmap_mode != "cv":
        print("Pass two or more repeated-run logs to obtain per-expert CV values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())