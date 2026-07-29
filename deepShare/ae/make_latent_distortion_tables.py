#!/usr/bin/env python3
"""Create two presentation-ready table images from latent-analysis CSV files.

Outputs
-------
out/latent_distribution/
    latent_distribution_table.png
    latent_distribution_table.csv
    latent_distribution_summary.csv

out/latent_interpolation_distribution/
    latent_interpolation_distribution_table.png
    latent_interpolation_distribution_table.csv
    latent_interpolation_summary.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    mask = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
    if int(mask.sum()) < 2:
        return float("nan")
    aa = a.to_numpy(dtype=float)[mask]
    bb = b.to_numpy(dtype=float)[mask]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(aa, bb)[0, 1])
    if method == "spearman":
        ra = pd.Series(aa).rank(method="average").to_numpy()
        rb = pd.Series(bb).rank(method="average").to_numpy()
        return float(np.corrcoef(ra, rb)[0, 1])
    raise ValueError(f"Unknown correlation method: {method}")


def fmt_float(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def fmt_signed(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):+.{digits}f}"


def fmt_percent(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):+.{digits}f}%"


def save_table_figure(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    """Save one PNG containing a compact summary table and a detailed table."""
    n_detail = len(detail_df)
    fig_height = max(9.0, 5.0 + 0.42 * n_detail)
    fig = plt.figure(figsize=(18, fig_height))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, max(1.2, n_detail / 7.0)])

    ax0 = fig.add_subplot(grid[0])
    ax0.axis("off")
    ax0.set_title(title, fontsize=17, pad=22)
    ax0.text(
        0.5,
        1.01,
        subtitle,
        transform=ax0.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    summary_table = ax0.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)
    summary_table.scale(1.0, 1.35)
    for col in range(len(summary_df.columns)):
        summary_table.auto_set_column_width(col)

    ax1 = fig.add_subplot(grid[1])
    ax1.axis("off")
    detail_table = ax1.table(
        cellText=detail_df.values,
        colLabels=detail_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    detail_table.auto_set_font_size(False)
    detail_table.set_fontsize(8)
    detail_table.scale(1.0, 1.22)
    for col in range(len(detail_df.columns)):
        detail_table.auto_set_column_width(col)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_distribution_table(csv_path: Path, out_dir: Path, top_k: int) -> None:
    required = {
        "shape_a",
        "shape_b",
        "ae_distance",
        "gnn_distance",
        "abs_difference",
        "signed_difference",
        "ratio_gnn_over_ae",
    }
    df = pd.read_csv(csv_path)
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Distribution CSV missing columns: {sorted(missing)}")

    for col in [
        "ae_distance",
        "gnn_distance",
        "abs_difference",
        "signed_difference",
        "ratio_gnn_over_ae",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ae_distance", "gnn_distance", "ratio_gnn_over_ae"]
    ).copy()
    if valid.empty:
        raise ValueError("No valid rows in distribution CSV.")

    ae = valid["ae_distance"].to_numpy(dtype=float)
    gnn = valid["gnn_distance"].to_numpy(dtype=float)
    ratio = valid["ratio_gnn_over_ae"].to_numpy(dtype=float)

    raw_error = gnn - ae
    raw_rmse = float(np.sqrt(np.mean(raw_error**2)))
    mean_ae = float(np.mean(ae))
    mean_gnn = float(np.mean(gnn))

    denom = float(np.dot(ae, ae))
    optimal_scale = float(np.dot(ae, gnn) / denom) if denom > 1e-12 else float("nan")
    corrected_error = gnn - optimal_scale * ae
    corrected_rmse = float(np.sqrt(np.mean(corrected_error**2)))

    slope, intercept = np.polyfit(ae, gnn, deg=1)
    predicted = slope * ae + intercept
    ss_res = float(np.sum((gnn - predicted) ** 2))
    ss_tot = float(np.sum((gnn - np.mean(gnn)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    pearson = safe_corr(valid["ae_distance"], valid["gnn_distance"], "pearson")
    spearman = safe_corr(valid["ae_distance"], valid["gnn_distance"], "spearman")

    contracted_fraction = float(np.mean(ratio < 1.0))
    expanded_fraction = float(np.mean(ratio > 1.0))
    mean_scale_change_pct = float((mean_gnn / mean_ae - 1.0) * 100.0)

    summary_rows = [
        ("Shape pairs", str(len(valid)), "Number of centroid pairs"),
        ("Mean AE distance", fmt_float(mean_ae), "Reference latent-space distance"),
        ("Mean GNN distance", fmt_float(mean_gnn), "Predicted latent-space distance"),
        ("Mean distance change", fmt_percent(mean_scale_change_pct), "Negative means global contraction"),
        ("Optimal scale alpha", fmt_float(optimal_scale), "Best fit: GNN distance ≈ alpha × AE distance"),
        ("Median GNN/AE ratio", fmt_float(float(np.median(ratio))), "1.0 means exact scale preservation"),
        ("Ratio Q25 / Q75", f"{np.quantile(ratio, 0.25):.6f} / {np.quantile(ratio, 0.75):.6f}", "Spread of pairwise scale distortion"),
        ("Contracted pair fraction", f"{contracted_fraction:.2%}", "Pairs with GNN/AE ratio < 1"),
        ("Expanded pair fraction", f"{expanded_fraction:.2%}", "Pairs with GNN/AE ratio > 1"),
        ("Mean absolute difference", fmt_float(float(np.mean(np.abs(raw_error)))), "Mean absolute pairwise error"),
        ("Raw RMSE", fmt_float(raw_rmse), "Distance error before scale correction"),
        ("Scale-corrected RMSE", fmt_float(corrected_rmse), "Residual error after one global scale correction"),
        ("Raw / corrected NRMSE", f"{raw_rmse / mean_ae:.6f} / {corrected_rmse / mean_ae:.6f}", "RMSE normalized by mean AE distance"),
        ("Linear slope / intercept", f"{slope:.6f} / {intercept:.6f}", "Regression: GNN = slope × AE + intercept"),
        ("Linear R²", fmt_float(r_squared), "Explained pairwise-distance variance"),
        ("Pearson / Spearman", f"{pearson:.6f} / {spearman:.6f}", "Linear / rank-structure preservation"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value", "Interpretation"])

    valid["pair"] = valid["shape_a"].astype(str) + " / " + valid["shape_b"].astype(str)
    valid["scale_error_abs"] = (valid["ratio_gnn_over_ae"] - 1.0).abs()
    valid["scale_change_pct"] = (valid["ratio_gnn_over_ae"] - 1.0) * 100.0
    ranked = valid.sort_values(
        ["scale_error_abs", "abs_difference"], ascending=[False, False]
    ).reset_index(drop=True)

    export_cols = [
        "shape_a",
        "shape_b",
        "ae_distance",
        "gnn_distance",
        "signed_difference",
        "abs_difference",
        "ratio_gnn_over_ae",
        "scale_change_pct",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked[export_cols].to_csv(
        out_dir / "latent_distribution_table.csv", index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        out_dir / "latent_distribution_summary.csv", index=False, encoding="utf-8-sig"
    )

    top = ranked.head(top_k).copy()
    detail_df = pd.DataFrame({
        "Pair": top["pair"],
        "AE dist.": top["ae_distance"].map(lambda x: f"{x:.4f}"),
        "GNN dist.": top["gnn_distance"].map(lambda x: f"{x:.4f}"),
        "GNN-AE": top["signed_difference"].map(lambda x: f"{x:+.4f}"),
        "Ratio": top["ratio_gnn_over_ae"].map(lambda x: f"{x:.4f}"),
        "Scale change": top["scale_change_pct"].map(lambda x: f"{x:+.2f}%"),
    })

    save_table_figure(
        summary_df=summary_df,
        detail_df=detail_df,
        title="Latent Distribution Distortion",
        subtitle=f"Top {len(detail_df)} shape pairs ranked by |GNN/AE ratio - 1|",
        out_path=out_dir / "latent_distribution_table.png",
    )


def sorted_indexed_columns(columns: list[str], prefix: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix):]
        try:
            index = int(suffix)
        except ValueError:
            continue
        found.append((index, col))
    return [col for _, col in sorted(found)]


def make_interpolation_table(csv_path: Path, out_dir: Path, top_k: int) -> None:
    df = pd.read_csv(csv_path)
    required = {
        "sample_index",
        "shape_type",
        "latent_l2",
        "latent_rmse",
        "latent_cosine",
        "latent_norm_ratio",
        "ae_chamfer",
        "gnn_chamfer",
        "degradation_abs",
        "degradation_pct",
        "monotonic_non_decreasing",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Interpolation CSV missing columns: {sorted(missing)}")

    numeric_cols = [
        "sample_index",
        "latent_l2",
        "latent_rmse",
        "latent_cosine",
        "latent_norm_ratio",
        "ae_chamfer",
        "gnn_chamfer",
        "degradation_abs",
        "degradation_pct",
    ]
    alpha_cols = sorted_indexed_columns(list(df.columns), "alpha_")
    chamfer_cols = sorted_indexed_columns(list(df.columns), "chamfer_")
    delta_cols = sorted_indexed_columns(list(df.columns), "delta_from_ae_")
    for col in numeric_cols + alpha_cols + chamfer_cols + delta_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ae_chamfer", "gnn_chamfer", "degradation_abs", "degradation_pct"]
    ).copy()
    if valid.empty:
        raise ValueError("No valid rows in interpolation CSV.")

    mono = valid["monotonic_non_decreasing"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    gnn_worse = valid["gnn_chamfer"] > valid["ae_chamfer"]

    summary_rows = [
        ("Samples", str(len(valid)), "Number of evaluated obstacles"),
        ("Mean AE Chamfer", fmt_float(valid["ae_chamfer"].mean(), 8), "Decoder output from stored AE target latent"),
        ("Mean GNN Chamfer", fmt_float(valid["gnn_chamfer"].mean(), 8), "Decoder output from mean GNN node latent"),
        ("Mean signed change", fmt_signed(valid["degradation_abs"].mean(), 8), "GNN Chamfer - AE Chamfer"),
        ("Std. signed change", fmt_float(valid["degradation_abs"].std(ddof=0), 8), "Sample-to-sample variation"),
        ("Mean percent change", fmt_percent(valid["degradation_pct"].mean(), 3), "Positive means worse reconstruction"),
        ("Std. percent change", f"{valid['degradation_pct'].std(ddof=0):.3f}%", "Spread of reconstruction change"),
        ("GNN worse fraction", f"{gnn_worse.mean():.2%}", "Samples with GNN Chamfer > AE Chamfer"),
        ("Monotonic degradation fraction", f"{mono.mean():.2%}", "Chamfer never decreases along AE→GNN path"),
        ("Mean latent L2", fmt_float(valid["latent_l2"].mean()), "Distance between AE and mean GNN latent"),
        ("Mean latent RMSE", fmt_float(valid["latent_rmse"].mean()), "Per-dimension latent error"),
        ("Mean latent cosine", fmt_float(valid["latent_cosine"].mean()), "Directional similarity"),
        ("Mean latent norm ratio", fmt_float(valid["latent_norm_ratio"].mean()), "GNN norm / AE norm"),
        ("Corr. latent L2 vs |Chamfer change|", fmt_float(safe_corr(valid["latent_l2"], valid["degradation_abs"].abs())), "Whether larger latent error produces larger output change"),
    ]

    if alpha_cols and chamfer_cols and len(alpha_cols) == len(chamfer_cols):
        for alpha_col, chamfer_col in zip(alpha_cols, chamfer_cols):
            alpha_mean = valid[alpha_col].mean()
            cd_mean = valid[chamfer_col].mean()
            cd_std = valid[chamfer_col].std(ddof=0)
            summary_rows.append((
                f"Mean Chamfer at t={alpha_mean:.3f}",
                f"{cd_mean:.8f} ± {cd_std:.8f}",
                "Mean decoded error along interpolation path",
            ))

    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value", "Interpretation"])

    valid["abs_percent_change"] = valid["degradation_pct"].abs()
    ranked = valid.sort_values(
        ["abs_percent_change", "latent_l2"], ascending=[False, False]
    ).reset_index(drop=True)

    export_cols = [
        "sample_index",
        "shape_type",
        "obstacle_scale",
        "num_robots",
        "latent_l2",
        "latent_rmse",
        "latent_cosine",
        "latent_norm_ratio",
        "ae_chamfer",
    ]
    export_cols += chamfer_cols[1:-1]
    export_cols += [
        "gnn_chamfer",
        "degradation_abs",
        "degradation_pct",
        "abs_percent_change",
        "monotonic_non_decreasing",
        "used_steps",
        "converged",
    ]
    export_cols = [c for c in export_cols if c in ranked.columns]

    out_dir.mkdir(parents=True, exist_ok=True)
    ranked[export_cols].to_csv(
        out_dir / "latent_interpolation_distribution_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        out_dir / "latent_interpolation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top = ranked.head(top_k).copy()
    table_data: dict[str, pd.Series] = {
        "Sample": top["sample_index"].map(lambda x: str(int(x))),
        "Shape": top["shape_type"].astype(str),
        "Latent L2": top["latent_l2"].map(lambda x: f"{x:.4f}"),
        "Cosine": top["latent_cosine"].map(lambda x: f"{x:.4f}"),
        "Norm ratio": top["latent_norm_ratio"].map(lambda x: f"{x:.4f}"),
        "AE CD": top["ae_chamfer"].map(lambda x: f"{x:.5f}"),
    }
    for idx, col in enumerate(chamfer_cols[1:-1], start=1):
        alpha_col = alpha_cols[idx] if idx < len(alpha_cols) else None
        if alpha_col is not None:
            alpha_value = top[alpha_col].mean()
            label = f"t={alpha_value:.3f}"
        else:
            label = f"Interp {idx}"
        table_data[label] = top[col].map(lambda x: f"{x:.5f}")
    table_data["GNN CD"] = top["gnn_chamfer"].map(lambda x: f"{x:.5f}")
    table_data["Delta"] = top["degradation_abs"].map(lambda x: f"{x:+.5f}")
    table_data["Change"] = top["degradation_pct"].map(lambda x: f"{x:+.2f}%")
    table_data["Monotonic"] = top["monotonic_non_decreasing"].astype(str)
    detail_df = pd.DataFrame(table_data)

    save_table_figure(
        summary_df=summary_df,
        detail_df=detail_df,
        title="Latent Interpolation Distortion",
        subtitle=f"Top {len(detail_df)} samples ranked by absolute AE→GNN Chamfer change (%)",
        out_path=out_dir / "latent_interpolation_distribution_table.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution_csv",
        default="out/latent_distribution_analysis/test/pairwise_shape_distances.csv",
    )
    parser.add_argument(
        "--interpolation_csv",
        default="out/latent_interpolation_chamfer/with_position/interpolation_results.csv",
    )
    parser.add_argument(
        "--distribution_out",
        default="out/latent_distribution",
    )
    parser.add_argument(
        "--interpolation_out",
        default="out/latent_interpolation_distribution",
    )
    parser.add_argument("--top_k", type=int, default=12)
    args = parser.parse_args()

    distribution_csv = Path(args.distribution_csv)
    interpolation_csv = Path(args.interpolation_csv)
    if not distribution_csv.is_file():
        raise FileNotFoundError(distribution_csv)
    if not interpolation_csv.is_file():
        raise FileNotFoundError(interpolation_csv)

    make_distribution_table(
        distribution_csv,
        Path(args.distribution_out),
        max(1, args.top_k),
    )
    make_interpolation_table(
        interpolation_csv,
        Path(args.interpolation_out),
        max(1, args.top_k),
    )

    print("Created:")
    print(Path(args.distribution_out) / "latent_distribution_table.png")
    print(Path(args.distribution_out) / "latent_distribution_table.csv")
    print(Path(args.distribution_out) / "latent_distribution_summary.csv")
    print(Path(args.interpolation_out) / "latent_interpolation_distribution_table.png")
    print(Path(args.interpolation_out) / "latent_interpolation_distribution_table.csv")
    print(Path(args.interpolation_out) / "latent_interpolation_summary.csv")


if __name__ == "__main__":
    main()