
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SELECTED = {
    "circle": [2],
    "cross": [3],
    "pentagon": [5],
    "projected_bathtub": [7, 8],
    "projected_bed": [9, 10],
    "projected_chair": [11, 12],
    "projected_desk": [13, 14],
    "projected_dresser": [15, 16],
    "projected_monitor": [17, 18],
    "projected_night_stand": [19, 20],
    "projected_sofa": [21, 22],
    "projected_table": [23, 24],
    "projected_toilet": [25, 26],
    "rectangle": [6],
    "star": [0],
    "triangle": [1],
    "u": [4],
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_latents(sample):
    obstacle = sample["obstacle"]
    analysis = obstacle["gnn_analysis"]

    z_ae = np.asarray(obstacle["ae_latent"], dtype=np.float64)
    z_gnn = np.asarray(analysis["gnn_latent_mean"], dtype=np.float64)

    if z_ae.ndim != 1 or z_gnn.ndim != 1:
        raise ValueError(f"Latent must be 1D: ae={z_ae.shape}, gnn={z_gnn.shape}")
    if z_ae.shape != z_gnn.shape:
        raise ValueError(f"Latent dimension mismatch: ae={z_ae.shape}, gnn={z_gnn.shape}")

    return z_ae, z_gnn


def parse_selected(path):
    if path is None:
        return DEFAULT_SELECTED
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): [int(x) for x in v] for k, v in obj.items()}


def collect_shape_latents(files, selected):
    shape_data = {}
    for shape, indices in selected.items():
        ae_list, gnn_list, used_files = [], [], []
        for idx in indices:
            if idx < 0 or idx >= len(files):
                raise IndexError(f"{shape}: index {idx} out of range for {len(files)} files")
            path = files[idx]
            z_ae, z_gnn = get_latents(load_json(path))
            ae_list.append(z_ae)
            gnn_list.append(z_gnn)
            used_files.append(path.name)
        shape_data[shape] = {
            "ae": np.stack(ae_list, axis=0),
            "gnn": np.stack(gnn_list, axis=0),
            "files": used_files,
        }
    return shape_data


def summarize_distribution(Z):
    return {
        "global_mean": float(Z.mean()),
        "global_std": float(Z.std(ddof=0)),
        "global_var": float(Z.var(ddof=0)),
        "dim_mean": Z.mean(axis=0),
        "dim_std": Z.std(axis=0, ddof=0),
        "dim_var": Z.var(axis=0, ddof=0),
    }


def shape_centroids(shape_data):
    return {
        shape: {
            "ae": data["ae"].mean(axis=0),
            "gnn": data["gnn"].mean(axis=0),
        }
        for shape, data in shape_data.items()
    }


def build_pairwise_rows(centroids):
    rows = []
    for a, b in combinations(centroids.keys(), 2):
        d_ae = float(np.linalg.norm(centroids[a]["ae"] - centroids[b]["ae"]))
        d_gnn = float(np.linalg.norm(centroids[a]["gnn"] - centroids[b]["gnn"]))
        rows.append({
            "shape_a": a,
            "shape_b": b,
            "ae_distance": d_ae,
            "gnn_distance": d_gnn,
            "abs_difference": abs(d_gnn - d_ae),
            "signed_difference": d_gnn - d_ae,
            "ratio_gnn_over_ae": d_gnn / d_ae if d_ae > 1e-12 else float("nan"),
        })
    return rows


def rankdata_simple(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def make_scatter(rows, out_path, annotate_top_k):
    ae = np.asarray([r["ae_distance"] for r in rows])
    gnn = np.asarray([r["gnn_distance"] for r in rows])

    pearson = float(np.corrcoef(ae, gnn)[0, 1])
    spearman = float(
        np.corrcoef(
            rankdata_simple(ae),
            rankdata_simple(gnn),
        )[0, 1]
    )

    lo = float(min(ae.min(), gnn.min()))
    hi = float(max(ae.max(), gnn.max()))

    # 각 클래스 쌍의 종류 구분
    categories = []

    for r in rows:
        shape_a = r["shape_a"]
        shape_b = r["shape_b"]

        a_is_projected = shape_a.startswith("projected_")
        b_is_projected = shape_b.startswith("projected_")

        if not a_is_projected and not b_is_projected:
            # 일반 클래스와 일반 클래스
            categories.append("nonprojected_pair")

        elif a_is_projected and b_is_projected:
            # projected 클래스와 projected 클래스
            categories.append("projected_pair")

        else:
            # projected 클래스와 일반 클래스
            categories.append("mixed_pair")

    categories = np.asarray(categories)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)

    # 1. projected가 붙지 않은 클래스 간 조합: 푸른 원
    mask = categories == "nonprojected_pair"
    ax.scatter(
        ae[mask],
        gnn[mask],
        marker="o",
        color="tab:blue",
        s=55,
        alpha=0.8,
        label="Non-projected / Non-projected",
    )

    # 2. projected와 non-projected 간 조합: 분홍 사각형
    mask = categories == "mixed_pair"
    ax.scatter(
        ae[mask],
        gnn[mask],
        marker="s",
        color="hotpink",
        s=55,
        alpha=0.8,
        label="Projected / Non-projected",
    )

    # 3. projected 클래스 간 조합: 연두색 별
    mask = categories == "projected_pair"
    ax.scatter(
        ae[mask],
        gnn[mask],
        marker="*",
        color="lightgreen",
        edgecolors="green",
        linewidths=0.5,
        s=120,
        alpha=0.85,
        label="Projected / Projected",
    )

    # AE 거리와 GNN 거리가 같은 기준선
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color="gray",
        linewidth=1.2,
        label="Equal distance",
    )

    ax.set_xlabel("AE latent pairwise distance")
    ax.set_ylabel("GNN latent pairwise distance")
    ax.set_title(
        "Shape-pair distance comparison\n"
        f"Pearson={pearson:.4f}, Spearman={spearman:.4f}"
    )

    ax.grid(True, alpha=0.3)

    # 개별 점 주석은 제거하고 범례만 오른쪽 아래에 표시
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "num_pairs": int(len(rows)),
        "pearson": pearson,
        "spearman": spearman,
        "mean_ae_distance": float(ae.mean()),
        "mean_gnn_distance": float(gnn.mean()),
        "mean_abs_difference": float(
            np.mean(np.abs(gnn - ae))
        ),
        "rmse_distance_difference": float(
            np.sqrt(np.mean((gnn - ae) ** 2))
        ),
    }


def make_dimension_plots(ae_stats, gnn_stats, out_dir):
    dims = np.arange(len(ae_stats["dim_mean"]))
    specs = [
        ("dim_mean", "Mean", "dimension_mean_comparison.png"),
        ("dim_std", "Standard deviation", "dimension_std_comparison.png"),
        ("dim_var", "Variance", "dimension_variance_comparison.png"),
    ]
    for key, ylabel, filename in specs:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax.plot(dims, ae_stats[key], label="AE")
        ax.plot(dims, gnn_stats[key], label="GNN")
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Per-dimension latent {ylabel.lower()}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_csv(rows, path):
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/gnn_latent_with_gnn/test_mixed")
    p.add_argument("--selected_json", default=None)
    p.add_argument("--out_dir", default="out/latent_distribution_analysis")
    p.add_argument("--annotate_top_k", type=int, default=10)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    selected = parse_selected(args.selected_json)
    shape_data = collect_shape_latents(files, selected)

    Z_ae = np.concatenate([shape_data[s]["ae"] for s in selected], axis=0)
    Z_gnn = np.concatenate([shape_data[s]["gnn"] for s in selected], axis=0)

    ae_stats = summarize_distribution(Z_ae)
    gnn_stats = summarize_distribution(Z_gnn)
    rows = build_pairwise_rows(shape_centroids(shape_data))
    scatter_summary = make_scatter(
        rows,
        out_dir / "ae_vs_gnn_pairwise_distance_scatter.png",
        args.annotate_top_k,
    )
    make_dimension_plots(ae_stats, gnn_stats, out_dir)
    save_csv(rows, out_dir / "pairwise_shape_distances.csv")

    summary = {
        "data_dir": str(data_dir),
        "num_json_files": len(files),
        "selected_samples": selected,
        "num_selected_samples": int(Z_ae.shape[0]),
        "latent_dim": int(Z_ae.shape[1]),
        "ae_distribution": jsonable(ae_stats),
        "gnn_distribution": jsonable(gnn_stats),
        "pairwise_distance_summary": scatter_summary,
        "shape_files": {shape: shape_data[shape]["files"] for shape in shape_data},
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Selected latent distribution")
    print("=" * 80)
    print(f"Samples    : {Z_ae.shape[0]}")
    print(f"Latent dim : {Z_ae.shape[1]}")
    print()
    print(
        "AE  global mean/std/var : "
        f"{ae_stats['global_mean']:.6f} / "
        f"{ae_stats['global_std']:.6f} / "
        f"{ae_stats['global_var']:.6f}"
    )
    print(
        "GNN global mean/std/var : "
        f"{gnn_stats['global_mean']:.6f} / "
        f"{gnn_stats['global_std']:.6f} / "
        f"{gnn_stats['global_var']:.6f}"
    )
    print()
    for k, v in scatter_summary.items():
        print(f"{k:28s}: {v}")
    print()
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()