import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.lidar_pointcloud_dataset import LidarObstacleAEDataset
from losses.chamfer import chamfer_distance
from models.autoencoder import PointNet2AutoEncoder
from models.gnn_global import build_robot_gnn_model


def load_ae_decoder(cfg, device):
    """GNN checkpoint가 참조하는 AE checkpoint에서 decoder를 복원한다."""
    ae_ckpt_path = cfg["ae"]["checkpoint"]
    ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_cfg = ckpt["cfg"]

    ae = PointNet2AutoEncoder(
        encoder_mode=ae_cfg["model"].get("encoder_mode", "ssg"),
        decoder_mode=ae_cfg["model"].get("decoder_mode", "mlp"),
        latent_dim=ae_cfg["model"]["latent_dim"],
        input_channels=ae_cfg["model"].get("input_channels", 0),
        target_num_points=ae_cfg["data"]["target_num_points"],
        output_dim=ae_cfg["model"].get("output_dim", 3),
        base_radius=ae_cfg["model"].get("base_radius", 1.0),
        npoint1=ae_cfg["model"].get("npoint1", 32),
        npoint2=ae_cfg["model"].get("npoint2", 16),
        hidden_dim=ae_cfg["model"].get("hidden_dim", 128),
        k_cov=ae_cfg["model"].get("k_cov", 32),
        k_agg=ae_cfg["model"].get("k_agg", 16),
        use_attention=ae_cfg["model"].get("use_attention", True),
        decoder_hidden_dim=ae_cfg["model"].get("decoder_hidden_dim", 512),
        folding_grid_dim=ae_cfg["model"].get("folding_grid_dim", 1),
        folding_num_folds=ae_cfg["model"].get("folding_num_folds", 2),
    ).to(device)

    ae.load_state_dict(ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    decoder = ae.decoder
    decoder.eval()
    return decoder


def build_dataset(cfg, data_dir, seed=0):
    data_cfg = cfg["data"]
    return LidarObstacleAEDataset(
        data_dir=data_dir,
        input_num_points=data_cfg["input_num_points"],
        target_num_points=data_cfg["target_num_points"],
        include_miss=data_cfg.get("include_miss", False),
        normalize=data_cfg.get("normalize", True),
        seed=seed,
        use_world_frame=data_cfg.get("use_world_frame", True),
    )


def load_robot_gnn(checkpoint_path, device):
    """학습과 동일한 models.gnn_global builder로 GNN을 복원한다."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "cfg" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain both 'cfg' and 'model'.")

    cfg = ckpt["cfg"]
    decoder = load_ae_decoder(cfg, device)
    model = build_robot_gnn_model(cfg, decoder=decoder).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print("=" * 80)
    print("Loaded model")
    print("=" * 80)
    print(f"checkpoint   : {checkpoint_path}")
    print(f"epoch        : {ckpt.get('epoch', 'unknown')}")
    print(f"use_robot_xy : {cfg['model'].get('use_robot_xy', False)}")
    print(f"aggregator   : {cfg['model'].get('aggregator', 'unknown')}")
    print(f"AE checkpoint: {cfg['ae']['checkpoint']}")
    print()
    return model, cfg


def get_robot_xy_tensor(sample, device, required):
    robot_xy = sample.get("robot_xy")
    if robot_xy is None:
        robot_xy = sample.get("robot_pos")

    if robot_xy is None:
        if required:
            raise KeyError(
                "This checkpoint uses robot_xy, but the sample has no robot_xy."
            )
        return None

    robot_xy = torch.as_tensor(robot_xy, dtype=torch.float32, device=device)
    if robot_xy.dim() != 2 or robot_xy.size(-1) != 2:
        raise ValueError(
            f"robot_xy must be [N, 2], got {tuple(robot_xy.shape)}"
        )
    return robot_xy


def get_shape_type(sample):
    return str(sample.get("meta", {}).get("shape_type", "unknown"))


def get_obstacle_scale(sample):
    meta = sample.get("meta", {})
    return meta.get("obstacle_scale", meta.get("scale", "unknown"))


@torch.no_grad()
def single_chamfer(pred, target):
    """pred [Qp,3], target [Qt,3]에 대한 scalar Chamfer."""
    return float(
        chamfer_distance(
            pred.unsqueeze(0),
            target.unsqueeze(0),
        ).item()
    )


@torch.no_grad()
def evaluate_one_sample(model, sample, device, alphas):
    """
    z(t) = (1-t) z_AE + t z_GNN 을 decoder에 넣는다.

    z_GNN은 이전 분석의 gnn_latent_mean과 동일하게
    final_h의 robot-node 평균을 사용한다.
    """
    x = sample["x"].to(device)
    edge_index = sample["edge_index"].to(device)
    target = sample["target"].to(device)

    if "target_latent" not in sample:
        raise KeyError(
            "Sample has no target_latent. Use the latent-enriched dataset."
        )

    z_ae = sample["target_latent"].to(device).view(-1)
    robot_xy = get_robot_xy_tensor(
        sample,
        device=device,
        required=bool(model.use_robot_xy),
    )

    if robot_xy is not None and robot_xy.size(0) != x.size(0):
        raise ValueError(
            f"robot count mismatch: x={x.size(0)}, robot_xy={robot_xy.size(0)}"
        )

    _, final_h, info = model(
        x,
        edge_index,
        robot_xy=robot_xy,
    )

    if final_h.dim() != 2:
        raise ValueError(f"final_h must be [N,D], got {tuple(final_h.shape)}")

    z_gnn = final_h.mean(dim=0)
    if z_ae.shape != z_gnn.shape:
        raise ValueError(
            f"latent mismatch: AE={tuple(z_ae.shape)}, GNN={tuple(z_gnn.shape)}"
        )

    z_batch = torch.stack(
        [(1.0 - float(t)) * z_ae + float(t) * z_gnn for t in alphas],
        dim=0,
    )
    decoded = model.decoder(z_batch)

    if decoded.dim() != 3:
        raise ValueError(
            f"decoder output must be [K,Q,3], got {tuple(decoded.shape)}"
        )

    chamfers = [single_chamfer(decoded[k], target) for k in range(len(alphas))]
    baseline = chamfers[0]
    final = chamfers[-1]
    deltas_from_ae = [v - baseline for v in chamfers]
    step_deltas = [0.0] + [
        chamfers[k] - chamfers[k - 1] for k in range(1, len(chamfers))
    ]

    degradation_abs = final - baseline
    degradation_pct = (
        100.0 * degradation_abs / baseline
        if abs(baseline) > 1e-12
        else float("nan")
    )

    latent_l2 = float(torch.norm(z_gnn - z_ae).item())
    latent_rmse = float(torch.sqrt(torch.mean((z_gnn - z_ae) ** 2)).item())
    latent_cosine = float(
        F.cosine_similarity(
            z_ae.unsqueeze(0), z_gnn.unsqueeze(0), dim=-1
        ).item()
    )
    latent_norm_ratio = float(
        z_gnn.norm().item() / max(z_ae.norm().item(), 1e-12)
    )

    monotonic = all(
        chamfers[k] >= chamfers[k - 1] - 1e-12
        for k in range(1, len(chamfers))
    )

    return {
        "alphas": [float(t) for t in alphas],
        "chamfers": chamfers,
        "deltas_from_ae": deltas_from_ae,
        "step_deltas": step_deltas,
        "ae_chamfer": baseline,
        "gnn_chamfer": final,
        "degradation_abs": degradation_abs,
        "degradation_pct": degradation_pct,
        "monotonic_non_decreasing": monotonic,
        "latent_l2": latent_l2,
        "latent_rmse": latent_rmse,
        "latent_cosine": latent_cosine,
        "latent_norm_ratio": latent_norm_ratio,
        "num_robots": int(final_h.size(0)),
        "used_steps": info.get("used_steps"),
        "converged": info.get("converged"),
    }


def make_alphas(num_intermediate):
    if num_intermediate < 0:
        raise ValueError("num_intermediate must be >= 0")
    return np.linspace(0.0, 1.0, num_intermediate + 2).tolist()


def finite_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else float("nan")


def save_outputs(rows, summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "interpolation_results.json"
    csv_path = out_dir / "interpolation_results.csv"
    summary_path = out_dir / "summary.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    flat_rows = []
    for row in rows:
        flat = {
            "sample_index": row["sample_index"],
            "shape_type": row["shape_type"],
            "obstacle_scale": row["obstacle_scale"],
            "num_robots": row["num_robots"],
            "latent_l2": row["latent_l2"],
            "latent_rmse": row["latent_rmse"],
            "latent_cosine": row["latent_cosine"],
            "latent_norm_ratio": row["latent_norm_ratio"],
            "ae_chamfer": row["ae_chamfer"],
            "gnn_chamfer": row["gnn_chamfer"],
            "degradation_abs": row["degradation_abs"],
            "degradation_pct": row["degradation_pct"],
            "monotonic_non_decreasing": row["monotonic_non_decreasing"],
            "used_steps": row["used_steps"],
            "converged": row["converged"],
        }
        for k, alpha in enumerate(row["alphas"]):
            flat[f"alpha_{k}"] = alpha
            flat[f"chamfer_{k}"] = row["chamfers"][k]
            flat[f"delta_from_ae_{k}"] = row["deltas_from_ae"][k]
            flat[f"step_delta_{k}"] = row["step_deltas"][k]
        flat_rows.append(flat)

    if flat_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return json_path, csv_path, summary_path


def main():
    p = argparse.ArgumentParser(
        description=(
            "Interpolate from AE latent to GNN latent and measure decoded Chamfer."
        )
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="out/latent_interpolation_chamfer")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 means all samples after start_index.",
    )
    p.add_argument(
        "--num_intermediate",
        type=int,
        default=2,
        help="2 gives t=[0, 1/3, 2/3, 1].",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_robot_gnn(args.checkpoint, device)
    ds = build_dataset(cfg, args.data_dir, seed=args.seed)
    alphas = make_alphas(args.num_intermediate)

    start = max(0, args.start_index)
    end = len(ds)
    if args.max_samples > 0:
        end = min(end, start + args.max_samples)
    if start >= end:
        raise ValueError(f"No samples selected: start={start}, end={end}")

    print("=" * 80)
    print("Latent interpolation setup")
    print("=" * 80)
    print(f"data_dir         : {args.data_dir}")
    print(f"sample range     : [{start}, {end})")
    print(f"num samples      : {end - start}")
    print(f"num intermediate : {args.num_intermediate}")
    print("alphas           : " + ", ".join(f"{t:.6f}" for t in alphas))
    print("GNN latent       : mean(final_h, dim=0)")
    print()

    rows = []
    for idx in range(start, end):
        sample = ds[idx]
        result = evaluate_one_sample(model, sample, device, alphas)
        result["sample_index"] = idx
        result["shape_type"] = get_shape_type(sample)
        result["obstacle_scale"] = get_obstacle_scale(sample)
        rows.append(result)

        print("-" * 80)
        print(
            f"[sample {idx:04d}] shape={result['shape_type']} | "
            f"scale={result['obstacle_scale']} | robots={result['num_robots']}"
        )
        print(
            f"latent: L2={result['latent_l2']:.6f} | "
            f"RMSE={result['latent_rmse']:.6f} | "
            f"cos={result['latent_cosine']:.6f} | "
            f"norm_ratio={result['latent_norm_ratio']:.6f}"
        )

        for k, alpha in enumerate(result["alphas"]):
            label = "AE" if k == 0 else (
                "GNN" if k == len(result["alphas"]) - 1 else f"interp-{k}"
            )
            print(
                f"  {label:8s} t={alpha:.6f} | "
                f"Chamfer={result['chamfers'][k]:.8f} | "
                f"delta_from_AE={result['deltas_from_ae'][k]:+.8f} | "
                f"step_delta={result['step_deltas'][k]:+.8f}"
            )

        print(
            "  AE -> GNN degradation | "
            f"absolute={result['degradation_abs']:+.8f} | "
            f"percent={result['degradation_pct']:+.3f}% | "
            f"monotonic={result['monotonic_non_decreasing']}"
        )

    chamfer_matrix = np.asarray([r["chamfers"] for r in rows])
    delta_matrix = np.asarray([r["deltas_from_ae"] for r in rows])
    step_matrix = np.asarray([r["step_deltas"] for r in rows])

    mean_cd = chamfer_matrix.mean(axis=0)
    std_cd = chamfer_matrix.std(axis=0)
    mean_delta = delta_matrix.mean(axis=0)
    mean_step = step_matrix.mean(axis=0)

    summary = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "num_samples": len(rows),
        "num_intermediate": args.num_intermediate,
        "alphas": alphas,
        "gnn_latent_definition": "mean(final_h, dim=0)",
        "mean_chamfer_by_alpha": mean_cd.tolist(),
        "std_chamfer_by_alpha": std_cd.tolist(),
        "mean_delta_from_ae_by_alpha": mean_delta.tolist(),
        "mean_step_delta_by_alpha": mean_step.tolist(),
        "mean_ae_chamfer": finite_mean([r["ae_chamfer"] for r in rows]),
        "mean_gnn_chamfer": finite_mean([r["gnn_chamfer"] for r in rows]),
        "mean_degradation_abs": finite_mean([r["degradation_abs"] for r in rows]),
        "std_degradation_abs": finite_std([r["degradation_abs"] for r in rows]),
        "mean_degradation_pct": finite_mean([r["degradation_pct"] for r in rows]),
        "std_degradation_pct": finite_std([r["degradation_pct"] for r in rows]),
        "fraction_gnn_worse_than_ae": float(np.mean([
            r["gnn_chamfer"] > r["ae_chamfer"] for r in rows
        ])),
        "fraction_monotonic_non_decreasing": float(np.mean([
            r["monotonic_non_decreasing"] for r in rows
        ])),
        "mean_latent_l2": finite_mean([r["latent_l2"] for r in rows]),
        "mean_latent_rmse": finite_mean([r["latent_rmse"] for r in rows]),
        "mean_latent_cosine": finite_mean([r["latent_cosine"] for r in rows]),
        "mean_latent_norm_ratio": finite_mean([r["latent_norm_ratio"] for r in rows]),
    }

    print()
    print("=" * 80)
    print("Overall interpolation summary")
    print("=" * 80)
    for k, alpha in enumerate(alphas):
        print(
            f"t={alpha:.6f} | mean Chamfer={mean_cd[k]:.8f} ± {std_cd[k]:.8f} | "
            f"mean delta_from_AE={mean_delta[k]:+.8f} | "
            f"mean step_delta={mean_step[k]:+.8f}"
        )

    print()
    print(f"mean AE Chamfer               : {summary['mean_ae_chamfer']:.8f}")
    print(f"mean GNN Chamfer              : {summary['mean_gnn_chamfer']:.8f}")
    print(
        f"mean absolute degradation     : "
        f"{summary['mean_degradation_abs']:+.8f} ± "
        f"{summary['std_degradation_abs']:.8f}"
    )
    print(
        f"mean percent degradation      : "
        f"{summary['mean_degradation_pct']:+.3f}% ± "
        f"{summary['std_degradation_pct']:.3f}%"
    )
    print(
        f"fraction GNN worse than AE    : "
        f"{summary['fraction_gnn_worse_than_ae']:.2%}"
    )
    print(
        f"fraction monotonic degradation: "
        f"{summary['fraction_monotonic_non_decreasing']:.2%}"
    )
    print(f"mean latent L2                : {summary['mean_latent_l2']:.6f}")
    print(f"mean latent cosine            : {summary['mean_latent_cosine']:.6f}")
    print(f"mean latent norm ratio        : {summary['mean_latent_norm_ratio']:.6f}")

    out_dir = Path(args.out_dir)
    json_path, csv_path, summary_path = save_outputs(rows, summary, out_dir)
    print()
    print(f"Saved JSON   : {json_path}")
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()