import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.lidar_pointcloud_dataset import LidarObstacleAEDataset
from models.autoencoder import PointNet2AutoEncoder
#from models.robot_gnn import build_robot_gnn_model
from models.gnn_global import build_robot_gnn_model


def load_ae_decoder(cfg, device):
    """
    GNN checkpoint의 cfg가 가리키는 AE checkpoint에서 decoder를 복원한다.

    train_gnn_latent.py의 AE 생성 인자와 동일하게 맞춰,
    AE 구조 변경에 따른 state_dict 불일치를 방지한다.
    """
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
    """
    checkpoint 내부 cfg를 그대로 사용해 GNN 구조를 재생성한다.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "cfg" not in ckpt:
        raise KeyError(f"{checkpoint_path}: checkpoint에 'cfg'가 없습니다.")
    if "model" not in ckpt:
        raise KeyError(f"{checkpoint_path}: checkpoint에 'model'이 없습니다.")

    cfg = ckpt["cfg"]
    use_robot_xy = bool(cfg["model"].get("use_robot_xy", False))

    decoder = load_ae_decoder(cfg, device)
    model = build_robot_gnn_model(cfg, decoder=decoder).to(device)

    state_dict = ckpt["model"]
    checkpoint_has_pose = any(
        key.startswith("pose_encoder.") or key.startswith("node_fusion.")
        for key in state_dict
    )
    built_has_pose = (
        getattr(model, "pose_encoder", None) is not None
        and getattr(model, "node_fusion", None) is not None
    )

    if checkpoint_has_pose != built_has_pose:
        raise RuntimeError(
            "Checkpoint와 현재 생성된 GNN 구조의 위치 계층 구성이 다릅니다.\n"
            f"  cfg use_robot_xy : {use_robot_xy}\n"
            f"  checkpoint pose  : {checkpoint_has_pose}\n"
            f"  built model pose : {built_has_pose}\n"
            "models/robot_gnn.py의 build_robot_gnn_model()이 "
            "use_robot_xy, robot_xy_dim, pose_hidden_dim을 "
            "ConsensusRobotGNN에 전달하는지 확인하세요."
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loaded checkpoint : {checkpoint_path}")
    print(f"use_robot_xy      : {use_robot_xy}")
    print(f"aggregator        : {cfg['model'].get('aggregator', 'unknown')}")
    print(f"checkpoint epoch  : {ckpt.get('epoch', 'unknown')}")

    return model, cfg


def get_shape_type(sample):
    meta = sample.get("meta", {})
    return meta.get("shape_type", "unknown")


def get_robot_positions(sample):
    """그림에 표시할 로봇 위치를 NumPy 배열로 반환한다."""
    robot_xy = sample.get("robot_xy")

    if robot_xy is None:
        robot_xy = sample.get("robot_pos")

    if robot_xy is None:
        return None

    if torch.is_tensor(robot_xy):
        robot_xy = robot_xy.detach().cpu().numpy()
    else:
        robot_xy = torch.as_tensor(
            robot_xy,
            dtype=torch.float32,
        ).detach().cpu().numpy()

    if robot_xy.ndim != 2 or robot_xy.shape[1] < 2:
        raise ValueError(
            f"Robot positions must be [N, 2+] but got {robot_xy.shape}"
        )

    return robot_xy[:, :2]


def get_robot_xy_tensor(sample, device, required):
    """모델에 전달할 robot_xy tensor를 준비한다."""
    robot_xy = sample.get("robot_xy")

    if robot_xy is None:
        robot_xy = sample.get("robot_pos")

    if robot_xy is None:
        if required:
            raise KeyError(
                "이 checkpoint는 use_robot_xy=True이지만 "
                "Dataset sample에 'robot_xy'가 없습니다."
            )
        return None

    robot_xy = torch.as_tensor(
        robot_xy,
        dtype=torch.float32,
        device=device,
    )

    if robot_xy.dim() != 2 or robot_xy.size(-1) != 2:
        raise ValueError(
            f"robot_xy must be [N, 2], got {tuple(robot_xy.shape)}"
        )

    return robot_xy


def collect_indices_by_shape(ds, per_shape=4, start_index=0):
    shape_to_indices = defaultdict(list)

    for idx in range(start_index, len(ds)):
        sample = ds[idx]
        shape_type = get_shape_type(sample)

        if len(shape_to_indices[shape_type]) < per_shape:
            shape_to_indices[shape_type].append(idx)

    selected_indices = []
    for shape_type in sorted(shape_to_indices.keys()):
        selected_indices.extend(shape_to_indices[shape_type])

    return selected_indices, shape_to_indices


def plot_one_sample(
    sample,
    pred_nodes,
    info,
    save_path,
    sample_idx,
    max_robots=None,
):
    x = sample["x"].cpu().numpy()
    target = sample["target"].cpu().numpy()
    pred_nodes = pred_nodes.detach().cpu().numpy()

    num_robots = x.shape[0]

    if max_robots is None:
        robot_indices = list(range(num_robots))
    else:
        robot_indices = list(range(min(max_robots, num_robots)))

    n_show = len(robot_indices)
    if n_show == 0:
        raise ValueError(f"Sample {sample_idx} contains no robot nodes.")

    meta = sample.get("meta", {})
    shape_type = meta.get("shape_type", "unknown")
    obstacle_scale = meta.get("obstacle_scale", "unknown")

    robot_xy = get_robot_positions(sample)

    fig = plt.figure(figsize=(12, max(6, 2.3 * n_show)))
    gs = fig.add_gridspec(
        nrows=n_show,
        ncols=2,
        width_ratios=[1.45, 1.0],
        wspace=0.25,
        hspace=0.35,
    )

    env_ax = fig.add_subplot(gs[:, 0])

    env_ax.scatter(
        target[:, 0],
        target[:, 1],
        s=10,
        label="target obstacle",
        alpha=0.75,
    )

    for robot_idx in range(num_robots):
        partial = x[robot_idx]
        env_ax.scatter(
            partial[:, 0],
            partial[:, 1],
            s=16,
            marker="x",
            alpha=0.55,
        )

    if robot_xy is not None:
        if len(robot_xy) != num_robots:
            print(
                f"[WARN] Sample {sample_idx}: "
                f"x robots={num_robots}, robot_xy={len(robot_xy)}"
            )

        env_ax.scatter(
            robot_xy[:, 0],
            robot_xy[:, 1],
            s=55,
            marker="o",
            facecolors="none",
            edgecolors="black",
            label="robots",
        )

        for robot_idx in range(min(num_robots, len(robot_xy))):
            env_ax.text(
                robot_xy[robot_idx, 0],
                robot_xy[robot_idx, 1],
                str(robot_idx),
                fontsize=8,
                ha="center",
                va="center",
            )
    else:
        print(
            f"[WARN] Sample {sample_idx}: robot position not found. "
            "Check LidarObstacleAEDataset returns 'robot_xy'."
        )

    env_ax.set_title("Environment: obstacle + robots + observed point clouds")
    env_ax.set_aspect("equal", adjustable="box")
    env_ax.grid(True, alpha=0.25)
    env_ax.legend(loc="best")

    for row, robot_idx in enumerate(robot_indices):
        ax = fig.add_subplot(gs[row, 1])

        partial = x[robot_idx]
        pred = pred_nodes[robot_idx]

        ax.scatter(
            target[:, 0],
            target[:, 1],
            s=8,
            label="target",
            alpha=0.45,
        )
        ax.scatter(
            partial[:, 0],
            partial[:, 1],
            s=14,
            marker="x",
            label="partial",
            alpha=0.75,
        )
        ax.scatter(
            pred[:, 0],
            pred[:, 1],
            s=8,
            label="recon",
            alpha=0.85,
        )

        ax.set_title(f"Robot {robot_idx} reconstruction")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        if row == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Sample {sample_idx} | shape={shape_type} | scale={obstacle_scale} | "
        f"steps={info.get('used_steps')} | converged={info.get('converged')}",
        fontsize=13,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--per_shape", type=int, default=4)
    p.add_argument("--save_dir", default="out/robot_gnn_vis_by_shape")
    p.add_argument("--max_robots", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_robot_gnn(args.checkpoint, device)
    ds = build_dataset(cfg, args.data_dir, seed=args.seed)

    use_robot_xy = bool(cfg["model"].get("use_robot_xy", False))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    selected_indices, shape_to_indices = collect_indices_by_shape(
        ds,
        per_shape=args.per_shape,
        start_index=args.start_index,
    )

    print("Selected samples by shape:")
    for shape_type in sorted(shape_to_indices.keys()):
        print(f"  {shape_type}: {shape_to_indices[shape_type]}")

    for i in selected_indices:
        sample = ds[i]

        x = sample["x"].to(device)
        edge_index = sample["edge_index"].to(device)
        robot_xy = get_robot_xy_tensor(
            sample,
            device=device,
            required=use_robot_xy,
        )

        if robot_xy is not None and robot_xy.size(0) != x.size(0):
            raise ValueError(
                f"Sample {i}: robot count mismatch: "
                f"x={x.size(0)}, robot_xy={robot_xy.size(0)}"
            )

        with torch.no_grad():
            pred_nodes, final_h, info = model(
                x,
                edge_index,
                robot_xy=robot_xy,
            )

        shape_type = get_shape_type(sample)
        shape_dir = save_dir / str(shape_type)
        shape_dir.mkdir(parents=True, exist_ok=True)

        save_path = shape_dir / f"robot_gnn_recon_{shape_type}_{i:04d}.png"

        plot_one_sample(
            sample=sample,
            pred_nodes=pred_nodes,
            info=info,
            save_path=save_path,
            sample_idx=i,
            max_robots=args.max_robots,
        )

        print(
            f"Saved: {save_path} | "
            f"steps={info.get('used_steps')} | "
            f"converged={info.get('converged')} | "
            f"gap_history={info.get('gap_history')}"
        )


if __name__ == "__main__":
    main()