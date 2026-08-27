#=========================================================================
# GNN 시각화 코드
# python visualize_robot_gnn.py \
#   --checkpoint 학습한 모델 \
#   --data_dir 데이터셋 \
#   --per_shape 각 카테고리별 검증 횟수 \
#   --max_robots 옆에 로봇 기준 시각화할 횟수 \
#   --save_dir 출력 위치
#==================================================================================

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.lidar_pointcloud_dataset import LidarObstacleAEDataset
from models.autoencoder import PointNet2AutoEncoder
from models.robot_gnn import build_robot_gnn_model

import numpy as np
from losses.chamfer import chamfer_distance


def load_ae_decoder(cfg, device):
    ae_ckpt_path = cfg["ae"]["checkpoint"]

    ckpt = torch.load(
        ae_ckpt_path,
        map_location=device,
    )

    ae_cfg = ckpt["cfg"]
    model_cfg = ae_cfg["model"]
    data_cfg = ae_cfg["data"]

    decoder_mode = model_cfg.get(
        "decoder_mode",
        "folding",
    )

    print(f"[AE] checkpoint   : {ae_ckpt_path}")
    print(f"[AE] decoder_mode : {decoder_mode}")

    ae = PointNet2AutoEncoder(
        encoder_mode=model_cfg.get(
            "encoder_mode",
            "ssg",
        ),

        decoder_mode=decoder_mode,

        latent_dim=model_cfg["latent_dim"],

        input_channels=model_cfg.get(
            "input_channels",
            0,
        ),

        target_num_points=data_cfg[
            "target_num_points"
        ],

        output_dim=model_cfg.get(
            "output_dim",
            3,
        ),

        base_radius=model_cfg.get(
            "base_radius",
            1.0,
        ),

        npoint1=model_cfg.get(
            "npoint1",
            32,
        ),

        npoint2=model_cfg.get(
            "npoint2",
            16,
        ),
    ).to(device)

    # AE encoder는 visualization에서 필요 없음.
    # decoder weight만 checkpoint에서 추출.
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in ckpt["model"].items()
        if key.startswith("decoder.")
    }

    if not decoder_state:
        raise RuntimeError(
            f"No decoder weights found in {ae_ckpt_path}"
        )

    ae.decoder.load_state_dict(
        decoder_state,
        strict=True,
    )

    decoder = ae.decoder
    decoder.eval()

    for p in decoder.parameters():
        p.requires_grad = False

    print(
        f"[AE] decoder class : "
        f"{decoder.__class__.__name__}"
    )

    return decoder
    
def build_random_decoder_from_config(cfg, device):
    """
    E2E visualization용.
    AE checkpoint weight는 사용하지 않고
    architecture_config를 이용해 동일한 decoder 구조만 생성한다.

    실제 학습된 decoder weight는 이후
    GNN checkpoint의 model state_dict에서 로드된다.
    """

    ae_config_path = cfg["ae"]["architecture_config"]

    import yaml

    with open(ae_config_path, "r", encoding="utf-8") as f:
        ae_cfg = yaml.safe_load(f)

    model_cfg = ae_cfg["model"]
    data_cfg = ae_cfg["data"]

    ae = PointNet2AutoEncoder(
        encoder_mode=model_cfg.get(
            "encoder_mode",
            "ssg",
        ),
        decoder_mode=model_cfg.get(
            "decoder_mode",
            "folding",
        ),
        latent_dim=model_cfg["latent_dim"],
        input_channels=model_cfg.get(
            "input_channels",
            0,
        ),
        target_num_points=data_cfg["target_num_points"],
        output_dim=model_cfg.get(
            "output_dim",
            3,
        ),
        base_radius=model_cfg.get(
            "base_radius",
            1.0,
        ),
        npoint1=model_cfg.get(
            "npoint1",
            32,
        ),
        npoint2=model_cfg.get(
            "npoint2",
            16,
        ),
        hidden_dim=model_cfg.get(
            "hidden_dim",
            128,
        ),
        k_cov=model_cfg.get(
            "k_cov",
            32,
        ),
        k_agg=model_cfg.get(
            "k_agg",
            16,
        ),
        use_attention=model_cfg.get(
            "use_attention",
            True,
        ),
        decoder_hidden_dim=model_cfg.get(
            "decoder_hidden_dim",
            512,
        ),
        folding_grid_dim=model_cfg.get(
            "folding_grid_dim",
            1,
        ),
        folding_num_folds=model_cfg.get(
            "folding_num_folds",
            2,
        ),
    ).to(device)

    decoder = ae.decoder

    print("[Decoder] mode  : end_to_end")
    print(
        f"[Decoder] class : "
        f"{decoder.__class__.__name__}"
    )

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
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )

    cfg = ckpt["cfg"]

    ae_mode = cfg.get(
        "ae",
        {},
    ).get(
        "mode",
        "pretrained",
    )

    # ========================================================
    # Frozen pretrained decoder
    # ========================================================

    if ae_mode == "pretrained":

        print(
            "[GNN] decoder mode : "
            "pretrained"
        )

        decoder = load_ae_decoder(
            cfg,
            device,
        )

    # ========================================================
    # End-to-End decoder
    # ========================================================

    elif ae_mode == "end_to_end":

        print(
            "[GNN] decoder mode : "
            "end_to_end"
        )

        decoder = build_random_decoder_from_config(
            cfg,
            device,
        )

    else:

        raise ValueError(
            f"Unknown ae.mode: {ae_mode}"
        )


    model = build_robot_gnn_model(
        cfg,
        decoder=decoder,
    ).to(device)


    # --------------------------------------------------------
    # 중요:
    #
    # Frozen:
    #   GNN weight + frozen decoder weight가 checkpoint에 저장됨
    #
    # E2E:
    #   GNN weight + 학습된 decoder weight가 checkpoint에 저장됨
    #
    # 따라서 둘 다 여기서 전체 state_dict를 로드하면 됨.
    # --------------------------------------------------------

    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.eval()

    print(
        f"[GNN] checkpoint : "
        f"{checkpoint_path}"
    )

    return model, cfg


def get_shape_type(sample):
    meta = sample.get("meta", {})
    return meta.get("shape_type", "unknown")


def get_robot_positions(sample):
    """
    dataset에 저장된 실제 로봇 위치만 사용.
    없으면 None 반환.
    """
    robot_pos = None

    if "robot_xy" in sample:
        robot_pos = sample["robot_xy"]
    elif "robot_pos" in sample:
        robot_pos = sample["robot_pos"]
    elif "robots" in sample:
        robot_pos = sample["robots"]

    if robot_pos is None:
        return None

    if torch.is_tensor(robot_pos):
        return robot_pos.detach().cpu().numpy()

    return torch.as_tensor(robot_pos, dtype=torch.float32).detach().cpu().numpy()


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

@torch.no_grad()
def compute_sample_metrics(pred_nodes, final_h, target, info):
    """
    pred_nodes : [N, Q, 3]
    final_h    : [N, D]
    target     : [Q, 3]

    train_gnn.py의 validation metric과 동일한 방식으로
    한 sample의 quantitative metric을 계산.
    """

    # --------------------------------------------------------
    # Node-wise Chamfer Distance
    # --------------------------------------------------------

    node_cd = []

    for i in range(pred_nodes.size(0)):
        cd = chamfer_distance(
            pred_nodes[i:i + 1],
            target.unsqueeze(0),
        )
        node_cd.append(float(cd.item()))

    node_cd = np.asarray(
        node_cd,
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Consensus statistics
    # --------------------------------------------------------

    mean_h = final_h.mean(
        dim=0,
        keepdim=True,
    )

    dist = torch.norm(
        final_h - mean_h,
        dim=1,
    )

    dist_np = (
        dist
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


    return {
        # train_gnn.py의 val_loss와 사실상 같은 의미
        "loss": float(node_cd.mean()),

        "node_chamfer_mean": float(node_cd.mean()),
        "node_chamfer_max": float(node_cd.max()),
        "node_chamfer_min": float(node_cd.min()),
        "node_chamfer_std": float(node_cd.std()),

        "consensus_gap": float(dist_np.max()),
        "consensus_mean": float(dist_np.mean()),
        "consensus_std": float(dist_np.std()),

        "steps": float(info.get("used_steps", 0)),
        "converged": float(bool(info.get("converged", False))),
    }

@torch.no_grad()
def evaluate_full_dataset(model, ds, device):
    """
    전체 test dataset에 대해 quantitative metric 계산.

    train_gnn.py validation evaluate와 동일하게
    sample별 metric을 계산한 뒤 sample 평균을 사용한다.
    """

    all_metrics = []

    print()
    print("======================================")
    print("Evaluating full dataset...")
    print(f"Samples: {len(ds)}")
    print("======================================")

    for i in range(len(ds)):
        sample = ds[i]

        x = sample["x"].to(device)
        edge_index = sample["edge_index"].to(device)
        target = sample["target"].to(device)

        pred_nodes, final_h, info = model(
            x,
            edge_index,
        )

        metrics = compute_sample_metrics(
            pred_nodes,
            final_h,
            target,
            info,
        )

        all_metrics.append(metrics)

        if (i + 1) % 100 == 0 or (i + 1) == len(ds):
            print(
                f"[Test] {i + 1}/{len(ds)}"
            )


    keys = [
        "loss",
        "node_chamfer_mean",
        "node_chamfer_max",
        "node_chamfer_min",
        "node_chamfer_std",
        "consensus_gap",
        "consensus_mean",
        "consensus_std",
        "steps",
        "converged",
    ]


    avg = {}

    for key in keys:
        avg[key] = float(
            np.mean([
                m[key]
                for m in all_metrics
            ])
        )


    return avg


def plot_one_sample(
    sample,
    pred_nodes,
    info,
    metrics,
    save_path,
    sample_idx,
    max_robots=None,
):
    x = sample["x"].cpu().numpy()                  # [N, P, 3]
    target = sample["target"].cpu().numpy()        # [Q, 3]
    pred_nodes = pred_nodes.detach().cpu().numpy() # [N, Q, 3]

    num_robots = x.shape[0]

    if max_robots is None:
        robot_indices = list(range(num_robots))
    else:
        robot_indices = list(range(min(max_robots, num_robots)))

    n_show = len(robot_indices)

    meta = sample.get("meta", {})
    shape_type = meta.get("shape_type", "unknown")
    obstacle_scale = meta.get("obstacle_scale", "unknown")

    robot_pos = get_robot_positions(sample)

    fig = plt.figure(figsize=(12, max(6, 2.3 * n_show)))
    gs = fig.add_gridspec(
        nrows=n_show,
        ncols=2,
        width_ratios=[1.45, 1.0],
        wspace=0.25,
        hspace=0.35,
    )

    # ========================================================
    # Left: 전체 환경
    # ========================================================
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

    if robot_pos is not None:
        env_ax.scatter(
            robot_pos[:, 0],
            robot_pos[:, 1],
            s=55,
            marker="o",
            facecolors="none",
            edgecolors="black",
            label="robots",
        )

        for robot_idx in range(min(num_robots, len(robot_pos))):
            env_ax.text(
                robot_pos[robot_idx, 0],
                robot_pos[robot_idx, 1],
                str(robot_idx),
                fontsize=8,
                ha="center",
                va="center",
            )
    else:
        print(
            f"[WARN] Sample {sample_idx}: robot position not found. "
            "Check LidarObstacleAEDataset returns 'robot_xy', 'robot_pos', or 'robots'."
        )

    env_ax.set_title("Environment: obstacle + robots + observed point clouds")
    env_ax.set_aspect("equal", adjustable="box")
    env_ax.grid(True, alpha=0.25)
    env_ax.legend(loc="best")

    # ========================================================
    # Right: 각 로봇별 reconstruction
    # ========================================================
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
        (
            f"Sample {sample_idx} | "
            f"shape={shape_type} | "
            f"scale={obstacle_scale}\n"

            f"CD mean={metrics['node_chamfer_mean']:.6f} | "
            f"max={metrics['node_chamfer_max']:.6f} | "
            f"min={metrics['node_chamfer_min']:.6f} | "
            f"std={metrics['node_chamfer_std']:.6f}\n"

            f"Consensus gap={metrics['consensus_gap']:.6f} | "
            f"mean={metrics['consensus_mean']:.6f} | "
            f"std={metrics['consensus_std']:.6f} | "
            f"steps={metrics['steps']:.0f} | "
            f"converged={bool(metrics['converged'])}"
        ),
        fontsize=12,
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

    # 기존 num_samples 대신 shape별 개수
    p.add_argument("--per_shape", type=int, default=4)

    p.add_argument("--save_dir", default="out/robot_gnn_vis_by_shape")
    p.add_argument("--max_robots", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_robot_gnn(args.checkpoint, device)
    ds = build_dataset(cfg, args.data_dir, seed=args.seed)

    test_metrics = evaluate_full_dataset(
        model,
        ds,
        device,
    )


    print()
    print("============================================================")
    print("TEST SET QUANTITATIVE RESULTS")
    print("============================================================")

    print(
        f"Samples                 : {len(ds)}"
    )

    print(
        f"Test loss               : "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Node Chamfer mean       : "
        f"{test_metrics['node_chamfer_mean']:.6f}"
    )

    print(
        f"Node Chamfer max        : "
        f"{test_metrics['node_chamfer_max']:.6f}"
    )

    print(
        f"Node Chamfer min        : "
        f"{test_metrics['node_chamfer_min']:.6f}"
    )

    print(
        f"Node Chamfer std        : "
        f"{test_metrics['node_chamfer_std']:.6f}"
    )

    print(
        f"Consensus gap           : "
        f"{test_metrics['consensus_gap']:.6f}"
    )

    print(
        f"Consensus mean          : "
        f"{test_metrics['consensus_mean']:.6f}"
    )

    print(
        f"Consensus std           : "
        f"{test_metrics['consensus_std']:.6f}"
    )

    print(
        f"Average steps           : "
        f"{test_metrics['steps']:.2f}"
    )

    print(
        f"Convergence rate        : "
        f"{test_metrics['converged']:.2%}"
    )

    print("============================================================")
    print()

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

        with torch.no_grad():
            pred_nodes, final_h, info = model(x, edge_index)

        metrics = compute_sample_metrics(
            pred_nodes,
            final_h,
            sample["target"].to(device),
            info,
        )

        shape_type = get_shape_type(sample)
        shape_dir = save_dir / str(shape_type)
        shape_dir.mkdir(parents=True, exist_ok=True)

        save_path = shape_dir / f"robot_gnn_recon_{shape_type}_{i:04d}.png"

        plot_one_sample(
            sample=sample,
            pred_nodes=pred_nodes,
            info=info,
            metrics=metrics,
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