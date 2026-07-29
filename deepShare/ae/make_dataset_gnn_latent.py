# add_gnn_latent_to_dataset.py

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from datasets.lidar_pointcloud_dataset import LidarObstacleAEDataset
from models.autoencoder import PointNet2AutoEncoder
from models.robot_gnn import build_robot_gnn_model


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_config_from_checkpoint(ckpt, config_path=None):
    """
    우선순위:
      1) --config로 지정한 YAML
      2) GNN checkpoint 내부의 cfg
    """
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if "cfg" not in ckpt:
        raise KeyError(
            "GNN checkpoint에 'cfg'가 없습니다. "
            "--config configs/gnn_latent_config.yaml 을 함께 지정하세요."
        )

    return ckpt["cfg"]


def build_ae_decoder(cfg, device):
    """
    GNN 모델 생성에 필요한 AE decoder를 checkpoint에서 복원한다.
    """
    ae_ckpt_path = cfg["ae"]["checkpoint"]
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_cfg = ae_ckpt["cfg"]

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

    ae.load_state_dict(ae_ckpt["model"])
    ae.eval()

    for p in ae.parameters():
        p.requires_grad = False

    return ae.decoder, ae_cfg


def build_gnn(gnn_checkpoint_path, config_path, device):
    """
    학습된 GNN과 해당 GNN이 사용하는 frozen AE decoder를 복원한다.
    """
    gnn_ckpt = torch.load(gnn_checkpoint_path, map_location=device)
    cfg = load_config_from_checkpoint(gnn_ckpt, config_path)

    decoder, ae_cfg = build_ae_decoder(cfg, device)

    ae_latent_dim = int(ae_cfg["model"]["latent_dim"])
    gnn_latent_dim = int(cfg["model"]["latent_dim"])

    if ae_latent_dim != gnn_latent_dim:
        raise ValueError(
            f"latent_dim mismatch: AE={ae_latent_dim}, GNN={gnn_latent_dim}"
        )

    model = build_robot_gnn_model(cfg, decoder=decoder).to(device)

    if "model" not in gnn_ckpt:
        raise KeyError(
            f"{gnn_checkpoint_path}: checkpoint 내부에 'model' state_dict가 없습니다."
        )

    model.load_state_dict(gnn_ckpt["model"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, cfg, gnn_ckpt


def make_dataset(data_cfg, data_dir, seed):
    return LidarObstacleAEDataset(
        data_dir=data_dir,
        input_num_points=data_cfg["input_num_points"],
        target_num_points=data_cfg["target_num_points"],
        include_miss=data_cfg.get("include_miss", False),
        normalize=data_cfg.get("normalize", True),
        seed=seed,
        use_world_frame=data_cfg.get("use_world_frame", True),
    )


def resolve_dataset_files(dataset, src_dir):
    """
    Dataset 내부 파일 목록 속성이 있으면 그것을 사용한다.
    없으면 src_dir의 정렬된 JSON 목록을 사용한다.

    LidarObstacleAEDataset이 내부에서 JSON을 정렬된 순서로 읽는다는
    전제와 일치하는지 길이를 검사한다.
    """
    candidate_attrs = (
        "files",
        "file_paths",
        "paths",
        "json_files",
        "samples",
    )

    for attr in candidate_attrs:
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)

            if isinstance(value, (list, tuple)) and len(value) == len(dataset):
                paths = []
                valid = True

                for item in value:
                    if isinstance(item, (str, Path)):
                        paths.append(Path(item))
                    elif isinstance(item, dict):
                        found = None
                        for key in ("path", "file", "file_path", "json_path"):
                            if key in item:
                                found = Path(item[key])
                                break
                        if found is None:
                            valid = False
                            break
                        paths.append(found)
                    else:
                        valid = False
                        break

                if valid:
                    return paths

    files = sorted(Path(src_dir).glob("*.json"))

    if len(files) != len(dataset):
        raise RuntimeError(
            "Dataset 길이와 JSON 파일 수가 다릅니다.\n"
            f"dataset={len(dataset)}, json_files={len(files)}\n"
            "LidarObstacleAEDataset의 파일 목록 속성을 확인해 "
            "resolve_dataset_files()에 추가하세요."
        )

    return files


def squeeze_item(batch, device):
    """
    DataLoader(batch_size=1) 기준.

      x             : [1, N, P, 3]
      edge_index    : [1, 2, E]
      target        : [1, Q, 3]
      target_latent : [1, D]
    """
    x = batch["x"].squeeze(0).to(device)
    edge_index = batch["edge_index"].squeeze(0).to(device)

    target = None
    if "target" in batch:
        target = batch["target"].squeeze(0).to(device)

    target_latent = None
    if "target_latent" in batch:
        target_latent = batch["target_latent"].squeeze(0).to(device)

    return x, edge_index, target, target_latent


def to_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Scalar tensor expected.")
        return float(value.detach().cpu().item())
    return float(value)


def to_bool(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Scalar tensor expected.")
        return bool(value.detach().cpu().item())
    return bool(value)


def make_analysis_record(
    final_h,
    target_latent,
    info,
    checkpoint_path,
    checkpoint_epoch,
):
    """
    final_h: [N, D]

    저장 항목:
      - 각 로봇별 GNN latent
      - 노드 평균 latent
      - 노드 간 consensus 통계
      - AE target latent와의 MSE/cosine
      - checkpoint 및 수렴 정보
    """
    final_h_cpu = final_h.detach().cpu()
    mean_h = final_h_cpu.mean(dim=0)

    centered = final_h_cpu - mean_h.unsqueeze(0)
    consensus_l2 = centered.norm(dim=1)

    record = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "num_robots": int(final_h_cpu.size(0)),
        "latent_dim": int(final_h_cpu.size(1)),
        "used_steps": to_float(info.get("used_steps", float("nan"))),
        "converged": to_bool(info.get("converged", False)),
        "gnn_latent_mean": mean_h.tolist(),
        "gnn_latent_nodes": final_h_cpu.tolist(),
        "consensus_l2_per_robot": consensus_l2.tolist(),
        "consensus_gap": float(consensus_l2.max().item()),
        "consensus_mean": float(consensus_l2.mean().item()),
        "gnn_mean_norm": float(mean_h.norm().item()),
        "gnn_node_norm_mean": float(final_h_cpu.norm(dim=1).mean().item()),
        "gnn_node_norm_std": float(
            final_h_cpu.norm(dim=1).std(unbiased=False).item()
        ),
    }

    if target_latent is not None:
        target_cpu = target_latent.detach().cpu().view(-1)

        if target_cpu.numel() != mean_h.numel():
            raise ValueError(
                "target_latent와 GNN latent 차원이 다릅니다: "
                f"target={target_cpu.numel()}, gnn={mean_h.numel()}"
            )

        target_nodes = target_cpu.unsqueeze(0).expand_as(final_h_cpu)

        node_mse = ((final_h_cpu - target_nodes) ** 2).mean(dim=1)
        node_cos = F.cosine_similarity(
            final_h_cpu,
            target_nodes,
            dim=1,
        )

        record.update(
            {
                "ae_target_latent": target_cpu.tolist(),
                "ae_target_norm": float(target_cpu.norm().item()),
                "mean_latent_mse": float(
                    F.mse_loss(mean_h, target_cpu).item()
                ),
                "mean_latent_cosine": float(
                    F.cosine_similarity(
                        mean_h.unsqueeze(0),
                        target_cpu.unsqueeze(0),
                        dim=1,
                    ).item()
                ),
                "node_latent_mse": node_mse.tolist(),
                "node_latent_mse_mean": float(node_mse.mean().item()),
                "node_latent_mse_max": float(node_mse.max().item()),
                "node_latent_cosine": node_cos.tolist(),
                "node_latent_cosine_mean": float(node_cos.mean().item()),
                "node_latent_cosine_min": float(node_cos.min().item()),
            }
        )

    return record


def attach_robot_latents(sample, final_h):
    """
    sample 안에 robots 리스트가 있고 로봇 수가 맞으면,
    각 robot dict에도 gnn_latent를 직접 추가한다.

    원본 구조가 다르거나 수가 맞지 않으면 obstacle.gnn_analysis에만 저장한다.
    """
    robots = sample.get("robots")
    if not isinstance(robots, list):
        return False

    if len(robots) != final_h.size(0):
        return False

    final_h_list = final_h.detach().cpu().tolist()

    for robot, latent in zip(robots, final_h_list):
        if isinstance(robot, dict):
            robot["gnn_latent"] = latent
            robot["gnn_latent_dim"] = len(latent)
        else:
            return False

    return True


@torch.no_grad()
def process_dir(
    src_dir,
    dst_dir,
    model,
    cfg,
    device,
    checkpoint_path,
    checkpoint_epoch,
    overwrite=False,
):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    dataset = make_dataset(
        data_cfg=cfg["data"],
        data_dir=str(src_dir),
        seed=int(cfg["training"].get("seed", 0)),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    files = resolve_dataset_files(dataset, src_dir)

    encoded = 0
    skipped = 0
    robot_attached = 0
    robot_not_attached = 0

    for src_path, batch in tqdm(
        zip(files, loader),
        total=len(files),
        desc=f"{src_dir} -> {dst_dir}",
    ):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel

        sample = load_json(src_path)

        if "obstacle" not in sample:
            raise KeyError(f"{src_path}: missing obstacle field")

        obstacle = sample["obstacle"]

        if "gnn_analysis" in obstacle and not overwrite:
            save_json(sample, dst_path)
            skipped += 1
            continue

        x, edge_index, target, target_latent = squeeze_item(batch, device)

        pred_nodes, final_h, info = model(x, edge_index)

        record = make_analysis_record(
            final_h=final_h,
            target_latent=target_latent,
            info=info,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint_epoch,
        )

        # 분석 결과는 obstacle 아래에 한 묶음으로 저장
        obstacle["gnn_analysis"] = record

        # 가능하면 각 robot dict에도 직접 latent를 추가
        attached = attach_robot_latents(sample, final_h)
        obstacle["gnn_analysis"]["attached_to_robot_entries"] = attached

        if attached:
            robot_attached += 1
        else:
            robot_not_attached += 1

        save_json(sample, dst_path)
        encoded += 1

    print(f"Done: {src_dir}")
    print(f"  encoded: {encoded}")
    print(f"  skipped: {skipped}")
    print(f"  robot entries attached: {robot_attached}")
    print(f"  robot entries not attached: {robot_not_attached}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--gnn_checkpoint", required=True)
    p.add_argument(
        "--config",
        default=None,
        help=(
            "선택 사항. 지정하지 않으면 GNN checkpoint 내부 cfg를 사용합니다. "
            "현재 코드와 checkpoint cfg가 다를 때만 명시하세요."
        ),
    )

    # AE latent가 이미 들어간 데이터셋을 입력으로 사용
    p.add_argument(
        "--src_train",
        default="data/gnn_latent/train_mixed",
    )
    p.add_argument(
        "--src_val",
        default="data/gnn_latent/val_mixed",
    )
    p.add_argument(
        "--src_test",
        default="data/gnn_latent/test_mixed",
    )

    # checkpoint별 분석 결과를 별도 디렉터리에 저장하는 것을 권장
    p.add_argument(
        "--dst_train",
        default="data/gnn_latent_with_gnn/train_mixed",
    )
    p.add_argument(
        "--dst_val",
        default="data/gnn_latent_with_gnn/val_mixed",
    )
    p.add_argument(
        "--dst_test",
        default="data/gnn_latent_with_gnn/test_mixed",
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="처리할 split만 선택할 수 있습니다.",
    )

    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA를 사용할 수 없어 CPU로 전환합니다.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, cfg, gnn_ckpt = build_gnn(
        gnn_checkpoint_path=args.gnn_checkpoint,
        config_path=args.config,
        device=device,
    )

    checkpoint_epoch = gnn_ckpt.get("epoch", None)

    print(f"Loaded GNN checkpoint: {args.gnn_checkpoint}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"GNN latent_dim: {cfg['model']['latent_dim']}")
    print(f"Aggregator: {cfg['model'].get('aggregator', 'unknown')}")
    print(f"Device: {device}")

    split_args = {
        "train": (args.src_train, args.dst_train),
        "val": (args.src_val, args.dst_val),
        "test": (args.src_test, args.dst_test),
    }

    for split in args.splits:
        src_dir, dst_dir = split_args[split]

        process_dir(
            src_dir=src_dir,
            dst_dir=dst_dir,
            model=model,
            cfg=cfg,
            device=device,
            checkpoint_path=args.gnn_checkpoint,
            checkpoint_epoch=checkpoint_epoch,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()