import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.lidar_pointcloud_dataset import LidarObstacleAEDataset
from losses.chamfer import chamfer_distance
from models.autoencoder import PointNet2AutoEncoder
from models.robot_gnn import build_robot_gnn_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(cfg, split, batch_size, workers, seed):
    ds = LidarObstacleAEDataset(
        data_dir=cfg[f"{split}_dir"],
        input_num_points=cfg["input_num_points"],
        target_num_points=cfg["target_num_points"],
        include_miss=cfg.get("include_miss", False),
        normalize=cfg.get("normalize", True),
        seed=seed,
        use_world_frame=cfg.get("use_world_frame", True),
    )

    # robot 수가 sample마다 다를 수 있으므로 처음에는 batch_size=1 권장
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        drop_last=False,
    )


def load_ae_decoder(cfg, device):
    ae_ckpt_path = cfg["ae"]["checkpoint"]
    ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_cfg = ckpt["cfg"]

    ae = PointNet2AutoEncoder(
        encoder_mode=ae_cfg["model"]["encoder_mode"],
        latent_dim=ae_cfg["model"]["latent_dim"],
        target_num_points=ae_cfg["data"]["target_num_points"],
        base_radius=ae_cfg["model"].get("base_radius", 1.0),
        npoint1=ae_cfg["model"].get("npoint1", 32),
        npoint2=ae_cfg["model"].get("npoint2", 16),
    ).to(device)

    ae.load_state_dict(ckpt["model"])
    ae.eval()

    decoder = ae.decoder

    if cfg["ae"].get("freeze_decoder", True):
        for p in decoder.parameters():
            p.requires_grad = False
        decoder.eval()

    return decoder, ae_cfg


def squeeze_batch(b, device):
    """
    batch_size=1 전제.

    Dataset:
      x          : [N, P, 3]
      edge_index : [2, E]
      target     : [Q, 3]

    DataLoader(batch_size=1):
      x          : [1, N, P, 3]
      edge_index : [1, 2, E]
      target     : [1, Q, 3]
    """
    x = b["x"].squeeze(0).to(device)
    edge_index = b["edge_index"].squeeze(0).to(device)
    target = b["target"].squeeze(0).to(device)
    return x, edge_index, target


def nodewise_chamfer_loss(pred_nodes, target):
    """
    pred_nodes : [N, Q, 3]
    target     : [Q, 3]

    각 노드의 복원 결과가 같은 전체 target을 맞추도록 학습.
    """
    num_nodes = pred_nodes.size(0)
    target_nodes = target.unsqueeze(0).expand(num_nodes, -1, -1)
    return chamfer_distance(pred_nodes, target_nodes)

@torch.no_grad()
def nodewise_chamfer_stats(pred_nodes, target):
    """
    pred_nodes : [N, Qp, 3]
    target     : [Qt, 3]
    return node-wise chamfer statistics
    """
    vals = []
    for i in range(pred_nodes.size(0)):
        cd = chamfer_distance(
            pred_nodes[i:i+1],
            target.unsqueeze(0),
        )
        vals.append(float(cd.item()))

    arr = np.asarray(vals, dtype=np.float32)

    return {
        "node_chamfer": vals,
        "node_chamfer_mean": float(arr.mean()),
        "node_chamfer_max": float(arr.max()),
        "node_chamfer_min": float(arr.min()),
        "node_chamfer_std": float(arr.std()),
    }


@torch.no_grad()
def consensus_stats(final_h):
    """
    final_h : [N, D]
    """
    mean_h = final_h.mean(dim=0, keepdim=True)
    dist = torch.norm(final_h - mean_h, dim=1)

    return {
        "consensus_l2": dist.detach().cpu().tolist(),
        "consensus_gap": float(dist.max().item()),
        "consensus_mean": float(dist.mean().item()),
        "consensus_std": float(dist.std(unbiased=False).item()),
    }

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    if hasattr(model, "decoder"):
        frozen = not any(p.requires_grad for p in model.decoder.parameters())
        if frozen:
            model.decoder.eval()

    losses = []
    used_steps = []
    converged = []

    h_norms = []
    h_maxs = []
    pred_maxs = []
    target_maxs = []

    for b in loader:
        x, edge_index, target = squeeze_batch(b, device)

        pred_nodes, final_h, info = model(x, edge_index)
        loss = nodewise_chamfer_loss(pred_nodes, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        used_steps.append(float(info["used_steps"]))
        converged.append(float(info["converged"]))

        with torch.no_grad():
            h_norms.append(float(final_h.norm(dim=1).mean().item()))
            h_maxs.append(float(final_h.abs().max().item()))
            pred_maxs.append(float(pred_nodes.abs().max().item()))
            target_maxs.append(float(target.abs().max().item()))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "steps": float(np.mean(used_steps)) if used_steps else float("nan"),
        "conv": float(np.mean(converged)) if converged else float("nan"),
        "h_norm": float(np.mean(h_norms)) if h_norms else float("nan"),
        "h_max": float(np.mean(h_maxs)) if h_maxs else float("nan"),
        "pred_max": float(np.mean(pred_maxs)) if pred_maxs else float("nan"),
        "target_max": float(np.mean(target_maxs)) if target_maxs else float("nan"),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    losses = []
    used_steps = []
    converged = []

    node_chamfer_means = []
    node_chamfer_maxs = []
    node_chamfer_mins = []
    node_chamfer_stds = []

    consensus_gaps = []
    consensus_means = []
    consensus_stds = []

    pred_num_points = []
    target_num_points = []

    for b in loader:
        x, edge_index, target = squeeze_batch(b, device)

        pred_nodes, final_h, info = model(x, edge_index)
        loss = nodewise_chamfer_loss(pred_nodes, target)

        losses.append(float(loss.item()))
        used_steps.append(float(info["used_steps"]))
        converged.append(float(info["converged"]))

        cd_stats = nodewise_chamfer_stats(pred_nodes, target)
        cs_stats = consensus_stats(final_h)

        node_chamfer_means.append(cd_stats["node_chamfer_mean"])
        node_chamfer_maxs.append(cd_stats["node_chamfer_max"])
        node_chamfer_mins.append(cd_stats["node_chamfer_min"])
        node_chamfer_stds.append(cd_stats["node_chamfer_std"])

        consensus_gaps.append(cs_stats["consensus_gap"])
        consensus_means.append(cs_stats["consensus_mean"])
        consensus_stds.append(cs_stats["consensus_std"])

        pred_num_points.append(float(pred_nodes.size(1)))
        target_num_points.append(float(target.size(0)))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "steps": float(np.mean(used_steps)) if used_steps else float("nan"),
        "conv": float(np.mean(converged)) if converged else float("nan"),

        "node_chamfer_mean": float(np.mean(node_chamfer_means)) if node_chamfer_means else float("nan"),
        "node_chamfer_max": float(np.mean(node_chamfer_maxs)) if node_chamfer_maxs else float("nan"),
        "node_chamfer_min": float(np.mean(node_chamfer_mins)) if node_chamfer_mins else float("nan"),
        "node_chamfer_std": float(np.mean(node_chamfer_stds)) if node_chamfer_stds else float("nan"),

        "consensus_gap": float(np.mean(consensus_gaps)) if consensus_gaps else float("nan"),
        "consensus_mean": float(np.mean(consensus_means)) if consensus_means else float("nan"),
        "consensus_std": float(np.mean(consensus_stds)) if consensus_stds else float("nan"),

        "pred_num_points": float(np.mean(pred_num_points)) if pred_num_points else float("nan"),
        "target_num_points": float(np.mean(target_num_points)) if target_num_points else float("nan"),
    }


def save_checkpoint(path, model, optimizer, cfg, epoch, val_metrics):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"].get("seed", 0))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    batch_size = int(cfg["training"].get("batch_size", 1))
    if batch_size != 1:
        raise ValueError(
            "현재 train_robot_gnn.py는 variable num_robots 때문에 batch_size=1만 지원합니다."
        )

    train_loader = make_loader(
        cfg["data"],
        "train",
        batch_size=batch_size,
        workers=cfg["training"].get("num_workers", 0),
        seed=seed,
    )

    val_loader = make_loader(
        cfg["data"],
        "val",
        batch_size=batch_size,
        workers=cfg["training"].get("num_workers", 0),
        seed=seed + 1234,
    )

    decoder, ae_cfg = load_ae_decoder(cfg, device)

    # GNN latent_dim과 AE latent_dim 일치 확인
    ae_latent_dim = int(ae_cfg["model"]["latent_dim"])
    gnn_latent_dim = int(cfg["model"]["latent_dim"])

    if ae_latent_dim != gnn_latent_dim:
        raise ValueError(
            f"latent_dim mismatch: AE latent_dim={ae_latent_dim}, "
            f"GNN latent_dim={gnn_latent_dim}"
        )

    model = build_robot_gnn_model(cfg, decoder=decoder).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    best = float("inf")
    hist = []
    patience = int(cfg["training"].get("patience", 50))
    min_delta = float(cfg["training"].get("min_delta", 1e-5))
    bad_epochs = 0

    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"].get("save_every", 10))

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,

            "train_loss": train_metrics["loss"],
            "train_steps": train_metrics["steps"],
            "train_conv": train_metrics["conv"],
            "train_h_norm": train_metrics["h_norm"],
            "train_h_max": train_metrics["h_max"],
            "train_pred_max": train_metrics["pred_max"],
            "train_target_max": train_metrics["target_max"],

            "val_loss": val_metrics["loss"],
            "val_steps": val_metrics["steps"],
            "val_conv": val_metrics["conv"],

            "val_node_chamfer_mean": val_metrics["node_chamfer_mean"],
            "val_node_chamfer_max": val_metrics["node_chamfer_max"],
            "val_node_chamfer_min": val_metrics["node_chamfer_min"],
            "val_node_chamfer_std": val_metrics["node_chamfer_std"],

            "val_consensus_gap": val_metrics["consensus_gap"],
            "val_consensus_mean": val_metrics["consensus_mean"],
            "val_consensus_std": val_metrics["consensus_std"],

            "val_pred_num_points": val_metrics["pred_num_points"],
            "val_target_num_points": val_metrics["target_num_points"],
        }
        hist.append(row)

        print(
            f"[{epoch:04d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"h_max={train_metrics['h_max']:.2f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"node_cd_mean={val_metrics['node_chamfer_mean']:.6f} "
            f"node_cd_max={val_metrics['node_chamfer_max']:.6f} "
            f"gap={val_metrics['consensus_gap']:.4f} "
            f"pts={val_metrics['pred_num_points']:.0f}/{val_metrics['target_num_points']:.0f} "
            f"val_steps={val_metrics['steps']:.2f} "
            f"val_conv={val_metrics['conv']:.2%}"
        )

        if val_metrics["loss"] < best - min_delta:
            best = val_metrics["loss"]
            bad_epochs = 0

            save_checkpoint(
                out_dir / "model_best.pt",
                model,
                optimizer,
                cfg,
                epoch,
                val_metrics,
            )
        else:
            bad_epochs += 1

        if epoch % save_every == 0:
            save_checkpoint(
                out_dir / f"model_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                cfg,
                epoch,
                val_metrics,
            )

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)

        if bad_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val loss: {best:.6f}"
            )
            break

    print(f"Best val loss: {best:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/robot_gnn_config.yaml")
    args = p.parse_args()
    main(args.config)