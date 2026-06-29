import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from datasets.lidar_pointcloud_dataset import LidarObstacleAEDataset
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

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        drop_last=False,
    )


def load_ae_decoder(cfg, device):
    """
    기존 build_robot_gnn_model 구조가 decoder 인자를 요구할 수 있으므로
    AE decoder만 로드한다. 이 스크립트의 loss는 decoder 출력이 아니라
    GNN final_h와 target_latent 사이에서 계산된다.
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

    decoder = ae.decoder

    if cfg["ae"].get("freeze_decoder", True):
        for p in decoder.parameters():
            p.requires_grad = False
        decoder.eval()

    return decoder, ae_cfg


def squeeze_batch(b, device):
    """
    batch_size=1 전제.

    Dataset item:
      x             : [N, P, 3]
      edge_index    : [2, E]
      target        : [Q, 3]
      target_latent : [D]

    DataLoader(batch_size=1):
      x             : [1, N, P, 3]
      edge_index    : [1, 2, E]
      target        : [1, Q, 3]
      target_latent : [1, D]
    """
    x = b["x"].squeeze(0).to(device)
    edge_index = b["edge_index"].squeeze(0).to(device)
    target = b["target"].squeeze(0).to(device)

    if "target_latent" not in b:
        raise KeyError(
            "Dataset must return b['target_latent']. "
            "Modify LidarObstacleAEDataset to read obstacle['ae_latent']."
        )

    target_latent = b["target_latent"].squeeze(0).to(device)

    return x, edge_index, target, target_latent


def nodewise_latent_loss(final_h, target_latent, loss_type="mse"):
    """
    final_h       : [N, D]
    target_latent : [D]

    모든 로봇 노드의 최종 hidden/latent가 동일한 AE latent로 수렴하도록 학습한다.
    """
    if target_latent.dim() != 1:
        target_latent = target_latent.view(-1)

    if final_h.size(-1) != target_latent.size(-1):
        raise ValueError(
            f"latent dim mismatch: final_h={final_h.size(-1)}, "
            f"target_latent={target_latent.size(-1)}"
        )

    target_nodes = target_latent.unsqueeze(0).expand(final_h.size(0), -1)

    if loss_type == "mse":
        return F.mse_loss(final_h, target_nodes)

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(final_h, target_nodes)

    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(final_h, target_nodes, dim=-1).mean()

    if loss_type == "mse_cosine":
        mse = F.mse_loss(final_h, target_nodes)
        cos = 1.0 - F.cosine_similarity(final_h, target_nodes, dim=-1).mean()
        return mse + 0.1 * cos

    raise ValueError(f"Unknown latent_loss_type: {loss_type}")


def train_one_epoch(model, loader, optimizer, device, cfg):
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
    target_latent_norms = []
    latent_mses = []
    latent_cosines = []

    loss_type = cfg["training"].get("latent_loss_type", "mse")
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    for b in loader:
        x, edge_index, target, target_latent = squeeze_batch(b, device)

        pred_nodes, final_h, info = model(x, edge_index)
        loss = nodewise_latent_loss(final_h, target_latent, loss_type=loss_type)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))
        used_steps.append(float(info["used_steps"]))
        converged.append(float(info["converged"]))

        with torch.no_grad():
            target_nodes = target_latent.unsqueeze(0).expand(final_h.size(0), -1)
            mse = F.mse_loss(final_h, target_nodes)
            cosine = F.cosine_similarity(final_h, target_nodes, dim=-1).mean()

            h_norms.append(float(final_h.norm(dim=1).mean().item()))
            h_maxs.append(float(final_h.abs().max().item()))
            target_latent_norms.append(float(target_latent.norm().item()))
            latent_mses.append(float(mse.item()))
            latent_cosines.append(float(cosine.item()))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "latent_mse": float(np.mean(latent_mses)) if latent_mses else float("nan"),
        "latent_cosine": float(np.mean(latent_cosines)) if latent_cosines else float("nan"),
        "steps": float(np.mean(used_steps)) if used_steps else float("nan"),
        "conv": float(np.mean(converged)) if converged else float("nan"),
        "h_norm": float(np.mean(h_norms)) if h_norms else float("nan"),
        "h_max": float(np.mean(h_maxs)) if h_maxs else float("nan"),
        "target_latent_norm": float(np.mean(target_latent_norms)) if target_latent_norms else float("nan"),
    }


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()

    losses = []
    latent_mses = []
    latent_cosines = []
    used_steps = []
    converged = []

    loss_type = cfg["training"].get("latent_loss_type", "mse")

    for b in loader:
        x, edge_index, target, target_latent = squeeze_batch(b, device)

        pred_nodes, final_h, info = model(x, edge_index)
        loss = nodewise_latent_loss(final_h, target_latent, loss_type=loss_type)

        target_nodes = target_latent.unsqueeze(0).expand(final_h.size(0), -1)
        mse = F.mse_loss(final_h, target_nodes)
        cosine = F.cosine_similarity(final_h, target_nodes, dim=-1).mean()

        losses.append(float(loss.item()))
        latent_mses.append(float(mse.item()))
        latent_cosines.append(float(cosine.item()))
        used_steps.append(float(info["used_steps"]))
        converged.append(float(info["converged"]))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "latent_mse": float(np.mean(latent_mses)) if latent_mses else float("nan"),
        "latent_cosine": float(np.mean(latent_cosines)) if latent_cosines else float("nan"),
        "steps": float(np.mean(used_steps)) if used_steps else float("nan"),
        "conv": float(np.mean(converged)) if converged else float("nan"),
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
            "현재 train_gnn_latent.py는 variable num_robots 때문에 batch_size=1만 지원합니다."
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
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_metrics = evaluate(model, val_loader, device, cfg)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_latent_mse": train_metrics["latent_mse"],
            "train_latent_cosine": train_metrics["latent_cosine"],
            "train_steps": train_metrics["steps"],
            "train_conv": train_metrics["conv"],
            "train_h_norm": train_metrics["h_norm"],
            "train_h_max": train_metrics["h_max"],
            "train_target_latent_norm": train_metrics["target_latent_norm"],
            "val_loss": val_metrics["loss"],
            "val_latent_mse": val_metrics["latent_mse"],
            "val_latent_cosine": val_metrics["latent_cosine"],
            "val_steps": val_metrics["steps"],
            "val_conv": val_metrics["conv"],
        }
        hist.append(row)

        print(
            f"[{epoch:04d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_mse={train_metrics['latent_mse']:.6f} "
            f"train_cos={train_metrics['latent_cosine']:.4f} "
            f"h_norm={train_metrics['h_norm']:.2f} "
            f"h_max={train_metrics['h_max']:.2f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mse={val_metrics['latent_mse']:.6f} "
            f"val_cos={val_metrics['latent_cosine']:.4f} "
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
