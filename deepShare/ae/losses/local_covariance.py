import torch


def index_points(points, idx):
    """
    points: (B, N, C)
    idx   : (B, N, K)
    return: (B, N, K, C)
    """
    B = points.shape[0]
    batch_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand_as(idx)
    return points[batch_idx, idx]


def local_covariance_descriptor(points, k=16, use_xyz_dim=3):
    """
    points: (B, N, C)
    return: (B, N, D)

    각 point 기준 k개 이웃의 covariance descriptor를 계산.
    기본적으로 앞 3차원 xyz만 사용.
    """
    xyz = points[:, :, :use_xyz_dim].contiguous()

    B, N, C = xyz.shape
    k = min(k, max(N - 1, 1))

    dist = torch.cdist(xyz, xyz, p=2)

    # 자기 자신 제외
    idx = dist.topk(k=k + 1, dim=-1, largest=False).indices[:, :, 1:]

    neigh = index_points(xyz, idx)          # (B, N, K, C)
    rel = neigh - xyz[:, :, None, :]        # (B, N, K, C)

    cov = torch.einsum("bnki,bnkj->bnij", rel, rel) / float(k)

    # 3D 기준 3x3 = 9차원 descriptor
    cov_flat = cov.reshape(B, N, C * C)

    return cov_flat


def local_variance_loss(pred, target, k=16, reduction="mean"):
    """
    pred  : (B, Np, 3)
    target: (B, Nt, 3)

    pred와 target의 local covariance descriptor 분포를 비교한다.
    point 순서가 다를 수 있으므로 descriptor를 정렬한 뒤 비교한다.
    """
    pred_desc = local_covariance_descriptor(pred, k=k)
    target_desc = local_covariance_descriptor(target, k=k)

    # point 순서가 없으므로 descriptor magnitude 기준으로 정렬
    pred_score = torch.norm(pred_desc, dim=-1)
    target_score = torch.norm(target_desc, dim=-1)

    pred_order = pred_score.argsort(dim=1)
    target_order = target_score.argsort(dim=1)

    B = pred.shape[0]

    pred_sorted = pred_desc[
        torch.arange(B, device=pred.device).view(B, 1),
        pred_order,
    ]

    target_sorted = target_desc[
        torch.arange(B, device=target.device).view(B, 1),
        target_order,
    ]

    # N이 다를 경우 작은 쪽에 맞춤
    M = min(pred_sorted.shape[1], target_sorted.shape[1])
    pred_sorted = pred_sorted[:, :M]
    target_sorted = target_sorted[:, :M]

    loss = (pred_sorted - target_sorted).pow(2).mean(dim=(1, 2))

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss