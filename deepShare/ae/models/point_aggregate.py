# models/point_aggregate.py

import torch
import torch.nn as nn


def square_distance(src, dst):
    """
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M)
    """
    return torch.cdist(src, dst, p=2) ** 2


def index_points(points, idx):
    """
    points: (B, N, C)
    idx   : (B, S) or (B, S, K)
    return:
      (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]

    batch_idx = torch.arange(B, device=device).view(
        [B] + [1] * (idx.dim() - 1)
    ).expand_as(idx)

    return points[batch_idx, idx]


def farthest_point_sample(xyz, npoint):
    """
    xyz: (B, N, 3)
    return fps_idx: (B, npoint)
    """
    B, N, _ = xyz.shape
    device = xyz.device

    if N == 0:
        raise ValueError("empty point cloud")

    n_unique = min(int(npoint), N)

    centroids = torch.zeros(B, n_unique, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)

    for i in range(n_unique):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, -1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)

        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    if npoint > N:
        extra = torch.randint(0, n_unique, (B, npoint - n_unique), device=device)
        centroids = torch.cat([centroids, centroids.gather(1, extra)], dim=1)

    return centroids


def knn_point(k, xyz, new_xyz):
    """
    xyz    : (B, N, 3)
    new_xyz: (B, S, 3)
    return idx: (B, S, k)
    """
    B, N, _ = xyz.shape
    k = min(k, N)

    dist = square_distance(new_xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False, sorted=False).indices

    return idx


class SharedMLP(nn.Module):
    def __init__(self, channels, use_bn=True):
        super().__init__()

        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))

            if use_bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))

            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)
        self.use_bn = use_bn

    def forward(self, x):
        """
        x: (..., C)
        """
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        if self.use_bn:
            x = self.net(x)
        else:
            x = self.net(x)

        return x.view(*original_shape, -1)


class PointAggregateLayer(nn.Module):
    """
    FPS + kNN grouping + local feature aggregation.

    입력:
      xyz : (B, N, 3)
      feat: (B, N, C)

    출력:
      new_xyz : (B, S, 3)
      new_feat: (B, S, out_dim)
    """

    def __init__(
        self,
        npoint,
        k,
        in_dim,
        out_dim,
        hidden_dim=None,
        use_relative_xyz=True,
        pool="max",
    ):
        super().__init__()

        self.npoint = npoint
        self.k = k
        self.use_relative_xyz = use_relative_xyz
        self.pool = pool

        hidden_dim = hidden_dim or out_dim

        local_in_dim = in_dim
        if use_relative_xyz:
            local_in_dim += 3

        self.local_mlp = SharedMLP(
            [local_in_dim, hidden_dim, out_dim],
            use_bn=True,
        )

    def forward(self, xyz, feat):
        B, N, _ = xyz.shape

        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        group_idx = knn_point(self.k, xyz, new_xyz)

        grouped_xyz = index_points(xyz, group_idx)
        grouped_feat = index_points(feat, group_idx)

        if self.use_relative_xyz:
            rel_xyz = grouped_xyz - new_xyz[:, :, None, :]
            local_input = torch.cat([rel_xyz, grouped_feat], dim=-1)
        else:
            local_input = grouped_feat

        local_feat = self.local_mlp(local_input)

        if self.pool == "max":
            new_feat = local_feat.max(dim=2)[0]
        elif self.pool == "mean":
            new_feat = local_feat.mean(dim=2)
        elif self.pool == "max_mean":
            new_feat = torch.cat(
                [local_feat.max(dim=2)[0], local_feat.mean(dim=2)],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

        return new_xyz, new_feat


class PointSelfAttentionBlock(nn.Module):
    """
    point feature self-attention block.
    point 수는 유지하고 feature만 갱신.
    """

    def __init__(self, dim, num_heads=4, ff_mult=2):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(inplace=True),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, feat):
        """
        feat: (B, N, C)
        """
        h, _ = self.attn(feat, feat, feat, need_weights=False)
        feat = self.norm1(feat + h)

        h = self.ffn(feat)
        feat = self.norm2(feat + h)

        return feat


class AttentionAggregateLayer(nn.Module):
    """
    Attention으로 feature를 먼저 갱신한 뒤,
    FPS + kNN aggregation으로 point 수를 줄이는 layer.
    """

    def __init__(
        self,
        npoint,
        k,
        dim,
        out_dim,
        num_heads=4,
        pool="max",
    ):
        super().__init__()

        self.attn = PointSelfAttentionBlock(dim, num_heads=num_heads)

        self.aggregate = PointAggregateLayer(
            npoint=npoint,
            k=k,
            in_dim=dim,
            out_dim=out_dim,
            hidden_dim=out_dim,
            use_relative_xyz=True,
            pool=pool,
        )

    def forward(self, xyz, feat):
        feat = self.attn(feat)
        new_xyz, new_feat = self.aggregate(xyz, feat)
        return new_xyz, new_feat