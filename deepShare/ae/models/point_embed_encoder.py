# models/point_embed_encoder.py

import torch
import torch.nn as nn

from .point_aggregate import (
    knn_point,
    index_points,
    PointAggregateLayer,
    AttentionAggregateLayer,
)


def local_covariance_features(xyz, k=32):
    """
    xyz: (B, N, 3)
    return cov_flat: (B, N, 9)

    k를 크게 잡아 2-hop neighborhood를 근사한다.
    """
    idx = knn_point(k, xyz, xyz)          # (B, N, k)
    neigh = index_points(xyz, idx)        # (B, N, k, 3)

    rel = neigh - xyz[:, :, None, :]      # (B, N, k, 3)

    cov = torch.einsum(
        "bnki,bnkj->bnij",
        rel,
        rel,
    ) / max(k, 1)                         # (B, N, 3, 3)

    return cov.reshape(xyz.shape[0], xyz.shape[1], 9)


class PointCovEmbedding(nn.Module):
    """
    각 포인트에 대해:
      좌표 embedding + local covariance embedding
    을 만든다.
    """

    def __init__(
        self,
        coord_dim=3,
        cov_dim=9,
        hidden_dim=128,
        out_dim=128,
    ):
        super().__init__()

        self.coord_embed = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        self.cov_embed = nn.Sequential(
            nn.Linear(cov_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, cov):
        coord_feat = self.coord_embed(xyz)
        cov_feat = self.cov_embed(cov)

        feat = torch.cat([coord_feat, cov_feat], dim=-1)
        feat = self.fuse(feat)

        return feat


class PointEmbedEncoder(nn.Module):
    """
    네가 구상한 encoder 구조:

    input point cloud
      ↓
    좌표 + 2-hop 근사 covariance embedding
      ↓
    3~4단계 point aggregation / attention aggregation
      ↓
    global max + mean pooling
      ↓
    MLP projection
      ↓
    latent vector
    """

    def __init__(
        self,
        input_dim=3,
        latent_dim=128,
        hidden_dim=128,
        k_cov=32,
        k_agg=16,
        npoints=(64, 32, 16),
        use_attention=True,
        num_heads=4,
        projection_hidden=512,
    ):
        super().__init__()

        if input_dim < 3:
            raise ValueError("PointEmbedEncoder expects at least 3D input: (x, y, z).")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.k_cov = k_cov

        self.embed = PointCovEmbedding(
            coord_dim=3,
            cov_dim=9,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )

        self.extra_embed = (
            nn.Sequential(
                nn.Linear(input_dim - 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if input_dim > 3
            else None
        )

        layers = []
        in_dim = hidden_dim

        for npoint in npoints:
            if use_attention:
                layer = AttentionAggregateLayer(
                    npoint=npoint,
                    k=k_agg,
                    dim=in_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    pool="max",
                )
            else:
                layer = PointAggregateLayer(
                    npoint=npoint,
                    k=k_agg,
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    use_relative_xyz=True,
                    pool="max",
                )

            layers.append(layer)
            in_dim = hidden_dim

        self.layers = nn.ModuleList(layers)

        # max pooling + mean pooling을 같이 써서 정보 보존 강화
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * 2, projection_hidden),
            nn.LayerNorm(projection_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden, projection_hidden),
            nn.LayerNorm(projection_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden, latent_dim),
        )

    def forward(self, points):
        """
        points: (B, N, C)
          C >= 3
          현재는 points[:, :, :3]만 좌표로 사용.
        """

        xyz = points[:, :, :3].contiguous()

        extra = (
            points[:, :, 3:].contiguous()
            if points.shape[-1] > 3
            else None
        )

        # 1. local covariance feature
        cov = local_covariance_features(
            xyz,
            k=min(self.k_cov, xyz.shape[1]),
        )

        # 2. coordinate + covariance embedding
        feat = self.embed(xyz, cov)
        if extra is not None and self.extra_embed is not None:
            extra_feat = self.extra_embed(extra)
            feat = feat + extra_feat

        # 3. hierarchical point compression
        for layer in self.layers:
            xyz, feat = layer(xyz, feat)

        # 4. global pooling
        global_max = feat.max(dim=1)[0]
        global_mean = feat.mean(dim=1)

        global_feat = torch.cat([global_max, global_mean], dim=-1)

        # 5. latent projection
        z = self.project(global_feat)

        return z