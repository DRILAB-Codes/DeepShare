import math

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.skip = (
            nn.Identity()
            if in_dim == out_dim
            else nn.Linear(in_dim, out_dim)
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.net(x) + self.skip(x))


def _index_points(points, idx):
    """
    points: (B, N, C)
    idx:    (B, N, K)

    return:
        grouped points: (B, N, K, C)
    """
    B = points.shape[0]

    batch_idx = torch.arange(
        B,
        device=points.device,
    ).view(B, 1, 1)

    return points[batch_idx, idx]


def _knn_indices(points, k):
    """
    Pure-PyTorch kNN.

    points: (B, N, D)

    return:
        idx: (B, N, K)
    """
    N = points.shape[1]
    k = min(k, N)

    dist = torch.cdist(points, points)

    return dist.topk(
        k=k,
        dim=-1,
        largest=False,
    ).indices


class LocalSkipTransformer(nn.Module):
    """
    Lightweight pure-PyTorch version of the local Skip-Transformer idea.

    Current point features are used as query.
    The previous SPD displacement features are used as key when available.

    Attention is restricted to k-nearest spatial neighbors.
    """

    def __init__(
        self,
        feat_dim=128,
        attn_dim=64,
        k=16,
        pos_hidden_dim=64,
    ):
        super().__init__()

        self.k = k

        self.value_fuse = ResidualMLP(
            feat_dim * 2,
            feat_dim,
            feat_dim,
        )

        self.to_q = nn.Linear(feat_dim, attn_dim)
        self.to_k = nn.Linear(feat_dim, attn_dim)
        self.to_v = nn.Linear(feat_dim, attn_dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_hidden_dim),
            nn.GELU(),
            nn.Linear(pos_hidden_dim, attn_dim),
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.GELU(),
            nn.Linear(attn_dim * 2, attn_dim),
        )

        self.out = nn.Linear(attn_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(
        self,
        pos,
        key_feat,
        query_feat,
    ):
        """
        pos:
            (B, N, 3)

        key_feat:
            previous SPD splitting/displacement features
            (B, N, feat_dim)

        query_feat:
            current point features
            (B, N, feat_dim)
        """

        value_feat = self.value_fuse(
            torch.cat(
                [key_feat, query_feat],
                dim=-1,
            )
        )

        identity = value_feat

        q = self.to_q(query_feat)
        k = self.to_k(key_feat)
        v = self.to_v(value_feat)

        idx = _knn_indices(
            pos,
            self.k,
        )

        k_nb = _index_points(k, idx)
        v_nb = _index_points(v, idx)
        pos_nb = _index_points(pos, idx)

        q_rel = q.unsqueeze(2) - k_nb
        pos_rel = pos.unsqueeze(2) - pos_nb

        pos_emb = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(
            q_rel + pos_emb
        )

        # channel-wise local attention over K neighbors
        attention = torch.softmax(
            attention,
            dim=2,
        )

        agg = (
            attention
            * (v_nb + pos_emb)
        ).sum(dim=2)

        return self.norm(
            self.out(agg) + identity
        )


class SeedGenerator(nn.Module):
    """
    Generate a sparse coarse point cloud from latent z.

    z:
        (B, latent_dim)

    output:
        seed points: (B, num_seed, out_dim)
    """

    def __init__(
        self,
        latent_dim=128,
        num_seed=32,
        feat_dim=128,
        out_dim=3,
    ):
        super().__init__()

        self.num_seed = num_seed

        self.seed_tokens = nn.Parameter(
            torch.randn(
                1,
                num_seed,
                feat_dim,
            )
            * 0.02
        )

        self.latent_proj = nn.Sequential(
            nn.Linear(
                latent_dim,
                feat_dim,
            ),
            nn.GELU(),
            nn.Linear(
                feat_dim,
                feat_dim,
            ),
        )

        self.seed_mlp = ResidualMLP(
            feat_dim * 2,
            feat_dim * 2,
            feat_dim,
        )

        self.coord_head = nn.Sequential(
            nn.Linear(
                feat_dim,
                feat_dim,
            ),
            nn.GELU(),
            nn.Linear(
                feat_dim,
                out_dim,
            ),
        )

    def forward(self, z):
        B = z.shape[0]

        tokens = self.seed_tokens.expand(
            B,
            -1,
            -1,
        )

        global_feat = self.latent_proj(
            z
        )

        global_feat = global_feat.unsqueeze(
            1
        ).expand(
            -1,
            self.num_seed,
            -1,
        )

        seed_feat = self.seed_mlp(
            torch.cat(
                [tokens, global_feat],
                dim=-1,
            )
        )

        seed_points = self.coord_head(
            seed_feat
        )

        return seed_points


class SnowflakePointDeconv(nn.Module):
    """
    Snowflake-inspired point deconvolution.

    Each parent point is split into `up_factor` child points.
    Child coordinates are:

        child = parent + delta

    where delta is conditioned on:
      - current point geometry
      - global latent vector
      - local Skip-Transformer context
      - child branch embedding
      - previous SPD displacement feature
    """

    def __init__(
        self,
        latent_dim=128,
        feat_dim=128,
        out_dim=3,
        up_factor=2,
        stage=0,
        k=16,
        attn_dim=64,
        radius=2.0,
        offset_scale=1.0,
        bounded=True,
    ):
        super().__init__()

        self.up_factor = up_factor
        self.stage = stage

        self.radius = radius
        self.offset_scale = offset_scale
        self.bounded = bounded

        self.out_dim = out_dim

        self.point_mlp = nn.Sequential(
            nn.Linear(
                out_dim,
                64,
            ),
            nn.GELU(),
            nn.Linear(
                64,
                feat_dim,
            ),
            nn.GELU(),
        )

        self.query_mlp = ResidualMLP(
            feat_dim * 2 + latent_dim,
            feat_dim * 2,
            feat_dim,
        )

        self.skip_transformer = LocalSkipTransformer(
            feat_dim=feat_dim,
            attn_dim=attn_dim,
            k=k,
        )

        branch_dim = min(
            32,
            feat_dim,
        )

        self.branch_embed = nn.Parameter(
            torch.randn(
                up_factor,
                branch_dim,
            )
            * 0.02
        )

        self.split_mlp = ResidualMLP(
            feat_dim
            + latent_dim
            + branch_dim,
            feat_dim * 2,
            feat_dim,
        )

        self.delta_feature = ResidualMLP(
            feat_dim * 2,
            feat_dim,
            feat_dim,
        )

        self.delta_head = nn.Sequential(
            nn.Linear(
                feat_dim,
                feat_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                feat_dim // 2,
                out_dim,
            ),
        )

    def forward(
        self,
        points,
        z,
        prev_split_feat=None,
    ):
        """
        points:
            (B, N, out_dim)

        z:
            (B, latent_dim)

        prev_split_feat:
            previous SPD feature
            (B, N, feat_dim)
            or None

        returns:
            child_points:
                (B, N * up_factor, out_dim)

            child_feat:
                (B, N * up_factor, feat_dim)
        """

        B, N, _ = points.shape

        local_feat = self.point_mlp(
            points
        )

        global_local = local_feat.max(
            dim=1,
            keepdim=True,
        ).values

        global_local = global_local.expand(
            -1,
            N,
            -1,
        )

        z_expand = z.unsqueeze(
            1
        ).expand(
            -1,
            N,
            -1,
        )

        query_feat = self.query_mlp(
            torch.cat(
                [
                    local_feat,
                    global_local,
                    z_expand,
                ],
                dim=-1,
            )
        )

        if prev_split_feat is None:
            key_feat = query_feat
        else:
            key_feat = prev_split_feat

        context = self.skip_transformer(
            points,
            key_feat,
            query_feat,
        )

        U = self.up_factor

        context_up = context.unsqueeze(
            2
        ).expand(
            -1,
            -1,
            U,
            -1,
        )

        z_up = z[
            :,
            None,
            None,
            :
        ].expand(
            -1,
            N,
            U,
            -1,
        )

        branch = self.branch_embed[
            None,
            None,
            :,
            :
        ].expand(
            B,
            N,
            -1,
            -1,
        )

        split_feat = self.split_mlp(
            torch.cat(
                [
                    context_up,
                    z_up,
                    branch,
                ],
                dim=-1,
            )
        )

        child_feat = self.delta_feature(
            torch.cat(
                [
                    split_feat,
                    context_up,
                ],
                dim=-1,
            )
        )

        delta = self.delta_head(
            child_feat
        )

        if self.bounded:
            scale = (
                self.offset_scale
                / (
                    self.radius
                    ** self.stage
                )
            )

            delta = (
                torch.tanh(delta)
                * scale
            )

        child_points = (
            points.unsqueeze(2).expand(
                -1,
                -1,
                U,
                -1,
            )
            + delta
        )

        child_points = child_points.reshape(
            B,
            N * U,
            self.out_dim,
        )

        child_feat = child_feat.reshape(
            B,
            N * U,
            -1,
        )

        return (
            child_points,
            child_feat,
        )


class SnowflakeDecoder(nn.Module):
    """
    Latent-to-point-cloud Snowflake-inspired decoder.

    Default:
        32 seeds
          -> SPD x2 = 64
          -> SPD x2 = 128

    Input:
        z: (B, latent_dim)

    Output:
        points: (B, num_points, out_dim)

    This keeps the same external interface as FoldingDecoder.
    """

    def __init__(
        self,
        latent_dim=128,
        num_points=128,
        out_dim=3,
        hidden_dim=128,
        num_seed=32,
        up_factors=(2, 2),
        k=16,
        attn_dim=64,
        radius=2.0,
        offset_scale=1.0,
        bounded=True,
    ):
        super().__init__()

        expected_points = (
            num_seed
            * math.prod(up_factors)
        )

        if expected_points != num_points:
            raise ValueError(
                "num_seed * prod(up_factors) "
                "must equal num_points: "
                f"{num_seed} * "
                f"{math.prod(up_factors)} "
                f"= {expected_points}, "
                f"num_points={num_points}"
            )

        self.latent_dim = latent_dim
        self.num_points = num_points
        self.out_dim = out_dim

        self.seed_generator = SeedGenerator(
            latent_dim=latent_dim,
            num_seed=num_seed,
            feat_dim=hidden_dim,
            out_dim=out_dim,
        )

        self.spd_layers = nn.ModuleList(
            [
                SnowflakePointDeconv(
                    latent_dim=latent_dim,
                    feat_dim=hidden_dim,
                    out_dim=out_dim,
                    up_factor=factor,
                    stage=i,
                    k=k,
                    attn_dim=attn_dim,
                    radius=radius,
                    offset_scale=offset_scale,
                    bounded=bounded,
                )
                for i, factor in enumerate(
                    up_factors
                )
            ]
        )

    def forward(
        self,
        z,
        return_stages=False,
    ):
        """
        z:
            (B, latent_dim)

        return_stages=False:
            (B, num_points, out_dim)

        return_stages=True:
            [
                seed points,
                SPD stage 1,
                SPD stage 2,
                ...
            ]
        """

        points = self.seed_generator(
            z
        )

        stages = [points]

        prev_split_feat = None

        for spd in self.spd_layers:
            (
                points,
                prev_split_feat,
            ) = spd(
                points,
                z,
                prev_split_feat,
            )

            stages.append(points)

        if return_stages:
            return stages

        return points


if __name__ == "__main__":
    decoder = SnowflakeDecoder(
        latent_dim=128,
        num_points=128,
        out_dim=3,
        hidden_dim=128,
        num_seed=32,
        up_factors=(2, 2),
    )

    z = torch.randn(
        4,
        128,
    )

    points = decoder(z)

    print(
        "final:",
        points.shape,
    )

    stages = decoder(
        z,
        return_stages=True,
    )

    for i, p in enumerate(stages):
        print(
            f"stage {i}:",
            p.shape,
        )