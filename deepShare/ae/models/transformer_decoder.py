import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    """
    AE 호환 Transformer decoder.

    입력:
      z: (B, latent_dim)

    출력:
      points: (B, num_points, out_dim)
    """

    def __init__(
        self,
        latent_dim=128,
        num_points=128,
        out_dim=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_points = num_points
        self.out_dim = out_dim
        self.d_model = d_model

        self.z_proj = nn.Linear(latent_dim, d_model)

        self.query = nn.Parameter(
            torch.randn(1, num_points, d_model) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, z):
        B = z.shape[0]

        z_token = self.z_proj(z).unsqueeze(1)          # (B, 1, D)
        queries = self.query.expand(B, -1, -1)         # (B, Q, D)

        h = queries + z_token                          # latent conditioning
        h = self.transformer(h)

        points = self.out_proj(h)                      # (B, Q, out_dim)
        return points