import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,)
        """
        half = self.dim // 2
        device = t.device

        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=device).float()
            / max(half - 1, 1)
        )

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class DiffusionDecoder(nn.Module):
    """
    AE 호환 conditional diffusion-style decoder 기본형.

    주의:
      진짜 diffusion training은 target, timestep, noise loss가 필요해서
      별도 학습 루프가 필요하다.

    여기서는 기존 AE 코드와 호환되도록:
      decoder(z) -> points
    를 만족하는 deterministic 기본 forward를 제공한다.

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
        hidden_dim=256,
        num_layers=4,
        time_dim=128,
        default_t=0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_points = num_points
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.default_t = default_t

        self.template = nn.Parameter(
            torch.randn(1, num_points, hidden_dim) * 0.02
        )

        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                ]
            )

        self.net = nn.Sequential(*layers)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z, t=None):
        B = z.shape[0]
        device = z.device

        if t is None:
            t = torch.full(
                (B,),
                fill_value=self.default_t,
                device=device,
                dtype=torch.long,
            )

        h = self.template.expand(B, -1, -1)

        z_cond = self.z_proj(z).unsqueeze(1)
        t_cond = self.time_embed(t).unsqueeze(1)

        h = h + z_cond + t_cond
        h = self.net(h)

        points = self.out_proj(h)
        return points