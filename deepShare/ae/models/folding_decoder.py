# models/folding_decoder.py

import torch
import torch.nn as nn


class FoldingDecoder(nn.Module):
    """
    1D FoldingNet-style decoder.

    latent vector z와 1D canonical grid를 concat한 뒤,
    shared MLP로 boundary point cloud를 생성한다.

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
        hidden_dim=512,
        grid_dim=1,
        num_folds=2,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_points = num_points
        self.out_dim = out_dim
        self.grid_dim = grid_dim
        self.num_folds = num_folds

        grid = torch.linspace(-1.0, 1.0, num_points).view(1, num_points, 1)

        if grid_dim == 1:
            self.register_buffer("grid", grid)

        elif grid_dim == 2:
            side = int(num_points ** 0.5)
            if side * side != num_points:
                raise ValueError("For grid_dim=2, num_points must be a perfect square.")

            u = torch.linspace(-1.0, 1.0, side)
            v = torch.linspace(-1.0, 1.0, side)
            uu, vv = torch.meshgrid(u, v, indexing="ij")
            grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
            self.register_buffer("grid", grid.view(1, num_points, 2))

        else:
            raise ValueError("grid_dim must be 1 or 2.")

        self.fold1 = nn.Sequential(
            nn.Linear(latent_dim + grid_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        if num_folds >= 2:
            self.fold2 = nn.Sequential(
                nn.Linear(latent_dim + out_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.fold2 = None

    def forward(self, z):
        """
        z: (B, latent_dim)
        """
        B = z.shape[0]

        grid = self.grid.expand(B, -1, -1)
        z_expand = z[:, None, :].expand(-1, self.num_points, -1)

        fold_input = torch.cat([grid, z_expand], dim=-1)
        points = self.fold1(fold_input)

        if self.fold2 is not None:
            fold_input = torch.cat([points, z_expand], dim=-1)
            points = self.fold2(fold_input)

        return points