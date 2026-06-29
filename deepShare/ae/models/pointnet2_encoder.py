import torch.nn as nn
from .pointnet2_modules import PointNetSetAbstraction, PointNetSetAbstractionMsg

class PointNet2Encoder(nn.Module):
    def __init__(self, mode="ssg", input_channels=0, latent_dim=128, base_radius=1.0, npoint1=32, npoint2=16):
        super().__init__()
        if mode not in {"ssg", "msg"}:
            raise ValueError("mode must be 'ssg' or 'msg'")
        self.mode = mode
        if mode == "ssg":
            self.sa1 = PointNetSetAbstraction(npoint1, 0.15 * base_radius, 16, input_channels, [64, 64, 128], False)
            self.sa2 = PointNetSetAbstraction(npoint2, 0.30 * base_radius, 32, 128, [128, 128, 256], False)
            self.sa3 = PointNetSetAbstraction(None, None, None, 256, [256, 512, 1024], True)
        else:
            self.sa1 = PointNetSetAbstractionMsg(npoint1, [0.10 * base_radius, 0.20 * base_radius, 0.40 * base_radius], [8, 16, 32], input_channels, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            self.sa2 = PointNetSetAbstractionMsg(npoint2, [0.20 * base_radius, 0.40 * base_radius, 0.80 * base_radius], [16, 32, 32], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
            self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)
        self.project = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, latent_dim))
    def forward(self, points):
        xyz = points[:, :, :3].contiguous()
        extra = points[:, :, 3:].contiguous() if points.shape[-1] > 3 else None
        l1_xyz, l1_points = self.sa1(xyz, extra)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        return self.project(l3_points.squeeze(1))
