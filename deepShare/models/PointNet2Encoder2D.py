class PointNet2Encoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1 = SetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=2,
            mlp=[64, 64, 128],
        )
        self.sa2 = SetAbstraction(
            npoint=32,
            radius=0.4,
            nsample=32,
            in_channel=128 + 2,
            mlp=[128, 128, 256],
        )
        self.fc = nn.Linear(256, 128)

    def forward(self, xyz):
        points = None
        xyz, points = self.sa1(xyz, points)   # [B,128,2], [B,128,128]
        xyz, points = self.sa2(xyz, points)   # [B,32,2],  [B,32,256]
        global_feat = points.max(dim=1)[0]    # [B,256]
        z = self.fc(global_feat)              # [B,128]
        return z