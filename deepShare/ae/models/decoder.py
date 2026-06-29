import torch.nn as nn

class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_points=128, out_dim=3, hidden_dim=512):
        super().__init__()
        self.num_points = num_points
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_points * out_dim))
    def forward(self, z):
        return self.net(z).view(z.shape[0], self.num_points, self.out_dim)
