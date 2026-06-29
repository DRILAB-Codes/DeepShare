import torch
import torch.nn as nn


def square_distance(src, dst):
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view([B] + [1] * (idx.dim() - 1)).expand_as(idx)
    return points[batch_indices, idx]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    if N == 0:
        raise ValueError("empty point cloud")
    n_unique = min(int(npoint), N)
    centroids = torch.zeros(B, n_unique, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(n_unique):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    if npoint > N:
        extra = torch.randint(0, n_unique, (B, npoint - n_unique), device=device)
        centroids = torch.cat([centroids, centroids.gather(1, extra)], dim=1)
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :min(nsample, N)]
    if group_idx.shape[-1] < nsample:
        group_idx = torch.cat([group_idx, group_idx[:, :, :1].repeat(1, 1, nsample - group_idx.shape[-1])], dim=-1)
    first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_idx[group_idx == N] = first[group_idx == N]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


class SharedMLP2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers += [nn.Conv2d(channels[i], channels[i + 1], 1, bias=False), nn.BatchNorm2d(channels[i + 1]), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp = SharedMLP2d([in_channel + 3] + mlp)
    def forward(self, xyz, points=None):
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device, dtype=xyz.dtype)
            grouped_xyz_norm = xyz.view(B, 1, N, 3) - new_xyz[:, :, None, :]
            if points is not None:
                new_points = torch.cat([grouped_xyz_norm, points.view(B, 1, N, -1)], dim=-1)
            else:
                new_points = grouped_xyz_norm
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2)[0].permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint, self.radius_list, self.nsample_list = npoint, radius_list, nsample_list
        self.mlps = nn.ModuleList([SharedMLP2d([in_channel + 3] + mlp) for mlp in mlp_list])
    def forward(self, xyz, points=None):
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        outs = []
        for radius, nsample, mlp in zip(self.radius_list, self.nsample_list, self.mlps):
            idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]
            grouped = torch.cat([grouped_xyz_norm, index_points(points, idx)], dim=-1) if points is not None else grouped_xyz_norm
            grouped = grouped.permute(0, 3, 2, 1).contiguous()
            feat = mlp(grouped).max(dim=2)[0]
            outs.append(feat)
        return new_xyz, torch.cat(outs, dim=1).permute(0, 2, 1).contiguous()
