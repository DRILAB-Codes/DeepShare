import torch

def chamfer_distance(pred, target, reduction="mean"):
    dist = torch.cdist(pred, target, p=2) ** 2
    a = dist.min(dim=2).values.mean(dim=1)
    b = dist.min(dim=1).values.mean(dim=1)
    loss = a + b
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
