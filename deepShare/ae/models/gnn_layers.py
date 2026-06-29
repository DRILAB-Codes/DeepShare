import torch
import torch.nn as nn


def add_self_loops(edge_index: torch.Tensor, num_nodes: int):
    """
    edge_index: [2, E]
    return    : [2, E + N]
    """
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    return torch.cat([edge_index, self_loops], dim=1)


def consensus_gap(h: torch.Tensor):
    """
    h: [N, D]
    각 노드 임베딩이 평균 임베딩에서 얼마나 떨어져 있는지의 최대값.
    """
    mean_h = h.mean(dim=0, keepdim=True)
    deviations = torch.norm(h - mean_h, dim=-1)
    return deviations.max()


class SimpleGraphMean(nn.Module):
    """
    Mean aggregation layer.

    h_i' = mean_{j in N(i)} W h_j
    """

    def __init__(self, in_dim: int, out_dim: int, include_self: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.include_self = include_self
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        x          : [N, in_dim]
        edge_index : [2, E]

        return:
          out : [N, out_dim]
          aux : dict
        """
        num_nodes = x.size(0)

        if self.include_self:
            edge_index = add_self_loops(edge_index, num_nodes)

        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x)

        out = torch.zeros(
            num_nodes,
            h.size(1),
            device=x.device,
            dtype=x.dtype,
        )
        degree = torch.zeros(
            num_nodes,
            device=x.device,
            dtype=x.dtype,
        )

        out.index_add_(0, dst, h[src])
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))

        degree = degree.clamp(min=1.0).unsqueeze(-1)
        out = out / degree

        return out, {
            "alpha": None,
            "edge_index": edge_index,
        }


class SimpleGraphAttention(nn.Module):
    """
    Simple GAT-style aggregation layer.

    attention score:
      e_ij = LeakyReLU(a_src^T W h_j + a_dst^T W h_i)

    여기서 edge_index[0] = src, edge_index[1] = dst.
    즉 src node의 message가 dst node로 전달됨.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        include_self: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(out_dim))
        self.attn_dst = nn.Parameter(torch.empty(out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.include_self = include_self
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        x          : [N, in_dim]
        edge_index : [2, E]

        return:
          out : [N, out_dim]
          aux : dict
        """
        num_nodes = x.size(0)

        if self.include_self:
            edge_index = add_self_loops(edge_index, num_nodes)

        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x)

        e_src = (h[src] * self.attn_src).sum(dim=-1)
        e_dst = (h[dst] * self.attn_dst).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)

        alpha = torch.zeros_like(e)

        # dst node별 softmax
        for node in range(num_nodes):
            mask = dst == node
            if mask.any():
                alpha[mask] = torch.softmax(e[mask], dim=0)

        messages = alpha.unsqueeze(-1) * h[src]

        out = torch.zeros(
            num_nodes,
            h.size(1),
            device=x.device,
            dtype=x.dtype,
        )
        out.index_add_(0, dst, messages)

        return out, {
            "alpha": alpha,
            "edge_index": edge_index,
        }

class SimpleDualGraphAttention(nn.Module):
    """
    Dual attention aggregation layer.

    route 1: alpha_sim  = softmax(e)
    route 2: alpha_diff = softmax(-e)

    최종 출력:
      out = W_o [msg_sim, msg_diff]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        include_self: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

        self.attn_src = nn.Parameter(torch.empty(out_dim))
        self.attn_dst = nn.Parameter(torch.empty(out_dim))

        self.mix = nn.Linear(out_dim * 2, out_dim, bias=True)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.include_self = include_self
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))
        nn.init.xavier_uniform_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        num_nodes = x.size(0)

        if self.include_self:
            edge_index = add_self_loops(edge_index, num_nodes)

        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x)

        e_src = (h[src] * self.attn_src).sum(dim=-1)
        e_dst = (h[dst] * self.attn_dst).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)

        alpha_sim = torch.zeros_like(e)
        alpha_diff = torch.zeros_like(e)

        for node in range(num_nodes):
            mask = dst == node
            if mask.any():
                alpha_sim[mask] = torch.softmax(e[mask], dim=0)
                alpha_diff[mask] = torch.softmax(-e[mask], dim=0)

        msg_sim = alpha_sim.unsqueeze(-1) * h[src]
        msg_diff = alpha_diff.unsqueeze(-1) * h[src]

        out_sim = torch.zeros(
            num_nodes,
            h.size(1),
            device=x.device,
            dtype=x.dtype,
        )
        out_diff = torch.zeros_like(out_sim)

        out_sim.index_add_(0, dst, msg_sim)
        out_diff.index_add_(0, dst, msg_diff)

        out = self.mix(torch.cat([out_sim, out_diff], dim=-1))

        return out, {
            "alpha": alpha_sim,
            "alpha_sim": alpha_sim,
            "alpha_diff": alpha_diff,
            "edge_index": edge_index,
        }

def build_gnn_layer(
    aggregator: str,
    in_dim: int,
    out_dim: int,
    include_self: bool = True,
):
    """
    aggregator:
      - "mean"
      - "attention"
      - "dual_attention"
    """
    if aggregator == "mean":
        return SimpleGraphMean(
            in_dim=in_dim,
            out_dim=out_dim,
            include_self=include_self,
        )

    if aggregator == "attention":
        return SimpleGraphAttention(
            in_dim=in_dim,
            out_dim=out_dim,
            include_self=include_self,
        )
    
    if aggregator == "dual_attention":
        return SimpleDualGraphAttention(
            in_dim=in_dim,
            out_dim=out_dim,
            include_self=include_self,
        )

    raise ValueError(f"Unknown aggregator: {aggregator}")