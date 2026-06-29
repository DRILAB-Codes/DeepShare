import torch
import torch.nn as nn

from models.gnn_layers import build_gnn_layer, consensus_gap


class RobotPointEncoder(nn.Module):
    """
    각 로봇의 partial point cloud를 node embedding으로 변환.

    input : x [N, P, 3]
    output: h [N, latent_dim]
    """

    def __init__(self, point_dim=3, hidden_dim=128, latent_dim=64):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: [N, P, 3]
        """
        point_feat = self.point_mlp(x)     # [N, P, latent_dim]
        node_feat = point_feat.max(dim=1).values  # [N, latent_dim]
        return node_feat


class ConsensusRobotGNN(nn.Module):
    """
    Consensus 기반 분산 로봇 GNN.

    pipeline:
      robot partial point cloud
      -> RobotPointEncoder 단순 MLP 임베딩
      -> iterative GNN message passing
      -> node-wise AE decoder
      -> reconstructed obstacle boundary per node
    """

    def __init__(
        self,
        decoder,
        input_point_dim=3,
        node_hidden_dim=128,
        latent_dim=64,
        aggregator="attention",
        include_self=True,
        max_steps=10,
        stable_tol=0.03,
        stable_patience=3,
        stop_when_converged=True,
    ):
        super().__init__()

        self.node_encoder = RobotPointEncoder(
            point_dim=input_point_dim,
            hidden_dim=node_hidden_dim,
            latent_dim=latent_dim,
        )

        self.initial_gnn = build_gnn_layer(
            aggregator=aggregator,
            in_dim=latent_dim,
            out_dim=latent_dim,
            include_self=include_self,
        )

        self.shared_gnn = build_gnn_layer(
            aggregator=aggregator,
            in_dim=latent_dim,
            out_dim=latent_dim,
            include_self=include_self,
        )

        self.decoder = decoder

        self.aggregator = aggregator
        self.include_self = include_self
        self.max_steps = int(max_steps)
        self.stable_tol = float(stable_tol)
        self.stable_patience = int(stable_patience)
        self.stop_when_converged = bool(stop_when_converged)

    def forward(self, x, edge_index):
        """
        x          : [N, P, 3]
        edge_index : [2, E]

        returns:
          pred_nodes : [N, Q, 3]
          final_h    : [N, latent_dim]
          info       : dict
        """

        gap_history = []
        stable_count = 0
        aux_history = []

        # 1. 각 로봇 partial point cloud -> 초기 node embedding
        h = self.node_encoder(x)  # [N, latent_dim]

        # 2. 첫 message passing
        h, aux = self.initial_gnn(h, edge_index)
        aux_history.append(aux)

        gap = consensus_gap(h)
        gap_history.append(float(gap.detach().cpu()))
        used_steps = 1

        if gap.item() < self.stable_tol:
            stable_count = 1

        # 3. 반복 message passing
        while used_steps < self.max_steps:
            h, aux = self.shared_gnn(h, edge_index)
            aux_history.append(aux)

            gap = consensus_gap(h)
            gap_history.append(float(gap.detach().cpu()))
            used_steps += 1

            if gap.item() < self.stable_tol:
                stable_count += 1
            else:
                stable_count = 0

            if self.stop_when_converged and stable_count >= self.stable_patience:
                break

        # 4. 각 노드 embedding으로 AE decoder 수행
        pred_nodes = self.decoder(h)  # [N, Q, 3]

        info = {
            "used_steps": used_steps,
            "stable_count": stable_count,
            "converged": stable_count >= self.stable_patience,
            "gap_history": gap_history,
            "aggregator": self.aggregator,
            "include_self": self.include_self,
            "node_embeddings": h,
            "aux_history": aux_history,
        }

        return pred_nodes, h, info


def build_robot_gnn_model(cfg, decoder):
    """
    cfg["model"] 예시:
      method: consensus
      latent_dim: 64
      node_hidden_dim: 128
      aggregator: attention
      include_self: true
      max_steps: 10
      stable_tol: 0.03
      stable_patience: 3
      stop_when_converged: true
    """

    model_cfg = cfg["model"]
    method = model_cfg.get("method", "consensus")

    if method == "consensus":
        return ConsensusRobotGNN(
            decoder=decoder,
            input_point_dim=model_cfg.get("input_point_dim", 3),
            node_hidden_dim=model_cfg.get("node_hidden_dim", 128),
            latent_dim=model_cfg["latent_dim"],
            aggregator=model_cfg.get("aggregator", "attention"),
            include_self=model_cfg.get("include_self", True),
            max_steps=model_cfg.get("max_steps", 10),
            stable_tol=model_cfg.get("stable_tol", 0.03),
            stable_patience=model_cfg.get("stable_patience", 3),
            stop_when_converged=model_cfg.get("stop_when_converged", True),
        )

    raise ValueError(f"Unknown robot GNN method: {method}")