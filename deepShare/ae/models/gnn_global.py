import torch
import torch.nn as nn

from models.gnn_layers import build_gnn_layer, consensus_gap
from models.pointnet2_encoder import PointNet2Encoder


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
        node_encoder_type="mlp",
        node_encoder_mode="ssg",
        base_radius=1.0,
        npoint1=32,
        npoint2=16,
        include_self=True,
        max_steps=10,
        stable_tol=0.03,
        stable_patience=3,
        stop_when_converged=True,
        use_robot_xy=True,
        robot_xy_dim=2,
        pose_hidden_dim=32,
    ):
        super().__init__()

        if node_encoder_type == "mlp":
            self.node_encoder = RobotPointEncoder(
                point_dim=input_point_dim,
                hidden_dim=node_hidden_dim,
                latent_dim=latent_dim,
            )

        elif node_encoder_type in {"pointnet2", "ssg", "msg"}:
            mode = node_encoder_mode
            if node_encoder_type in {"ssg", "msg"}:
                mode = node_encoder_type

            self.node_encoder = PointNet2Encoder(
                mode=mode,
                input_channels=max(input_point_dim - 3, 0),
                latent_dim=latent_dim,
                base_radius=base_radius,
                npoint1=npoint1,
                npoint2=npoint2,
            )

        else:
            raise ValueError(f"Unknown node_encoder_type: {node_encoder_type}")

        self.use_robot_xy = bool(use_robot_xy)
        self.robot_xy_dim = int(robot_xy_dim)
        self.pose_hidden_dim = int(pose_hidden_dim)

        if self.use_robot_xy:
            self.pose_encoder = nn.Sequential(
                nn.Linear(self.robot_xy_dim, self.pose_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.pose_hidden_dim, self.pose_hidden_dim),
                nn.ReLU(),
            )

            self.node_fusion = nn.Sequential(
                nn.Linear(
                    latent_dim + self.pose_hidden_dim,
                    latent_dim,
                ),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            )
        else:
            self.pose_encoder = None
            self.node_fusion = None

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

    def forward(self, x, edge_index, robot_xy=None):
        """
        x          : [N, P, 3]
        edge_index : [2, E]
        robot_xy   : [N, 2]

        returns:
          pred_nodes : [N, Q, 3]
          final_h    : [N, latent_dim]
          info       : dict
        """

        gap_history = []
        stable_count = 0
        aux_history = []

        # 1. 각 로봇 partial point cloud -> 초기 node embedding
        point_h = self.node_encoder(x)  # [N, latent_dim]

        # 2. robot position embedding + fusion
        if self.use_robot_xy:
            if robot_xy is None:
                raise ValueError(
                    "robot_xy is required when use_robot_xy=True"
                )

            if robot_xy.dim() != 2:
                raise ValueError(
                    f"robot_xy must be [N, 2], got {robot_xy.shape}"
                )

            if robot_xy.size(0) != point_h.size(0):
                raise ValueError(
                    "robot count mismatch: "
                    f"point_h={point_h.size(0)}, "
                    f"robot_xy={robot_xy.size(0)}"
                )

            if robot_xy.size(-1) != self.robot_xy_dim:
                raise ValueError(
                    f"robot_xy last dim must be {self.robot_xy_dim}, "
                    f"got {robot_xy.size(-1)}"
                )

            pose_h = self.pose_encoder(robot_xy)
            h = torch.cat([point_h, pose_h], dim=-1)
            h = self.node_fusion(h)

        else:
            h = point_h

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
            stop_when_converged=model_cfg.get(
                "stop_when_converged",
                True,
            ),
            node_encoder_type=model_cfg.get(
                "node_encoder_type",
                "mlp",
            ),
            node_encoder_mode=model_cfg.get(
                "node_encoder_mode",
                "ssg",
            ),
            base_radius=model_cfg.get("base_radius", 1.0),
            npoint1=model_cfg.get("npoint1", 32),
            npoint2=model_cfg.get("npoint2", 16),

            # 추가
            use_robot_xy=model_cfg.get("use_robot_xy", True),
            robot_xy_dim=model_cfg.get("robot_xy_dim", 2),
            pose_hidden_dim=model_cfg.get("pose_hidden_dim", 32),
        )

    raise ValueError(f"Unknown robot GNN method: {method}")