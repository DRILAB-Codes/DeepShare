import numpy as np


def build_comm_graph(robots, comm_range: float = 3.0, self_loop: bool = False):
    """Build an undirected communication graph using distance threshold."""
    src, dst = [], []
    edge_attr = []

    for i, ri in enumerate(robots):
        for j, rj in enumerate(robots):
            if i == j and not self_loop:
                continue
            dist = float(np.linalg.norm(ri.position - rj.position))
            if dist <= comm_range:
                src.append(i)
                dst.append(j)
                edge_attr.append([dist])

    edge_index = np.asarray([src, dst], dtype=np.int64)
    edge_attr = np.asarray(edge_attr, dtype=np.float32)
    return edge_index, edge_attr
