import networkx as nx
from typing import List, Dict
from data_loader import DataLoader


class TopologyBuilder:
    def __init__(self):
        self.topology_data = DataLoader.load_network_topology()
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:

        G = nx.Graph()

        G.add_nodes_from(self.topology_data["vertices"])
        G.add_weighted_edges_from(self.topology_data["edges"])

        return G

    def get_adjacent_nodes(self, node: int) -> List[int]:
        """获取指定顶点的相邻顶点（用于PER-DQN动作空间定义）"""
        return list(self.graph.neighbors(node))

    def get_edge_weight(self, node1: int, node2: int) -> float:
        """获取两个相邻顶点间的边权重（通信距离，用于成本计算）[cite: 80]"""
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]["weight"]
        raise ValueError(f"顶点{node1}与{node2}无直接连接（文献图4中无此边）")

    def get_graph(self) -> nx.Graph:
        """返回完整拓扑图（供其他模块调用）"""
        return self.graph