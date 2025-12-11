import random
import networkx as nx
from typing import List, Dict, Tuple
from data_loader import DataLoader
from topology_builder import TopologyBuilder
from cost_calculator import CostCalculator


class BaselineAlgorithms:
    def __init__(self):
        self.price_df = DataLoader.load_data_center_prices()
        self.task_demand = DataLoader.load_task_demand()
        self.topology = TopologyBuilder()
        self.cost_calc = CostCalculator()
        self.graph = self.topology.get_graph()
        self.dc_ids = list(range(1, 12))  # 11个数据中心ID

    def crf_algorithm(self, data_center_id: int) -> Dict:
        """CRF算法：计算资源优先，通信路径随机（文献3.1.3节）"""
        # 1. 计算当前数据中心的资源成本（CRF核心：优先资源成本低的中心）
        res_cost = self.cost_calc.calculate_resource_cost(data_center_id)
        # 2. 随机选择调度服务器（0）到当前数据中心的路径
        all_paths = list(nx.all_simple_paths(self.graph, source=0, target=data_center_id))
        if not all_paths:
            return {"data_center_id": data_center_id, "path": [], "resource_cost": res_cost, "total_cost": res_cost}

        random_path = random.choice(all_paths)
        # 3. 计算总成本
        total_cost = self.cost_calc.calculate_total_cost(data_center_id, random_path)
        return {
            "data_center_id": data_center_id,
            "path": random_path,
            "resource_cost": res_cost,
            "total_cost": total_cost
        }

    def spf_algorithm(self, data_center_id: int) -> Dict:
        """SPF算法：最短路径优先，忽略资源成本（文献3.1.3节）"""
        # 1. 找调度服务器（0）到当前数据中心的最短路径（按边权重=通信距离）
        shortest_path = nx.shortest_path(self.graph, source=0, target=data_center_id, weight="weight")
        # 2. 计算总成本
        total_cost = self.cost_calc.calculate_total_cost(data_center_id, shortest_path)
        return {
            "data_center_id": data_center_id,
            "path": shortest_path,
            "total_cost": total_cost
        }

    def run_all_baselines(self) -> Tuple[Dict, Dict]:
        """运行所有数据中心的CRF与SPF，返回结果字典"""
        crf_results = {dc: self.crf_algorithm(dc) for dc in self.dc_ids}
        spf_results = {dc: self.spf_algorithm(dc) for dc in self.dc_ids}
        return crf_results, spf_results