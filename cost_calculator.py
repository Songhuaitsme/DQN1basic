import pandas as pd
from typing import List, Dict
from data_loader import DataLoader
from topology_builder import TopologyBuilder


class CostCalculator:
    def __init__(self):
        self.price_df = DataLoader.load_data_center_prices()
        self.task_demand = DataLoader.load_task_demand()
        self.topology = TopologyBuilder()

    def calculate_resource_cost(self, data_center_id: int) -> float:
        """计算单个数据中心的资源使用成本（文献1.1.1节公式）"""
        if data_center_id == 0:
            # 节点0是调度服务器，没有资源成本
            return 0.0

        prices = self.price_df.loc[data_center_id]
        # 资源成本 = (CPU成本 + 内存成本 + 磁盘成本 + 带宽成本) × 运行时间
        cpu_cost = self.task_demand["cpu_cores"] * prices["cpu_price"]
        ram_cost = self.task_demand["ram_gb"] * prices["ram_price"]
        disk_cost = self.task_demand["disk_gb"] * prices["disk_price"]
        bandwidth_cost = self.task_demand["bandwidth_mb"] * prices["bandwidth_price"]
        total = (cpu_cost + ram_cost + disk_cost + bandwidth_cost) * self.task_demand["run_time_h"]
        return round(total, 2)

    def calculate_communication_cost(self, path: List[int]) -> float:
        """计算指定路径的网络通信成本（文献1.1.2节公式）"""
        total_comm_cost = 0.0
        task_bandwidth = self.task_demand["bandwidth_mb"]
        run_time = self.task_demand["run_time_h"]

        # 遍历路径中的每条边，累加通信成本
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_weight = self.topology.get_edge_weight(u, v)

            # 文献式(2) pc(s,vi) = Σ |e'i,j| * gw_j * t [cite: 91]
            # gw_j 指的是带宽成本。我们假设带宽价格在边上是对称的。
            # 我们需要取非调度服务器（0）的那个节点的价格。
            dc_id = v if v != 0 else u  # 排除调度服务器（0）
            if dc_id not in self.price_df.index:
                # 如果u,v都是0，这种情况不应该发生，但作为保护
                continue

            bandwidth_price = self.price_df.loc[dc_id, "bandwidth_price"]

            # 文献式(2)的 |e'i,j| * gw_j * t 部分
            # 注意：原cost_calculator.py的实现中，带宽价格(gw_j)取自终点v，
            # 并且错误地乘以了带宽需求(g)。
            # 正确的式(2) [cite: 91] 是 Σ |e'i,j| * g * w_j * t
            # （g=带宽需求, w_j=带宽单价）。
            # 您的cost_calculator.py中实现了：
            # edge_comm_cost = edge_weight * task_bandwidth * bandwidth_price * run_time
            # 这与文献(2) [cite: 91] 中 Σ |e'i,j| * g * w_j * t (g=task_bandwidth, w_j=bandwidth_price) 的单边成本一致。
            # 我们保留这个实现。

            edge_comm_cost = edge_weight * task_bandwidth * bandwidth_price * run_time
            total_comm_cost += edge_comm_cost

        return round(total_comm_cost, 2)

    def calculate_total_cost(self, data_center_id: int, path: List[int]) -> float:
        """计算单个数据中心的调度总成本（资源成本+通信成本，文献1.1.3节）[cite: 96]"""
        res_cost = self.calculate_resource_cost(data_center_id)
        comm_cost = self.calculate_communication_cost(path)
        return round(res_cost + comm_cost, 2)