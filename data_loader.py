# data_loader.py
import pandas as pd
from typing import Dict

class DataLoader:
    @staticmethod
    def load_network_topology() -> Dict:
        """加载文献图4的网络拓扑：顶点（0=调度服务器，1-11=数据中心）、边权重（通信距离）"""
        return {
            "vertices": list(range(12)),  # 0~11共12个顶点 [cite: 201]
            "edges": [
                (0, 1, 110), (0, 2, 100), (1, 3, 30),
                (1, 6, 20), (2, 3, 50), (2, 9, 10),
                (3, 9, 20), (3, 8, 10), (3, 4, 40),
                (4, 6, 15), (4, 5, 20), (4, 10, 10),
                (5, 9, 10), (5, 11, 20), (6, 7, 20)
            ]  # (起点, 终点, 通信距离)
        }

    @staticmethod
    def load_data_center_prices() -> pd.DataFrame:
        """加载文献表3：数据中心资源单位时间价格"""
        price_data = {
            "data_center_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "cpu_price": [0.1164, 0.1122, 0.1063, 0.1102, 0.1190, 0.1010, 0.1250, 0.1050, 0.1110, 0.1220, 0.1170],
            "ram_price": [1.581, 1.585, 1.565, 1.567, 1.605, 1.705, 1.495, 1.475, 1.464, 1.450, 1.563],
            "disk_price": [0.058, 0.061, 0.069, 0.063, 0.060, 0.072, 0.052, 0.035, 0.051, 0.045, 0.055],
            "bandwidth_price": [0.334, 0.456, 0.385, 0.489, 0.399, 0.367, 0.347, 0.397, 0.343, 0.385, 0.356],
            "electricity_price": [0.10, 0.12, 0.08, 0.15, 0.11, 0.09, 0.13, 0.08, 0.10, 0.14, 0.11]

        }
        # 价格数据基于文献表3 [cite: 251]
        return pd.DataFrame(price_data).set_index("data_center_id")

    @staticmethod
    def load_task_demand() -> Dict:
        """加载文献3.1.3节实验任务需求"""
        return {
            "cpu_cores": 2048,    # 任务所需CPU核数
            "ram_gb": 4,          # 任务所需内存
            "disk_gb": 128,       # 任务所需磁盘
            "bandwidth_mb": 200,  # 任务所需带宽
            "run_time_h": 1       # 运行时间（文献未明确，设为1小时）
        }
