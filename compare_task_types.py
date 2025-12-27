import numpy as np
import pandas as pd
import tensorflow as tf
from dqn_agent import DQNAgent

# === 1. 重新设计的极端差异化任务 ===
# 核心策略：
# 1. 通信主导型：高带宽 -> 距离近、带宽便宜的节点优势 (预期: 节点1)
# 2. 计算主导型：超高CPU + 极低带宽 -> 忽略距离，CPU最便宜的节点优势 (预期: 节点6)
# 3. 存储主导型：超高磁盘 + 极低带宽 -> 忽略距离，磁盘最便宜的节点优势 (预期: 节点8)

TEST_CASES = [
    {
        "id": "任务0",
        "name": "原始任务",
        "desc": "",
        "cpu_cores": 2048,
        "ram_gb": 4,
        "disk_gb": 128,  # [关键] 海量磁盘需求，放大磁盘价格差异
        "bandwidth_mb": 200,  # [关键] 极低带宽，消除距离惩罚
        "run_time_h": 1  # 长时间存储
    },
    {
        "id": "任务A",
        "name": "通信密集型 (流媒体转发)",
        "desc": "带宽需求极大，但计算极少。优先选择带宽便宜且距离近的节点。",
        "cpu_cores": 100,  # CPU需求低
        "ram_gb": 1,
        "disk_gb": 10,
        "bandwidth_mb": 2000,  # [关键] 极高带宽，放大通信成本权重
        "run_time_h": 1
    },
    {
        "id": "任务B",
        "name": "算力密集型 (离线渲染)",
        "desc": "CPU消耗极大，数据需预加载(运行中带宽低)。优先选择CPU单价最低的节点。",
        "cpu_cores": 8000,  # [关键] 极高CPU，放大CPU价格差异
        "ram_gb": 16,
        "disk_gb": 100,
        "bandwidth_mb": 1,  # [关键] 极低带宽，消除通信距离带来的惩罚
        "run_time_h": 10  # 长时间运行，进一步放大资源单价的影响
    },
    {
        "id": "任务C",
        "name": "存储密集型 (冷数据归档)",
        "desc": "占用海量磁盘，几乎无网络IO。优先选择磁盘单价最低的节点。",
        "cpu_cores": 100,
        "ram_gb": 4,
        "disk_gb": 20000,  # [关键] 海量磁盘需求，放大磁盘价格差异
        "bandwidth_mb": 1,  # [关键] 极低带宽，消除距离惩罚
        "run_time_h": 24  # 长时间存储
    }
]


def main():
    print("==================================================")
    print("          多类型任务 DQN 调度优化对比 (差异化版)")
    print("==================================================")

    # 2. 初始化智能体
    agent = DQNAgent()

    # 尝试加载模型
    model_path = "dqn_shortest_path_model"
    try:
        agent.main_net = tf.keras.models.load_model(model_path)
        print(f"成功加载模型: {model_path}")
        agent.epsilon = 0.0
    except:
        print("警告: 未找到模型文件，将使用随机策略 (结果可能不准确)")

    summary_list = []

    # 3. 循环执行任务
    for task in TEST_CASES:
        print(f"\n>>> 正在分析: {task['name']}")
        print(f"    特征: {task['desc']}")

        # 注入任务需求
        agent.cost_calc.task_demand = task

        best_node = -1
        min_total_cost = float('inf')
        best_breakdown = {}

        # 遍历所有节点
        for dc in range(1, 12):
            path = agent.get_shortest_path(dc)

            total = agent.cost_calc.calculate_total_cost(dc, path)
            res_cost = agent.cost_calc.calculate_resource_cost(dc)
            comm_cost = agent.cost_calc.calculate_communication_cost(path)

            # 更新最优解
            if total < min_total_cost:
                min_total_cost = total
                best_node = dc
                best_breakdown = {"res": res_cost, "comm": comm_cost}

        # 判断主导成本
        cost_type = "通信主导" if best_breakdown['comm'] > best_breakdown['res'] else "资源主导"

        summary_list.append({
            "任务ID": task['id'],
            "任务类型": task['name'].split(" ")[0],
            "最优节点": best_node,
            "最低总成本": round(min_total_cost, 2),
            "资源成本": round(best_breakdown['res'], 2),
            "通信成本": round(best_breakdown['comm'], 2),
            "成本构成": cost_type
        })

    # 4. 输出最终对比表
    print("\n\n" + "=" * 80)
    print(f"{'最终对比总结':^80}")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_list)

    # 设置显示格式
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.east_asian_width', True)  # 解决中文对齐

    # 调整列顺序
    cols = ["任务ID", "任务类型", "最优节点", "最低总成本", "资源成本", "通信成本", "成本构成"]
    print(summary_df[cols].to_string(index=False))
    print("=" * 80)

    # 5. 预期原理解析
    print("\n【预期结果解析】")
    print("1. 任务A (通信型): 节点 1 胜出。原因：节点1距离调度器近，且带宽单价最低(0.334)。")
    print("2. 任务B (算力型): 节点 6 胜出。原因：带宽极低忽略了距离惩罚，节点6拥有全网最低CPU单价(0.1010)。")
    print("3. 任务C (存储型): 节点 8 胜出。原因：带宽极低忽略了距离惩罚，节点8拥有全网最低磁盘单价(0.035)。")

    # 保存结果
    summary_df.to_csv("task_comparison_optimized.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()