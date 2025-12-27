'''单个任务不同节点成本对比'''

import numpy as np
import pandas as pd
import tensorflow as tf
from dqn_agent import DQNAgent
from data_loader import DataLoader
from cost_calculator import CostCalculator

# 1. 定义文献原始任务 (参照 evaluate_model.py 中的 TEST_CASE_1)
ORIGINAL_TASK = {
    "name": "文献原始任务 (高CPU, 中带宽)",
    "cpu_cores": 2048,
    "ram_gb": 4,
    "disk_gb": 128,
    "bandwidth_mb": 200,
    "run_time_h": 1
}


def main():
    print(f"=== 开始执行任务: {ORIGINAL_TASK['name']} ===")
    print(f"任务需求: CPU={ORIGINAL_TASK['cpu_cores']}核, 带宽={ORIGINAL_TASK['bandwidth_mb']}Mb")

    # 2. 初始化 DQN 智能体
    agent = DQNAgent()

    # 尝试加载训练好的模型 (如果存在)
    model_path = "dqn_shortest_path_model"
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        agent.main_net = loaded_model
        print(f"成功加载预训练模型: {model_path}")
        # 推理模式：关闭随机探索
        agent.epsilon = 0.0
    except:
        print("警告: 未找到预训练模型，将使用未训练的随机策略进行演示。")
        print("请先运行 main.py 进行训练。")

    # 3. 注入任务需求到成本计算器
    # 这一步至关重要，否则计算的是 data_loader.py 里的默认任务
    agent.cost_calc.task_demand = ORIGINAL_TASK

    # 4. 遍历所有数据中心 (ID 1-11) 运行 DQN 并计算成本
    results = []
    dc_ids = list(range(1, 12))

    print("\n正在计算各节点成本...")
    for dc in dc_ids:
        # 步骤 A: 利用 DQN 规划路径 (从 DC -> 调度中心)
        path = agent.get_shortest_path(dc)

        # 步骤 B: 计算成本 (资源成本 + 通信成本)
        total_cost = agent.cost_calc.calculate_total_cost(dc, path)
        resource_cost = agent.cost_calc.calculate_resource_cost(dc)
        comm_cost = agent.cost_calc.calculate_communication_cost(path)

        results.append({
            "DC_ID": dc,
            "总成本": total_cost,
            "资源成本": resource_cost,
            "通信成本": comm_cost,
            "规划路径": str(path)
        })

    # 5. 转换为 DataFrame 并找出最优解
    df = pd.DataFrame(results)

    # 找到总成本最低的行
    best_row = df.loc[df["总成本"].idxmin()]
    best_dc_id = int(best_row["DC_ID"])
    min_cost = best_row["总成本"]

    # 6. 输出详细对比表
    print("\n=== 不同节点成本对比 (DQN算法) ===")
    # 格式化输出，标注最优
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # 添加一列 "标注"
    df["标注"] = df["DC_ID"].apply(lambda x: "★ 最优推荐" if x == best_dc_id else "")

    print(df.set_index("DC_ID"))

    print("\n" + "=" * 40)
    print(f"结论: 对于该原始任务，DQN 算法推荐的最优数据中心是:【节点 {best_dc_id}】")
    print(f"最低总成本: {min_cost:.2f}")
    print(f"规划路径: {best_row['规划路径']}")
    print("=" * 40)

    # 可选：保存到 CSV
    df.to_csv("dqn_task_optimization_result.csv", index=False, encoding="utf-8-sig")
    print("\n结果已保存至 dqn_task_optimization_result.csv")


if __name__ == "__main__":
    main()