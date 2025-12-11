# evaluate_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List

# 导入我们的 DRL 智能体和基线算法
from dqn_agent import DQNAgent
from baseline_algorithms import BaselineAlgorithms

# (data_loader, cost_calculator 等会被 Agent 自动导入)

# --- 在这里定义你的新测试用例 ---

TEST_CASE_1 = {
    "name": "文献原始任务 (高CPU, 中带宽)",
    "cpu_cores": 2048,
    "ram_gb": 4,
    "disk_gb": 128,
    "bandwidth_mb": 200,
    "run_time_h": 1
}

# 测试用例 2: 带宽密集型任务
TEST_CASE_2 = {
    "name": "带宽密集型任务 (高带宽)",
    "cpu_cores": 512,  # CPU 需求低
    "ram_gb": 4,
    "disk_gb": 64,
    "bandwidth_mb": 1000,  # 带宽需求极高
    "run_time_h": 1
}

# 测试用例 3: 算力密集型任务
TEST_CASE_3 = {
    "name": "算力密集型任务 (高CPU)",
    "cpu_cores": 8192,  # CPU 需求极高
    "ram_gb": 16,
    "disk_gb": 256,
    "bandwidth_mb": 100,  # 带宽需求低
    "run_time_h": 1
}

# 将所有测试用例放入列表
ALL_TEST_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


# ------------------------------------

def run_evaluation(agent: DQNAgent, baseline: BaselineAlgorithms, test_case: Dict):
    """
    使用指定的测试用例运行一次完整的评估
    """
    print(f"\n--- 正在测试: {test_case['name']} ---")
    print(f"    需求: CPU: {test_case['cpu_cores']}, 带宽: {test_case['bandwidth_mb']}Mb")

    # 关键一步: 将新的任务需求 "注入" 到 agent 和 baseline 的成本计算器中
    # 这会覆盖它们在初始化时从 data_loader 加载的默认值
    agent.cost_calc.task_demand = test_case
    baseline.cost_calc.task_demand = test_case
    baseline.task_demand = test_case  # 基线算法类也存了一份

    # --- 开始评估 ---

    dc_ids = list(range(1, 12))  # 11 个数据中心
    results = []

    # 1. 运行 DQN
    dqn_paths = {}
    dqn_costs = {}
    for dc in dc_ids:
        path = agent.get_shortest_path(dc)
        cost = agent.cost_calc.calculate_total_cost(dc, path)
        dqn_paths[dc] = str(path)
        dqn_costs[dc] = cost

    # 2. 运行基线
    crf_results, spf_results = baseline.run_all_baselines()

    # 3. 汇总结果
    for dc in dc_ids:
        results.append({
            "数据中心ID": dc,
            "DQN 成本": dqn_costs.get(dc, 0),
            "CRF 成本": crf_results.get(dc, {}).get("total_cost", 0),
            "SPF 成本": spf_results.get(dc, {}).get("total_cost", 0),
            "DQN 路径": dqn_paths.get(dc, "N/A")
        })

    # 4. 打印结果
    df = pd.DataFrame(results).set_index("数据中心ID")
    print("\n--- 逐项成本对比 ---")
    print(df)

    # 计算并打印平均值
    print("\n--- 平均成本总结 ---")
    avg_dqn = df["DQN 成本"].mean()
    avg_crf = df["CRF 成本"].mean()
    avg_spf = df["SPF 成本"].mean()

    print(f"DQN 平均总成本: {avg_dqn:.2f}")
    print(f"CRF 平均总成本: {avg_crf:.2f}")
    print(f"SPF 平均总成本: {avg_spf:.2f}")

    # 打印降幅
    if avg_crf > 0:
        print(f"DQN 较 CRF 降幅: {((avg_crf - avg_dqn) / avg_crf * 100):.2f}%")
    if avg_spf > 0:
        print(f"DQN 较 SPF 降幅: {((avg_spf - avg_dqn) / avg_spf * 100):.2f}%")


def main():
    print("=================================================")
    print("          DQN 模型评估脚本 (12 节点)           ")
    print("=================================================")
    print("... 确保您的 data_loader.py 是 12 节点版本 ...")

    model_path = "dqn_shortest_path_model"

    try:
        # 1. 加载预训练的 Keras 模型
        print(f"\n正在从 '{model_path}' 加载预训练模型...")
        loaded_model = tf.keras.models.load_model(model_path)
        print("模型加载成功。")

        # 2. 初始化智能体和基线
        # (我们仍然需要初始化 DQNAgent, 因为需要它的环境和方法)
        agent = DQNAgent()

        # 关键一步: 用我们加载的“大脑”替换掉它刚初始化的“大脑”
        agent.main_net = loaded_model

        # (推理时不需要探索, 将 epsilon 设为 0)
        agent.epsilon = 0.0

        baseline = BaselineAlgorithms()

        # 3. 循环执行所有测试用例
        for case in ALL_TEST_CASES:
            run_evaluation(agent, baseline, case)
            print("-" * 50)

        print("\n评估完成。")

    except IOError:
        print(f"\n错误: 找不到模型目录 '{model_path}'。")
        print("请先运行 douban_spider.py 训练并保存模型。")
    except Exception as e:
        print(f"\n发生意外错误: {e}")


if __name__ == "__main__":
    main()