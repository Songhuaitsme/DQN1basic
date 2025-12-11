import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Dict, List
from dqn_agent import DQNAgent  # 确保导入的是 dqn_agent
from baseline_algorithms import BaselineAlgorithms
from data_loader import DataLoader

# === 【新增：解决matplotlib中文乱码问题】 ===
try:
    # 尝试使用 'SimHei' (黑体)
    plt.rcParams['font.sans-serif'] = ['SimHei']
except:
    try:
        # 如果黑体不行, 尝试 'Microsoft YaHei' (微软雅黑)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    except:
        # 如果都失败了, 打印一个警告 (但程序会继续)
        print("警告：未能设置中文字体（SimHei 或 Microsoft YaHei），图表可能显示乱码。")
# 解决保存图像时 负号'-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# =======================================

def smooth_curve(y: List[float], window_size: int = 100) -> np.ndarray:
    """滑动平均平滑曲线（匹配文献图5的平滑效果）"""
    if len(y) < window_size:
        return np.array(y)
    return np.convolve(y, np.ones(window_size) / window_size, mode="valid")


def main():
    # 1. 初始化模块
    dqn_agent = DQNAgent()
    baseline = BaselineAlgorithms()
    dc_ids = list(range(1, 12))
    num_episodes = dqn_agent.num_episodes

    # 2. DQN训练与收敛监控
    print(f"开始DQN训练（{num_episodes}个情节）...")
    episode_rewards = []
    for ep in range(num_episodes):
        total_reward = dqn_agent.train_episode(ep + 1)
        episode_rewards.append(total_reward)

        # 每1000个情节打印进度
        if (ep + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            # （DQN agent 内部的打印更频繁, 这里的打印用于在 TensorBoard 记录平滑值）
            print(
                f"=== 进度: {ep + 1:5d}/{num_episodes} | ε: {dqn_agent.epsilon:.4f} | 最近100情节平均奖励: {avg_reward:.2f} ===")

            # === 记录平滑奖励到 TensorBoard ===
            with dqn_agent.writer.as_default():
                tf.summary.scalar('avg_reward_100ep', avg_reward, step=ep + 1)
            # =================================

    # 3. 绘制收敛曲线
    smoothed_rewards = smooth_curve(episode_rewards)
    print("\n训练完成。正在生成最终收敛曲线...")
    plt.figure(figsize=(10, 6))
    # (标签和标题现在可以正确显示中文了)
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="DQN（滑动窗口=100）")
    plt.xlabel("情节数")
    plt.ylabel("平均奖励")
    plt.title("DQN训练收敛曲线")
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_convergence.png", dpi=300, bbox_inches="tight")
    print("\n收敛曲线已保存为：dqn_convergence.png")

    # === 【新增：保存模型】 ===
    print("\n正在保存DQN模型...")
    # 我们将 agent 内部的主网络(main_net)保存到磁盘
    # 根据 dqn_agent.py 中 DQNAgent 类的定义，self.main_net 是训练好的 tf.keras.Model 实例。
    dqn_agent.main_net.save("dqn_shortest_path_model")
    # [cite_end]
    print("模型已保存为：dqn_shortest_path_model 目录")
    # ==========================

    # 4. 运行DQN推理与基线算法
    print("\n开始DQN与基线算法（CRF/SPF）对比实验...")
    # 4.1 DQN结果
    dqn_results = {}
    for dc in dc_ids:
        # 推理时调用 get_shortest_path 找到路径
        path = dqn_agent.get_shortest_path(dc)
        # 使用 CostCalculator 计算总成本
        total_cost = dqn_agent.cost_calc.calculate_total_cost(dc, path)
        dqn_results[dc] = {"path": path, "total_cost": total_cost}

    dqn_valid_costs = [v["total_cost"] for v in dqn_results.values() if v["total_cost"] > 0]
    dqn_avg_cost = np.mean(dqn_valid_costs) if dqn_valid_costs else 0

    # 4.2 基线算法结果（CRF/SPF）
    # 调用 baseline_algorithms.py 中的 run_all_baselines
    crf_results, spf_results = baseline.run_all_baselines()

    crf_valid_costs = [v["total_cost"] for v in crf_results.values() if v["total_cost"] > 0]
    crf_avg_cost = np.mean(crf_valid_costs) if crf_valid_costs else 0

    spf_valid_costs = [v["total_cost"] for v in spf_results.values() if v["total_cost"] > 0]
    spf_avg_cost = np.mean(spf_valid_costs) if spf_valid_costs else 0

    # 5. 输出对比结果 (引用文献中的对比目标)
    # [cite_start]输出DQN、CRF、SPF的平均总成本 [cite: 270]
    print("\n=== 文献3.1.3节实验结论验证 [cite: 270] ===")
    print(f"DQN      平均总成本: {dqn_avg_cost:.2f}")
    print(f"CRF      平均总成本: {crf_avg_cost:.2f}")
    print(f"SPF      平均总成本: {spf_avg_cost:.2f}")

    if crf_avg_cost > 0:
        crf_reduction = ((crf_avg_cost - dqn_avg_cost) / crf_avg_cost * 100)
        # [cite_start]与文献的CRF降幅目标(3.6%)对比 [cite: 270]
        print(f"DQN较CRF降幅: {crf_reduction:.2f}% (文献目标：3.6% )")
    if spf_avg_cost > 0:
        spf_reduction = ((spf_avg_cost - dqn_avg_cost) / spf_avg_cost * 100)
        # [cite_start]与文献的SPF降幅目标(10.0%)对比 [cite: 270]
        print(f"DQN较SPF降幅: {spf_reduction:.2f}% (文献目标：10.0% )")

    # 6. 保存详细结果到CSV
    result_data = []
    for dc in dc_ids:
        result_data.append({
            "数据中心ID": dc,
            "DQN路径": str(dqn_results.get(dc, {}).get("path", "N/A")),
            "DQN总成本": dqn_results.get(dc, {}).get("total_cost", 0),
            "CRF总成本": crf_results.get(dc, {}).get("total_cost", 0),
            "SPF总成本": spf_results.get(dc, {}).get("total_cost", 0)
        })
    pd.DataFrame(result_data).to_csv("data_center_selection_results.csv", index=False, encoding="utf-8-sig")
    print("\n详细结果已保存为：data_center_selection_results.csv")


if __name__ == "__main__":
    main()