import numpy as np
import tensorflow as tf
from collections import deque
from typing import List, Tuple, Dict
from topology_builder import TopologyBuilder
from cost_calculator import CostCalculator
from data_loader import DataLoader
import datetime
import random


# 子模块1: 标准经验回放池
class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        # 匹配文献表2的内存容量
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """向经验池添加一个 (s, a, r, s', done) 元组"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Tuple]:
        """从经验池中随机采样一个批次"""
        if len(self.buffer) < batch_size:
            return []  # 经验不足
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# 子模块2: DQN网络(文献2.1节结构)
class DQNNetwork(tf.keras.Model):
    def __init__(self, action_dim: int):
        super().__init__()
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError(f"action_dim 必须是一个正整数, 但收到了: {action_dim}")
        # 文献表2指定2个隐藏层
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")  # 隐藏层1
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")  # 隐藏层2
        self.output_layer = tf.keras.layers.Dense(action_dim, activation="linear")  # 动作Q值输出

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# 子模块3: 标准DQN智能体 (文献2.2节算法意图)
class DQNAgent:
    def __init__(self):
        self.topology = TopologyBuilder()
        self.cost_calc = CostCalculator()
        self.price_df = DataLoader.load_data_center_prices()

        # === 【关键修复 1：状态表示】 ===
        # 获取节点总数 (0~11, 共12个)
        self.num_nodes = len(self.topology.get_graph().nodes())
        # 状态维度不再是1, 而是节点的数量 (用于One-Hot编码)
        self.state_dim = self.num_nodes  # (原: 1)

        # 动作空间维度 (不变)
        self.max_action_dim = 1
        if self.topology.get_graph().nodes():
            self.max_action_dim = max(
                max(len(self.topology.get_adjacent_nodes(n)) for n in self.topology.get_graph().nodes()) or 1, 1
            )

        # === 【超参数 (使用上一版稳定的参数)】 ===
        self.learning_rate = 0.0001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 10000
        self.batch_size = 128
        self.target_update_freq = 10
        self.num_episodes = 17000

        # === 【网络初始化 (现在DQNNetwork的输入层维度是 self.state_dim)】 ===
        self.main_net = DQNNetwork(self.max_action_dim)
        self.target_net = DQNNetwork(self.max_action_dim)
        self.target_net.set_weights(self.main_net.get_weights())

        # 梯度裁剪 (不变)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.replay_buffer = ReplayBuffer(capacity=5000)
        self.train_step_count = 0

        # === TensorBoard 设置 ===
        log_dir = "logs/dqn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = tf.summary.create_file_writer(log_dir)
        print(f"TensorBoard 日志已启动, 请运行: tensorboard --logdir {log_dir}")
        self._validate_topology()

    def _validate_topology(self):
        """验证拓扑结构, 确保所有节点都有相邻节点"""
        graph = self.topology.get_graph()
        if not graph.nodes():
            print("警告: 拓扑图为空。")
            return
        for node in graph.nodes():
            if len(self.topology.get_adjacent_nodes(node)) == 0 and node != 0:
                print(f"警告: 节点 {node} 没有相邻节点")

    # === 【关键修复 2：One-Hot 编码辅助函数】 ===
    def _one_hot_encode(self, node_id: int) -> np.ndarray:
        """将节点ID (例如 3) 转换为 One-Hot 向量 (例如 [0,0,0,1,0...])"""
        encoded = np.zeros(self.num_nodes, dtype=np.float32)
        if 0 <= node_id < self.num_nodes:
            encoded[node_id] = 1.0
        return encoded

    def choose_action(self, state: np.ndarray) -> int:
        """ε-greedy动作选择(文献2.1节)"""

        # === 【关键修复 3：从 state 中找出 current_node】 ===
        # state 不再是 [node_id], 而是 one-hot 向量
        current_node = np.argmax(state)  # (one-hot 向量中值为1的索引就是节点ID)

        adjacent_nodes = self.topology.get_adjacent_nodes(current_node)
        num_valid_actions = len(adjacent_nodes)

        if num_valid_actions == 0:
            return -1  # 返回一个无效动作索引

        if np.random.random() < self.epsilon:
            return np.random.choice(num_valid_actions)  # 随机探索
        else:
            # state 已经是正确的 one-hot 格式, 加一维 (batch_size=1)
            state_tensor = tf.expand_dims(state, axis=0)
            q_values = self.main_net(state_tensor)

            # 仅在有效动作空间内选择
            valid_q_slice = min(num_valid_actions, self.max_action_dim)
            if valid_q_slice == 0:
                return -1

            valid_q_values = q_values[0, :valid_q_slice]
            return tf.argmax(valid_q_values, axis=0).numpy()  # 贪心选择

    # ===【奖励函数 (已在上一版修复, 保持不变)】===
    def calculate_reward(self, current_node: int, next_node: int, done: bool) -> float:
        """
        奖励函数 (修正为标准的最短路径奖励)
        """
        if done:
            return 100.0
        if next_node not in self.topology.get_adjacent_nodes(current_node):
            return -10.0
        return -1.0

    @tf.function
    def _train_step(self,
                    states: tf.Tensor,
                    actions: tf.Tensor,
                    rewards: tf.Tensor,
                    next_states: tf.Tensor,
                    dones: tf.Tensor) -> tf.Tensor:
        """单步训练(更新主网络) - (此函数内部逻辑不变, 因为Tensors的维度已正确)"""

        batch_size = tf.shape(states)[0]
        action_dim = tf.shape(self.main_net(states))[1]

        with tf.GradientTape() as tape:
            # 计算目标Q值 (Double DQN)
            next_q_main = self.main_net(next_states)
            next_actions = tf.argmax(next_q_main, axis=1, output_type=tf.int32)
            next_actions_clipped = tf.clip_by_value(next_actions, 0, action_dim - 1)

            next_q_target_full = self.target_net(next_states)
            indices = tf.stack([tf.range(batch_size, dtype=tf.int32), next_actions_clipped], axis=1)
            next_q_target_values = tf.gather_nd(next_q_target_full, indices)

            target_q = rewards + self.gamma * next_q_target_values * (1.0 - dones)

            # 计算当前Q值与损失
            current_q_full = self.main_net(states)
            current_actions_clipped = tf.clip_by_value(actions, 0, action_dim - 1)
            current_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), current_actions_clipped], axis=1)

            current_q_values = tf.gather_nd(current_q_full, current_indices)

            loss = tf.keras.losses.MeanSquaredError()(target_q, current_q_values)

        # 更新主网络
        gradients = tape.gradient(loss, self.main_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_net.trainable_variables))

        return loss

    def train_episode(self, current_episode_num: int) -> float:
        """单情节训练(从随机数据中心出发, 返回调度服务器0)"""

        all_nodes = list(self.topology.get_graph().nodes())
        valid_start_nodes = [n for n in all_nodes if n != 0]
        if not valid_start_nodes:
            print("错误: 没有有效的数据中心起始节点。")
            return 0.0

        start_dc = np.random.choice(valid_start_nodes)

        # === 【关键修复 4：使用 One-Hot 编码】 ===
        state = self._one_hot_encode(start_dc)  # (原: np.array([start_dc], ...))

        total_reward = 0.0
        total_loss = 0.0
        train_steps_in_episode = 0
        done = False
        step_count = 0
        max_steps_per_episode = 1000

        while not done and step_count < max_steps_per_episode:
            step_count += 1
            # choose_action 现在接受 one-hot 向量
            action = self.choose_action(state)

            if action == -1:
                break

            # (current_node 现在从 one-hot state 中解码)
            current_node = np.argmax(state)
            adjacent_nodes = self.topology.get_adjacent_nodes(current_node)

            if action < 0 or action >= len(adjacent_nodes):
                print(f"错误: 无效动作 {action} (来自节点 {current_node}), 有效范围 0-{len(adjacent_nodes) - 1}")
                break

            next_node = adjacent_nodes[action]

            # === 【关键修复 5：使用 One-Hot 编码】 ===
            next_state = self._one_hot_encode(next_node)  # (原: np.array([next_node], ...))
            done = (next_node == 0)

            reward = self.calculate_reward(current_node, next_node, done)
            total_reward += reward

            # 存储 (s, a, r, s', done) (s 和 s' 现在是 one-hot)
            experience = (state, action, reward, next_state, done)
            self.replay_buffer.add(experience)

            # 经验池满批量后训练
            if len(self.replay_buffer) >= self.batch_size:
                try:
                    experiences = self.replay_buffer.sample(self.batch_size)
                    if not experiences:
                        continue

                    # 批量转换 Tensors (states_batch 和 next_states_batch 现在是 [batch_size, 12])
                    states_batch = tf.convert_to_tensor([e[0] for e in experiences], dtype=tf.float32)
                    actions_batch = tf.convert_to_tensor([e[1] for e in experiences], dtype=tf.int32)
                    rewards_batch = tf.convert_to_tensor([e[2] for e in experiences], dtype=tf.float32)
                    next_states_batch = tf.convert_to_tensor([e[3] for e in experiences], dtype=tf.float32)
                    dones_batch = tf.convert_to_tensor([e[4] for e in experiences], dtype=tf.float32)

                    # 执行训练
                    loss = self._train_step(
                        states_batch, actions_batch, rewards_batch,
                        next_states_batch, dones_batch
                    )

                    total_loss += tf.reduce_mean(loss).numpy()
                    train_steps_in_episode += 1
                    self.train_step_count += 1

                    if self.train_step_count % 100 == 0:
                        with self.writer.as_default():
                            tf.summary.scalar('train_loss', loss, step=self.train_step_count)

                except Exception as e:
                    print(f"训练步骤错误: {e}")

            state = next_state  # (state 已经是 one-hot)

        # 线性衰减 Epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # 按 *情节* 更新目标网络
        if current_episode_num % self.target_update_freq == 0:
            self.target_net.set_weights(self.main_net.get_weights())

        if step_count > 0:
            avg_loss = total_loss / train_steps_in_episode if train_steps_in_episode > 0 else 0.0

            if current_episode_num % 100 == 0:
                print(
                    f"情节 {current_episode_num:5d}: {start_dc:2d}->0 | 步数: {step_count:3d} | R: {total_reward:6.2f} | ε: {self.epsilon:.4f} | AvgLoss: {avg_loss:.4f}")

            with self.writer.as_default():
                tf.summary.scalar('episode_reward', total_reward, step=current_episode_num)
                tf.summary.scalar('episode_steps', step_count, step=current_episode_num)
                tf.summary.scalar('epsilon', self.epsilon, step=current_episode_num)
                if train_steps_in_episode > 0:
                    tf.summary.scalar('avg_episode_loss', avg_loss, step=current_episode_num)
        return total_reward

    def get_shortest_path(self, data_center_id: int) -> List[int]:
        """推理: 获取调度服务器(0)到目标数据中心的最短路径"""

        # === 【关键修复 6：使用 One-Hot 编码】 ===
        state = self._one_hot_encode(data_center_id)  # (原: np.array([data_center_id], ...))

        path = [data_center_id]
        done = False
        max_inference_steps = 100

        while not done and max_inference_steps > 0:
            max_inference_steps -= 1

            # (current_node 现在从 one-hot state 中解码)
            current_node = np.argmax(state)
            adjacent_nodes = self.topology.get_adjacent_nodes(current_node)
            num_valid_actions = len(adjacent_nodes)

            if num_valid_actions == 0:
                print(f"推理错误: 节点 {current_node} 没有相邻节点")
                return path

            # 仅使用贪心策略 (Epsilon=0)
            state_tensor = tf.expand_dims(state, axis=0)  # (state 已经是 one-hot)
            q_values = self.main_net(state_tensor)

            valid_q_slice = min(num_valid_actions, self.max_action_dim)
            if valid_q_slice == 0:
                print(f"推理错误: 节点 {current_node} 没有有效的Q值")
                return path

            valid_q_values = q_values[0, :valid_q_slice].numpy()
            if len(valid_q_values) == 0:
                print(f"推理错误: 节点 {current_node} 没有有效的Q值 (切片后)")
                return path

            action = np.argmax(valid_q_values)

            if action < 0 or action >= len(adjacent_nodes):
                print(f"推理错误: 选择了无效的动作索引 {action}")
                return path

            next_node = adjacent_nodes[action]

            if next_node in path:
                print(f"推理警告: 检测到环路 (节点 {next_node} 已在路径中), 终止。")
                path.append(next_node)
                return path

            path.append(next_node)
            done = (next_node == 0)

            # === 【关键修复 7：使用 One-Hot 编码】 ===
            state = self._one_hot_encode(next_node)  # (原: np.array([next_node], ...))

        if max_inference_steps == 0 and not done:
            print(f"推理警告: 达到最大步数, 路径未到达节点0。")

        # 反转路径: DC->0 变为 0->DC
        return path[::-1]