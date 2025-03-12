import os
import csv
import math
import random
import numpy as np
from PIL import Image

import gym
import gym_maze
from gym.wrappers import OrderEnforcing

# ============ 包装器：将旧的 step API (返回4值) 转为新的 step API (返回5值) ============
class OldStepToNewWrapper(gym.Wrapper):
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            # 把 done 作为 terminated，truncated 固定为 False
            return obs, reward, done, False, info
        else:
            return result

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# ============ 辅助函数：探索率、学习率、离散化状态、动作选择 ============
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def state_to_bucket(state):
    """将连续状态映射为离散索引 (bucket)。"""
    bucket_indices = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i] - 1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)

def select_action(state, explore_rate):
    """ε-greedy 策略：以 explore_rate 概率随机探索，否则选择 Q 表中最大值的动作。"""
    if random.random() < explore_rate:
        return env.action_space.sample()
    else:
        return int(np.argmax(q_table[state]))

# ============ 主训练函数：多 run，多 episode，去重路径并保存 ============
def train_and_save():
    global explore_rate, learning_rate

    # ========== 创建保存结果的目录结构 ==========
    base_dir = "map_10x10_plus"  # 你可以改成 "map_4x4" 或其他名称
    images_dir = os.path.join(base_dir, "path_images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # CSV 文件路径
    csv_path = os.path.join(base_dir, "unique_paths_10x10_plus.csv")
    # 写 CSV 表头
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_id", "step", "state", "action", "reward", "next_state", "image_path"])

    # 全局路径 ID，每个 episode 对应一个 path_id
    path_id = 0
    # 动作序列去重
    seen_paths = set()

    for run_id in range(N_RUNS):
        # 如果每个 run 需要独立训练，就重置 Q 表
        q_table[:] = 0.0
        # 重置探索率、学习率
        explore_rate = get_explore_rate(0)
        learning_rate = get_learning_rate(0)

        print(f"\n===== Run {run_id + 1}/{N_RUNS}, 共 {EPISODES_PER_RUN} 个 episode =====")

        for ep in range(EPISODES_PER_RUN):
            obs = env.reset()
            if isinstance(obs, tuple) and len(obs) == 2:
                obs = obs[0]
            state = state_to_bucket(obs)

            done = False
            step = 0
            total_reward = 0

            # 临时保存该 episode 的所有步（包括图像），以便在确认不重复后再写入磁盘
            trajectory_data = []
            frames_list = []
            action_sequence = []

            while not done and step < MAX_T:
                action = select_action(state, explore_rate)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                next_state = state_to_bucket(next_obs)
                total_reward += reward
                step += 1

                # 记录动作
                action_sequence.append(action)

                # 获取图像，先暂存在内存
                frame = env.render(mode="rgb_array")
                frames_list.append(frame)

                # Q-learning 更新
                best_q = np.amax(q_table[next_state])
                q_table[state + (action,)] += learning_rate * (reward + DISCOUNT_FACTOR * best_q - q_table[state + (action,)])
                
                # 暂存一步的数据
                trajectory_data.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state
                })
                state = next_state

            # episode 结束，判断是否重复路径
            path_key = tuple(action_sequence)
            if path_key in seen_paths:
                # 如果已经出现过，跳过保存
                continue
            else:
                # 新路径，保存数据
                seen_paths.add(path_key)
                current_path_id = path_id
                path_id += 1

                # 将临时数据写入 CSV，并保存图像
                step_idx = 0
                for step_info, frame_array in zip(trajectory_data, frames_list):
                    step_idx += 1
                    image_filename = f"path_{current_path_id}_step_{step_idx}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    Image.fromarray(frame_array).save(image_path)

                    # 写入 CSV
                    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            current_path_id,
                            step_idx,
                            step_info["state"],
                            step_info["action"],
                            step_info["reward"],
                            step_info["next_state"],
                            image_path
                        ])

            # 动态更新探索率、学习率
            explore_rate = get_explore_rate(ep)
            learning_rate = get_learning_rate(ep)

            if DEBUG_MODE >= 1:
                print(f"Episode {ep} done in {step} steps, total reward={total_reward}")

    print("\n训练完成，所有不重复路径已写入 CSV。")

# ============ 主程序入口 ============
if __name__ == "__main__":
    # 初始化迷宫环境
    env_id = "maze-random-10x10-plus"  # 你可以换成其他
    env = gym.make(env_id)
    env = OldStepToNewWrapper(env)
    env = OrderEnforcing(env, disable_render_order_enforcing=True)

    # 配置训练参数
    N_RUNS = 10            # 训练 10 个 run
    EPISODES_PER_RUN = 200  # 每个 run 2000 个 episode
    MAX_T = 1000           # 每个 episode 最大步数
    DEBUG_MODE = 0         # 控制打印调试信息

    # 环境观测空间的离散化
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    # 学习超参数
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
    DISCOUNT_FACTOR = 0.99

    # 初始化 Q 表
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    # 初始化探索率和学习率
    explore_rate = get_explore_rate(0)
    learning_rate = get_learning_rate(0)

    # 开始训练
    train_and_save()

    # 关闭环境
    env.close()
