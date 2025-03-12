import os
import csv
import math
import random
import numpy as np
from PIL import Image

import gym
import gym_maze
from gym.wrappers import OrderEnforcing

# ---------------------------
# 将旧的 step API (返回4个值) 转为新的 step API (返回5个值) 的包装器
# ---------------------------
class OldStepToNewWrapper(gym.Wrapper):
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            # 将 done 视为 terminated，truncated 固定为 False
            return obs, reward, done, False, info
        else:
            return result

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# ---------------------------
# 辅助函数：探索率、学习率、离散化状态、动作选择
# ---------------------------
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def state_to_bucket(state):
    """ 将连续状态映射为离散索引 (bucket)。 """
    bucket_indices = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)

def select_action(state, explore_rate):
    """ ε-greedy 策略：以 explore_rate 概率随机探索，否则选择 Q 值最大动作。 """
    if random.random() < explore_rate:
        return env.action_space.sample()
    else:
        return int(np.argmax(q_table[state]))

# ---------------------------
# 训练：10 个 run，每个 run 有 2000 个 episode
# ---------------------------
def train_and_save():
    """
    每个 run 都从头开始训练 Q-Table，共进行 2000 个 episode。
    每个 episode 生成一个 path，对应一条从 reset 到 done (或超时) 的轨迹。
    """
    global explore_rate, learning_rate

    # 创建保存结果的文件夹结构
    base_dir = "map_4x4"
    images_dir = os.path.join(base_dir, "path_images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # CSV 文件路径
    csv_path = os.path.join(base_dir, "unique_paths_4x4.csv")
    
    # 写 CSV 的表头
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_id", "step", "state", "action", "reward", "next_state", "image_path"])

    # path_id 全局递增，每个 episode 视为一个 path
    path_id = 0

    for run_id in range(N_RUNS):
        # 每个 run 重置 Q 表 (如需独立训练)
        q_table[:] = 0.0
        # 重置探索率与学习率
        explore_rate = get_explore_rate(0)
        learning_rate = get_learning_rate(0)
        
        print(f"\n===== Run {run_id + 1}/{N_RUNS} 开始，共 {EPISODES_PER_RUN} 个 episode =====")

        for ep in range(EPISODES_PER_RUN):
            # reset 环境，得到初始 state
            obs = env.reset()
            if isinstance(obs, tuple) and len(obs) == 2:
                obs = obs[0]
            state = state_to_bucket(obs)

            done = False
            total_reward = 0
            step = 0

            # 每个 episode 对应一个 path
            # path_id 用于区分不同路径
            current_path_id = path_id
            path_id += 1

            # 轨迹执行
            while not done and step < MAX_T:
                action = select_action(state, explore_rate)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = state_to_bucket(next_obs)
                total_reward += reward
                step += 1

                # 渲染当前帧
                frame = env.render(mode="rgb_array")
                image_filename = f"path_{current_path_id}_step_{step}.png"
                image_path = os.path.join(images_dir, image_filename)
                # 保存图像
                Image.fromarray(frame).save(image_path)

                # 更新 Q 表
                best_q = np.amax(q_table[next_state])
                q_table[state + (action,)] += learning_rate * (reward + DISCOUNT_FACTOR * best_q - q_table[state + (action,)])
                
                # 写入 CSV
                with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        current_path_id,
                        step,
                        state,
                        action,
                        reward,
                        next_state,
                        image_path
                    ])

                state = next_state

            # 动态更新探索率、学习率
            explore_rate = get_explore_rate(ep)
            learning_rate = get_learning_rate(ep)
            
            if DEBUG_MODE >= 1:
                print(f"Episode {ep} done in {step} steps, total reward={total_reward}")

    print("\n训练完成，所有路径均已保存到 CSV 中。")

# ---------------------------
# 主程序入口
# ---------------------------
if __name__ == "__main__":
    # 创建环境
    env = gym.make("maze-random-5x5")  # 这里假设有一个 4x4 的迷宫环境
    env = OldStepToNewWrapper(env)
    # 禁用 render 顺序检查
    env = OrderEnforcing(env, disable_render_order_enforcing=True)

    # 配置参数
    N_RUNS = 10            # 训练 10 个 run
    EPISODES_PER_RUN = 2000  # 每个 run 2000 个 episode
    MAX_T = 1000           # 每个 episode 的最大步数，避免无限循环
    DEBUG_MODE = 0         # 是否打印中间信息
    
    # 获取迷宫大小
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    # 学习相关
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
    DISCOUNT_FACTOR = 0.99

    # 初始化 Q 表
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    # 初始化全局探索率、学习率
    explore_rate = get_explore_rate(0)
    learning_rate = get_learning_rate(0)

    # 开始训练
    train_and_save()

    # 关闭环境
    env.close()
