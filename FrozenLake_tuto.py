from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


sns.set_theme()

# %load_ext lab_black




class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved


params = Params(
    total_episodes=200,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=10,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("./"),
)
params

# Set the seed
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exist
params.savefig_folder.mkdir(parents=True, exist_ok=True)



env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    ),
)


params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print(f"Action size: {params.action_size}")
print(f"State size: {params.state_size}")


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

# 辅助函数：根据当前状态和地图规格，返回允许的动作列表
def get_allowed_actions(state, map_size, total_actions):
    row = state // map_size
    col = state % map_size
    allowed = list(range(total_actions))
    # 当在最左侧时，不允许向左（动作 0）
    if col == 0 and 0 in allowed:
        allowed.remove(0)
    # 当在最右侧时，不允许向右（动作 2）
    if col == map_size - 1 and 2 in allowed:
        allowed.remove(2)
    # 当在最上侧时，不允许向上（动作 3）
    if row == 0 and 3 in allowed:
        allowed.remove(3)
    # 当在最下侧时，不允许向下（动作 1）
    if row == map_size - 1 and 1 in allowed:
        allowed.remove(1)
    return allowed

class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable, map_size):
        total_actions = qtable.shape[1]
        allowed_actions = get_allowed_actions(state, map_size, total_actions)
        # 如果没有允许的动作，则返回环境随机动作（理论上不应出现）
        if not allowed_actions:
            return action_space.sample()
            
        if rng.uniform(0, 1) < self.epsilon:
            # 探索：从允许的动作中随机选择
            return rng.choice(allowed_actions)
        else:
            # 利用：在允许的动作中选取 Q 值最大的动作
            q_values = qtable[state, :]
            # 只考虑允许的动作
            allowed_q = {a: q_values[a] for a in allowed_actions}
            max_val = max(allowed_q.values())
            # 可能存在多个动作 Q 值相同，则随机选取一个
            best_actions = [a for a, v in allowed_q.items() if v == max_val]
            return rng.choice(best_actions)

# 辅助函数：将状态编号转为行列（根据 map_size）
def state_to_pos(s, map_size):
    return (s // map_size, s % map_size)


#######################################################################################################################
#######################################################################################################################
# ----- 针对每个 map_size 进行实验 -----
map_sizes = [4,7,9,11]

# 对于每个 map_size，我们生成单独的结果文件夹和 CSV 文件
for map_size in map_sizes:
    print(f"\n======= 当前 map_size: {map_size}x{map_size} =======")
    # 更新 params 中的 map_size
    current_params = params._replace(map_size=map_size)
    
    # 为当前 map_size 创建专用文件夹
    save_dir = current_params.savefig_folder / f"map_{map_size}x{map_size}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 图片文件夹与 CSV 文件名
    IMAGE_DIR = save_dir / "path_images"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = save_dir / f"unique_paths_{map_size}x{map_size}.csv"
    
    # 创建环境（使用当前 map_size）
    current_map_desc = generate_random_map(size=map_size, p=current_params.proba_frozen, seed=current_params.seed)
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=current_params.is_slippery,
        render_mode="rgb_array",
        desc=current_map_desc,
    )
    # 更新动作与状态空间尺寸
    current_params = current_params._replace(
        action_size=env.action_space.n,
        state_size=env.observation_space.n
    )
    env.action_space.seed(current_params.seed)
    
    # 重新初始化 learner 和 explorer
    learner = Qlearning(
        learning_rate=current_params.learning_rate,
        gamma=current_params.gamma,
        state_size=current_params.state_size,
        action_size=current_params.action_size,
    )
    explorer = EpsilonGreedy(epsilon=current_params.epsilon)
    
    # 用于去重：记录已出现的动作序列
    seen_paths = set()
    unique_path_rows = []  # 保存每个路径每一步的记录
    path_id_counter = 0
    RANDOM_EXTRA_STEPS = 2  # 掉入洞后追加随机动作步数

    # 针对每个 run 与每个 episode 进行采样
    for run in range(current_params.n_runs):
        learner.reset_qtable()  # 每个 run 重置 Q 表
        for episode in tqdm(range(current_params.total_episodes), desc=f"Run {run}/{current_params.n_runs}"):
            state, _ = env.reset(seed=current_params.seed)
            done = False
            step_count = 0
            trajectory = []       # 记录当前 episode 每一步的信息
            action_sequence = []  # 用于去重的动作序列

            # 限制最大步数（防止无限循环）
            while not done and step_count < 90:
                action = explorer.choose_action(env.action_space, state, learner.qtable, map_size)
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                frame = env.render()  # 获取当前帧图片

                trajectory.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": new_state,
                    "frame": frame
                })
                action_sequence.append(action)
                # Q-learning 更新（可选）
                learner.qtable[state, action] = learner.update(state, action, reward, new_state)
                state = new_state
                step_count += 1

                # 检查当前 cell 类型
                row, col = state_to_pos(new_state, map_size)
                cell = current_map_desc[row][col] 
                if cell in ['H', 'G']:
                    done = True
                    # 如果掉入洞，按 50% 可能性决定是否追加随机动作
                    if cell == 'H':
                        if rng.random() >= 0.5:
                            for _ in range(RANDOM_EXTRA_STEPS):
                                extra_action = env.action_space.sample()
                                extra_frame = env.render()
                                trajectory.append({
                                    "state": state,
                                    "action": extra_action,
                                    "reward": 0,
                                    "next_state": state,
                                    "frame": extra_frame
                                })
                                action_sequence.append(extra_action)
                    break

            # episode 结束后，无论成功或失败，都直接保存路径
            path_key = tuple(action_sequence)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                current_path_id = path_id_counter
                path_id_counter += 1

                # 遍历该路径的每一步，保存图片并记录 CSV 行
                for step_idx, step_info in enumerate(trajectory):
                    image_filename = f"path_{current_path_id}_step_{step_idx}.png"
                    image_path = str(IMAGE_DIR / image_filename)
                    plt.imsave(image_path, step_info["frame"])
                    unique_path_rows.append({
                        "path_id": current_path_id,
                        "step": step_idx,
                        "state": step_info["state"],
                        "action": step_info["action"],
                        "reward": step_info["reward"],
                        "next_state": step_info["next_state"],
                        "image_path": image_path
                    })

    env.close()
    import csv
    # 将当前 map_size 的路径记录写入 CSV 文件
    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_id", "step", "state", "action", "reward", "next_state", "image_path"])
        for row in unique_path_rows:
            writer.writerow([
                row["path_id"],
                row["step"],
                row["state"],
                row["action"],
                row["reward"],
                row["next_state"],
                row["image_path"]
            ])
    print(f"map {map_size}x{map_size}：共收集到 {path_id_counter} 条不重复路径，路径详情保存于 {CSV_PATH}，图像保存在文件夹 {IMAGE_DIR}")

#######################################################################################################################

def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st