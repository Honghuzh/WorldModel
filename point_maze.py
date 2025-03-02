import gymnasium as gym
import gymnasium_robotics

# 注册环境
gym.register_envs(gymnasium_robotics)

# 创建 PointMaze_UMazeDense-v3 环境
env = gym.make('PointMaze_UMazeDense-v3', max_episode_steps=100, render_mode="human",width=3000,height=2600)

# 初始化视频录制
frames = []
# 初始化环境
obs, info = env.reset()

# 渲染并执行环境的一个回合
for _ in range(1000):
    env.render()  # 渲染环境，确保 render_mode 为 "human"
    action = env.action_space.sample()  # 随机采样一个动作
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    if terminated or truncated:
        break

env.close()
# 保存为GIF文件
imageio.mimsave("point_maze.gif", frames, fps=30)
