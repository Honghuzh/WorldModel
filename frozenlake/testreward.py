import pandas as pd

# 1. 读取 CSV 文件
df = pd.read_csv("./map_7x7/unique_paths_7x7.csv")

# 2. 判断是否存在 reward == 1.0 的行
has_reward_one = (df["reward"] == 1.0).any()
print("是否存在 reward == 1.0 的行:", has_reward_one)

# 3. 如果需要查看所有 reward == 1.0 的行，可以这样筛选
df_reward_one = df[df["reward"] == 1.0]
print("所有 reward == 1.0 的行:")
print(df_reward_one)
