import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
csv_file = "train_loss.csv"  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_file)

# 查看数据结构，确保你有 'batch' 和 'loss' 两列
print(df.head())

# 提取 'batch' 和 'loss' 列数据
batches = df['Epoch']  # 假设 batch 列名为 'batch'
losses = df['Loss']    # 假设 loss 列名为 'loss'

# 绘制图形
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(batches, losses, label='Loss', color='blue')

# 添加标签和标题
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss vs Batch')

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
