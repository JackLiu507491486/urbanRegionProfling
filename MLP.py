import csv

import clip
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image_transform(image)

def preprocess_text(text_path):
    with open(text_path, 'r') as f:
        text = f.read().strip()
    return clip.tokenize(text, truncate=True).to(device)

def extract_features(image_paths, text_paths):
    image_features = []
    text_features = []

    for image_path, text_path in zip(image_paths, text_paths):
        # 提取图像特征
        image = preprocess_image(image_path).unsqueeze(0)  # 添加 batch 维度
        image = image.to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image)
        image_features.append(image_feature)

        # 提取文本特征
        text = preprocess_text(text_path)
        text = text.to(device)
        with torch.no_grad():
            text_feature = model.encode_text(text)
        text_features.append(text_feature)

    return torch.cat(image_features), torch.cat(text_features)


# 设置设备（CUDA 或 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的 CLIP 模型架构（这里使用 "ViT-B/32" 作为示例）
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载训练好的权重
model.load_state_dict(torch.load("clip_model.pth", map_location=device))


# 设置为评估模式
model.eval()

# 读取 CSV 文件
c_df = pd.read_csv('C.csv', usecols=['filename', 'sum_value'])
light_df = pd.read_csv('light.csv', usecols=['filename', 'sum_value'])
pop_df = pd.read_csv('pop.csv', usecols=['filename', 'sum_value'])

# 重命名 sum_value 列以避免合并时冲突
c_df = c_df.rename(columns={'sum_value': 'C'})
light_df = light_df.rename(columns={'sum_value': 'light'})
pop_df = pop_df.rename(columns={'sum_value': 'pop'})

# 合并三个 CSV 文件
labels_df = c_df.merge(light_df, on='filename').merge(pop_df, on='filename')


# 加载图像和文本
image_folder = 'shanghai_sat'
text_folder = 'generated_descriptions'

image_paths = []
text_paths = []
labels = []

# 遍历 CSV 文件中的每一行
for index, row in labels_df.iterrows():
    filename = row['filename']
    image_path = os.path.join(image_folder, filename.replace('.tif','')+'.tif')
    text_path = os.path.join(text_folder, filename.replace('.tif','')+'_description.txt')
    temp1 = os.path.exists(image_path)
    temp2 = os.path.exists(text_path)

    # 检查图像和文本文件是否存在
    if os.path.exists(image_path) and os.path.exists(text_path):
        image_paths.append(image_path)
        text_paths.append(text_path)
        labels.append([row['pop']])  # 提取 C 1个标签



# 图像预处理
image_transform = Compose([
    Resize((224, 224)),  # 调整图像大小
    ToTensor(),  # 转换为张量
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # CLIP 的归一化参数
])


# 划分数据集
train_image_paths, test_image_paths, train_text_paths, test_text_paths, train_labels, test_labels = train_test_split(
    image_paths, text_paths, labels, test_size=0.05, random_state=42
)

train_image_paths, val_image_paths, train_text_paths, val_text_paths, train_labels, val_labels = train_test_split(
    train_image_paths, train_text_paths, train_labels, test_size=0.1, random_state=42
)  # 0.25 * 0.8 = 0.2 验证集



# 提取特征
train_image_features, train_text_features = extract_features(train_image_paths, train_text_paths)
val_image_features, val_text_features = extract_features(val_image_paths, val_text_paths)
test_image_features, test_text_features = extract_features(test_image_paths, test_text_paths)

# 初始化 MLP 模型
input_dim = train_image_features.shape[1] + train_text_features.shape[1]  # 图像特征 + 文本特征
mlp = MLP(input_dim, hidden_dim=512, output_dim=1)  # 输出 3 个指标
mlp.to(device)
mlp = mlp.float()
# 合并图像和文本特征
train_features = torch.cat([train_image_features, train_text_features], dim=1)
train_features = train_features.float().to(device)
val_features = torch.cat([val_image_features, val_text_features], dim=1)
val_features = val_features.float().to(device)
test_features = torch.cat([test_image_features, test_text_features], dim=1)
test_features = test_features.float().to(device)

# 转换为 TensorDataset
train_dataset = TensorDataset(train_features, torch.tensor(train_labels, dtype=torch.float32))
val_dataset = TensorDataset(val_features, torch.tensor(val_labels, dtype=torch.float32))
test_dataset = TensorDataset(test_features, torch.tensor(test_labels, dtype=torch.float32))

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=1e-4)

train_losses = []

# 训练循环
for epoch in range(100):  # 训练 25 个 epoch
    mlp.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = mlp(features)
        labels = labels.to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


#保存为 CSV 文件
with open("mlp_light_train_loss.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss"])
    for i, loss in enumerate(train_losses):
        writer.writerow([i + 1, loss])

torch.save(mlp.state_dict(), 'mlp_model_C.pth')


metrics_val = {}
metrics_test = {}
mlp.eval()
with torch.no_grad():
    val_outputs = mlp(val_features).to('cpu')
    val_labels = torch.tensor(val_labels, dtype=torch.float32).to('cpu')
    val_loss = criterion(val_outputs,  val_labels)
    metrics_val["mse"] = mean_squared_error(val_outputs,  val_labels)
    metrics_val["r2"] = r2_score(val_outputs,  val_labels)
    metrics_val["rmse"] = np.sqrt(metrics_val["mse"])
    metrics_val["mae"] = mean_absolute_error(val_outputs,  val_labels)
    # metrics_test["mape"] = mean_absolute_percentage_error(val_outputs,  val_labels)
    print(f'Validation Loss: {metrics_val}')

    test_outputs = mlp(test_features).to('cpu')
    test_labels = torch.tensor(test_labels, dtype=torch.float32).to('cpu')
    test_loss = criterion(test_outputs, test_labels)
    metrics_test["mse"] = mean_squared_error(test_outputs, test_labels)
    metrics_test["r2"] = r2_score(test_outputs, test_labels)
    metrics_test["rmse"] = np.sqrt(metrics_test["mse"])
    metrics_test["mae"] = mean_absolute_error(test_outputs, test_labels)
    # metrics_test["mape"] = mean_absolute_percentage_error(test_outputs, test_labels)
    print(f'Test Loss: {metrics_test}')