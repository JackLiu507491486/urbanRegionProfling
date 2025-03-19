import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip
import torch.optim as optim
import re
import csv

# 自定义数据集类
class ShanghaiSatDataset(Dataset):
    def __init__(self, image_dir, text_dir, transform=None, split="train", train_size=1400):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transform

        # 获取所有图像和文本文件的路径
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".tif")],
            key=lambda x: [int(num) if num.isdigit() else num for num in re.findall(r'\d+|\D+', x)]
        )
        self.text_files = sorted(
            [f for f in os.listdir(text_dir) if f.endswith(".txt")],
            key=lambda x: [int(num) if num.isdigit() else num for num in re.findall(r'\d+|\D+', x)]
        )

        # 确保图像和文本文件一一对应
        assert len(self.image_files) == len(self.text_files), "图像和文本文件数量不匹配"
        for img, txt in zip(self.image_files, self.text_files):
            # 打印文件名以调试
            print(f"Image: {img}, Text: {txt}")
            # 检查文件名前缀是否一致
            assert img.split(".")[0] == txt.split("_description")[0], f"图像和文本文件不匹配: {img} vs {txt}"

        # 随机划分训练集和测试集
        random.seed(42)  # 固定随机种子
        indices = list(range(len(self.image_files)))
        random.shuffle(indices)
        if split == "train":
            self.indices = indices[:train_size]
        else:
            self.indices = indices[train_size:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_files[self.indices[idx]])
        image = Image.open(img_path).convert("RGB")  # 转换为 RGB 格式

        # 加载文本
        txt_path = os.path.join(self.text_dir, self.text_files[self.indices[idx]])
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()


        # 图像预处理
        if self.transform:
            image = self.transform(image)

        return image, text

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # 归一化
])

# 加载数据集
train_dataset = ShanghaiSatDataset(
    image_dir="shanghai_sat",
    text_dir="generated_descriptions",
    transform=transform,
    split="train"
)

test_dataset = ShanghaiSatDataset(
    image_dir="shanghai_sat",
    text_dir="generated_descriptions",
    transform=transform,
    split="test"
)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.float()



# 记录训练和验证损失
train_losses = []
val_losses = []

# 损失函数
def contrastive_loss(logits_per_image, logits_per_text):
    labels = torch.arange(logits_per_image.size(0), device=device)
    loss_image = torch.nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_text = torch.nn.CrossEntropyLoss()(logits_per_text, labels)
    return (loss_image + loss_text) / 2


# 优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
torch.cuda.empty_cache()

# 训练循环
for epoch in range(10):  # 训练 10 个 epoch
    model.train()

    for images, texts in train_dataloader:
        images = images.to(device)
        texts = clip.tokenize(texts, truncate=True).to(device)


        # 前向传播
        logits_per_image, logits_per_text = model(images, texts)

        # 计算损失
        loss = contrastive_loss(logits_per_image, logits_per_text)


        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
        train_losses.append(loss.item())


#保存为 CSV 文件
with open("train_loss.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss"])
    for i, loss in enumerate(train_losses):
        writer.writerow([i + 1, loss])



# 保存模型
torch.save(model.state_dict(), "clip_model_Summary.pth")

# 验证
model.eval()
val_loss = 0.0

with torch.no_grad():
    for images, texts in test_dataloader:
        images = images.to(device)
        texts = clip.tokenize(texts, truncate=True).to(device)


        # 前向传播
        logits_per_image, logits_per_text = model(images, texts)

        # 计算损失
        val_loss += contrastive_loss(logits_per_image, logits_per_text).item()

    val_loss /= len(test_dataloader)
    print(f"Validation Loss: {val_loss:.4f}")