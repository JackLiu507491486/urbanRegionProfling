import os

# 定义文件夹路径
image_folder = "shanghai_sat"  # 图片文件夹
text_folder = "generated_descriptions"  # 文本文件夹

# 获取图片文件名（去除扩展名）
image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(".tif"))

# 获取文本文件名（去除扩展名）
text_files = set(os.path.splitext(f)[0].replace("_description", "") for f in os.listdir(text_folder) if f.endswith(".txt"))

# 检查是否一一对应
missing_images = text_files - image_files  # 在文本文件夹中存在但图片文件夹中缺失的文件
missing_texts = image_files - text_files  # 在图片文件夹中存在但文本文件夹中缺失的文件

# 输出结果
if not missing_images and not missing_texts:
    print("✅ 两个文件夹中的文件一一对应。")
else:
    if missing_images:
        print("❌ 以下文件在图片文件夹中缺失：")
        for file in missing_images:
            print(f"  - {file}.tif")
    if missing_texts:
        print("❌ 以下文件在文本文件夹中缺失：")
        for file in missing_texts:
            print(f"  - {file}_description.txt")