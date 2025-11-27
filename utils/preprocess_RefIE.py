import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 从项目文件中导入必要的模块
from ref_exposure_combine_clean import DecomYUVScaleNetSplit, yuv2rgb
from loadDataset import myTestEnhanceDataset
from image_utils import rgb2yuv  # 这是类，需要实例化

# --- 1. 配置参数 ---
# RefIE 模型权重路径（.pth 文件）
refie_model_path = './ckpt/RefIE_final.pth'  # 修改为你的实际路径
# 数据集文件路径（每行：低光彩色图 高光单色图 真值图）
dataset_txt_path = 'test_pair_list.txt'
# 输出目录：保存增强后的图像
output_dir = './preprocessed_data'
# 批次大小
batch_size = 4
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 初始化 ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载 RefIE 模型
model = DecomYUVScaleNetSplit().to(device)
model.load_state_dict(torch.load(refie_model_path, map_location=device))
model.eval()  # 设置为推理模式
print(f"Loaded RefIE model from {refie_model_path}")

# 数据加载
dataset = myTestEnhanceDataset(dataset_txt_path, line_index=-1)  # 加载所有样本
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 初始化 RGB -> YUV 转换器
rgb2yuv_transform = rgb2yuv()

print(f"Dataset loaded with {len(dataset)} samples. Starting preprocessing...")

# --- 3. 处理循环 ---
with torch.no_grad():  # 不计算梯度
    for batch_idx, sample in enumerate(dataloader):
        # 获取数据（假设返回 RGB 格式）
        lsr_rgb = sample['color'].to(device)  # 低光彩色图 (B, 3, H, W)
        hsr_mono = sample['mono'].to(device)  # 高光单色图 (B, 1, H, W)
        
        # 转换为 YUV（RefIE 输入格式）
        lsr_yuv = rgb2yuv_transform(lsr_rgb)  # 使用实例化的转换器
        # hsr_mono 已经是单色 Y，直接用作 ref_y
        
        # 运行 RefIE 模型
        enhanced_yuv, _ = model(lsr_yuv, hsr_mono)
        
        # 转换回 RGB 保存
        enhanced_rgb = yuv2rgb(enhanced_yuv)
        
        # 保存每个批次的图像
        for i in range(enhanced_rgb.size(0)):
            img = enhanced_rgb[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)  # 归一化到 0-255
            pil_img = Image.fromarray(img)
            
            # 文件名：batch_idx * batch_size + i
            img_name = f"enhanced_{batch_idx * batch_size + i:04d}.png"
            pil_img.save(os.path.join(output_dir, img_name))
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

print(f"Preprocessing completed! Enhanced images saved to {output_dir}")

# --- 4. 可选：生成新的数据集文件 ---
# 如果需要，可以生成一个新的 txt 文件，指向增强图像
new_dataset_txt = os.path.join(output_dir, 'enhanced_dataset.txt')
with open(new_dataset_txt, 'w') as f:
    for i in range(len(dataset)):
        enhanced_path = f"./preprocessed_data/enhanced_{i:04d}.png"
        # 假设原始数据集有对应的 HSR 和 GT，可以从原始 txt 复制
        # 这里简化，假设只用增强图像作为输入
        f.write(f"{enhanced_path}\n")  # 根据需要调整格式

print(f"New dataset file created: {new_dataset_txt}")
