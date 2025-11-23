
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

# 从项目文件中导入必要的模块
from ref_exposure_combine_clean import DecomYUVScaleNetSplit, yuv2rgb
from loadDataset import myTestEnhanceDataset
from myLoss import YUV_Loss, PSNR_loss
import numpy as np

# --- 1. 配置训练参数 ---
# 你可以根据需要修改这些超参数
learning_rate = 1e-4
batch_size = 4
epochs = 100
# 数据集文件路径，其中每一行包含：低光彩色图路径 高光单色图路径 真值图路径
dataset_txt_path = 'test_pair_list.txt' 
# 模型权重保存路径
checkpoint_dir = './ckpt'
# 检查点保存频率
save_every_epoch = 10

# --- 2. 初始化 ---
# 检查并创建权重保存目录
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 设置设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 3. 数据加载 ---
# 定义数据预处理（这里仅转换为Tensor，因为数据加载器中已处理）
transform = transforms.ToTensor()

# 创建数据集实例
# 注意：myTestEnhanceDataset 内部会将RGB转为YUV，并处理好输入
# 它返回一个字典，包含 'color' (LSR YUV), 'mono' (HSR Y), 'label' (真值 YUV)
train_dataset = myTestEnhanceDataset(
    root_txt=dataset_txt_path,
    line_index=-1 # -1 表示加载所有行
)

# 创建数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4, # 根据你的机器配置调整
    pin_memory=True
)

print(f"Dataset loaded with {len(train_dataset)} samples.")

# --- 4. 模型、损失函数和优化器 ---
# 初始化模型
model = DecomYUVScaleNetSplit().to(device)
model.train() # 设置为训练模式

# 定义损失函数
# YUV_Loss 会在Y通道损失较大时，侧重优化Y通道
criterion = YUV_Loss(threshold_uv=0.01).to(device)
# 也可以使用更简单的L1损失
# criterion = torch.nn.L1Loss().to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Model, Loss, and Optimizer initialized.")

# --- 5. 训练循环 ---
print("Starting training...")
for epoch in range(epochs):
    total_loss = 0.0
    
    for i, batch in enumerate(train_loader):
        # 获取数据并移到设备
        # 数据加载器已经将RGB转为YUV
        color_yuv = batch['color'].to(device)
        mono_y = batch['mono'].to(device)
        label_yuv = batch['label'].to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        # 模型输入: (B,3,H,W)的YUV彩色图, (B,1,H,W)的单色Y参考图
        # 模型输出: 最终增强的YUV图, 全局尺度因子
        enhanced_yuv, _ = model(color_yuv, mono_y)

        # 计算损失
        loss = criterion(enhanced_yuv, label_yuv)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f'--- Epoch [{epoch+1}/{epochs}] Finished, Average Loss: {avg_loss:.4f} ---')

    # --- 6. 保存模型权重 ---
    if (epoch + 1) % save_every_epoch == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'RefIE_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

print("Training finished.")
final_model_path = os.path.join(checkpoint_dir, 'RefIE_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

