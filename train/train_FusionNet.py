import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_msssim  # 用于 SSIM 计算
from tensorboardX import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt  # 添加matplotlib用于可视化

# 从项目文件中导入必要的模块
from test_flow_sample_refine_res_clean import DecomNet_attention  # 融合网络
from loadDataset import myTestEnhanceDataset  # 数据集
from myLoss import YUV_Loss, PSNR_loss  # 损失函数
import PWCNet  # 用于光流对齐（如果需要）

# --- 1. 配置训练参数 ---
learning_rate = 1e-4
batch_size = 4
epochs = 100
dataset_txt_path = 'test_pair_list.txt'  # 数据集文件路径，每行：LSR彩色图 HSR单色图 真值图
checkpoint_dir = '../ckpt'
save_every_epoch = 10
log_dir = './logs_fusion'  # TensorBoard 日志目录

# --- 2. 初始化 ---
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
model = DecomNet_attention(layer_num=5, channel=64, kernel_size=3).to(device)

# 初始化光流模型（用于对齐，如果训练数据需要）
flow_model = PWCNet.PWCDCNet().to(device)
flow_model.eval()  # 推理模式，不训练

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 损失函数
yuv_loss_fn = YUV_Loss()
psnr_loss_fn = PSNR_loss()

# TensorBoard 记录器
writer = SummaryWriter(log_dir)

# 数据加载
train_dataset = myTestEnhanceDataset(dataset_txt_path, transform=None)  # 假设数据集返回 LSR, HSR, GT
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 初始化列表用于收集指标
loss_history = []
psnr_history = []
ssim_history = []

# --- 3. 训练循环 ---
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_psnr = 0.0
    epoch_ssim = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, sample in enumerate(train_loader):
        # 假设 sample 返回：lsr_rgb, hsr_mono, gt_rgb
        lsr_rgb, hsr_mono, gt_rgb = sample['lsr'], sample['hsr'], sample['gt']
        lsr_rgb = lsr_rgb.to(device)
        hsr_mono = hsr_mono.to(device)
        gt_rgb = gt_rgb.to(device)
        
        # 数据预处理：RGB -> YUV
        from image_utils import rgb2yuv
        lsr_yuv = rgb2yuv(lsr_rgb)
        hsr_yuv = rgb2yuv(hsr_mono)  # 假设 HSR 是 RGB，如果是单色需调整
        gt_yuv = rgb2yuv(gt_rgb)
        
        # 光流对齐（简化：假设输入已对齐，或用 flow_model 计算）
        # 这里简化，假设 warp_x = hsr_yuv（实际需用光流 warp）
        warp_x = hsr_yuv  # 替换为实际 warp 函数
        ref_y = hsr_yuv[:, 0:1, :, :]  # HSR 的 Y 通道
        
        # 前向传播
        output = model(lsr_yuv, warp_x, ref_y)
        
        # 计算损失
        loss_yuv = yuv_loss_fn(output, gt_yuv)
        loss_psnr = psnr_loss_fn(output, gt_yuv)
        loss_ssim = 1 - torch_msssim.ssim(output, gt_yuv, data_range=1.0)  # SSIM 损失（1 - SSIM）
        total_loss = loss_yuv + loss_psnr + loss_ssim  # 组合损失
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 累积指标
        epoch_loss += total_loss.item()
        epoch_psnr += loss_psnr.item()
        epoch_ssim += loss_ssim.item()
        
        # 打印批次信息
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {total_loss.item():.4f}, PSNR: {loss_psnr.item():.4f}, SSIM: {loss_ssim.item():.4f}")
    
    # 平均指标
    avg_loss = epoch_loss / num_batches
    avg_psnr = epoch_psnr / num_batches
    avg_ssim = epoch_ssim / num_batches
    
    # 收集到历史列表
    loss_history.append(avg_loss)
    psnr_history.append(avg_psnr)
    ssim_history.append(avg_ssim)
    
    # 记录到 TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('PSNR/train', avg_psnr, epoch)
    writer.add_scalar('SSIM/train', avg_ssim, epoch)
    
    print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")
    
    # 保存模型
    if (epoch + 1) % save_every_epoch == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'fusion_model_epoch_{epoch+1}.pth'))
        print(f"Model saved at epoch {epoch+1}")

writer.close()

# --- 4. 可视化曲线 ---
epochs_list = list(range(1, epochs + 1))

plt.figure(figsize=(15, 5))

# LOSS 曲线
plt.subplot(1, 3, 1)
plt.plot(epochs_list, loss_history, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# PSNR 曲线
plt.subplot(1, 3, 2)
plt.plot(epochs_list, psnr_history, label='PSNR', color='orange')
plt.title('Training PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

# SSIM 曲线
plt.subplot(1, 3, 3)
plt.plot(epochs_list, ssim_history, label='SSIM', color='green')
plt.title('Training SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'training_curves.png'))
plt.show()

print("Training completed! Curves saved to training_curves.png")