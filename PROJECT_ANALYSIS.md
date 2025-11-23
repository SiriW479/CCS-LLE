# CCS-LLE 项目代码结构分析

## 📋 项目概述
论文：*Low-Light Color Imaging via Cross-Camera Synthesis* (IEEE JSTSP 2022)  
作者：Guo et al.  
框架：Cross-Camera Synthesis (CCS) - 三模块联合训练框架

---

## 🏗️ 核心架构与模块对应

### 1️⃣ **RefIE（Reference-based Illumination Enhancement）**

#### 相关文件：
- **主要实现**：`ref_exposure_combine_clean.py`

#### 核心类：
```python
✅ DecomYUVScaleNetSplit(nn.Module)
   ├─ ScaleYUVBlock：全局亮度尺度估计
   │  • 输入：low_res_Y 通道 + ref_Y（高分辨率单色图的Y）
   │  • 结构：多层 Conv + MaxPool/AvgPool 的层级特征融合
   │  • 输出：全局亮度缩放因子（3通道尺度）
   │  • 关键代码：
   │    - self.conv0 = nn.Conv2d(2, channel, 9, padding=4)  # 输入拼接 [LSR_Y, ref_Y]
   │    - self.maxpool / self.avgpool：金字塔采样
   │    - self.conv3(...) 上采样恢复分辨率
   │
   └─ SingleDecomNetSplit：细节增强
      • 输入：低曝光彩色图 × 全局尺度
      • 结构：5层 BasicBlock（残差块）+ Tanh激活
      • 输出：增强后的YUV图像

✅ ScaleYUVBlock(nn.Module)
   • Patch-wise 特征提取（金字塔结构）
   • 输出维度：(B, 3, H, W)

✅ SingleDecomNetSplit(nn.Module)
   • 残差学习网络
   • 群卷积（groups=2）节省计算量
   • 分解后加强化网络

✅ BasicBlock(nn.Module)
   • 标准残差块
   • 激活函数：LeakyReLU(0.1) 或 ReLU
```

#### 关键技术点：
- ✅ **Patch-wise 特征提取**：通过多层级 MaxPool/AvgPool 实现
- ✅ **参考引导**：直接使用参考图像的Y通道作为输入通道
- ✅ **YUV色彩空间**：分离处理亮度和色度

---

### 2️⃣ **RefAT（Reference-based Appearance Transfer）**

#### 相关文件：
- **主要实现**：`test_flow_sample_refine_res_clean.py`

#### 核心类：
```python
✅ DecomNet_attention(nn.Module)
   • 输入：(B, 7, H, W) = [低曝光彩色图(3ch), 光流变形参考图(3ch), 参考Y(1ch)]
   • 输入构成：
     │ x: LSR 彩色图 (3通道)
     │ warp_x: 用光流对齐后的参考图 (3通道)  ← 光流对齐机制！
     │ ref_y: 参考图的Y通道 (1通道)
   │
   • 结构：
     ├─ conv0: 9×9卷积核（感受野大）+ 64通道
     ├─ conv_l1 ~ conv_l5: 5层 BasicBlock（ReLU）
     ├─ conv1: 3×3卷积输出6通道 (3通道图像 + 3通道掩码)
     └─ Sigmoid激活（概率归一化）
   │
   • 输出：
     └─ 融合图像 (3ch) = 传递图像(mask) + 变形参考图(1-mask)
        └─ 掩码表示融合权重（Appearance Transfer的权重）
   │
   • 特殊输出选项：
     └─ output_mask=True 时返回 (融合图像, 掩码)

✅ BasicBlock(nn.Module) [同RefIE]
```

#### 关键技术点：
- ✅ **光流对齐（PWCNet）**：warp_x 是通过PWCNet光流估计后的变形图像
- ✅ **双向迁移**：
  - 彩色信息迁移：从参考图 → LSR彩色图 (通过 warp_x)
  - 纹理细节迁移：从LSR → 参考图 (反向通过掩码)
- ✅ **注意力机制**：学习掩码表示融合权重
- ✅ **强掩码选项**：`if strong_mask: mask = 1/(1+exp(-10*(mask-0.5)))`

---

### 3️⃣ **RefSR（Reference-based Super-Resolution）**

#### 相关文件：
- **主要实现**：`ref_SR_deshape_clean.py`

#### 核心类：
```python
✅ HDRNetwoBN(nn.Module)  ← 改进的HDRNet架构
   • 输入：
     ├─ low_res_input: 低分辨率彩色图 (B, 3, H_low, W_low)
     └─ full_res_input: 高分辨率引导图 (B, 1, H_high, W_high)【Y通道】
   │
   • 架构（双分支）：
     │
     ├─ 分支1：Splat（特征降采样）
     │  • 4层卷积，逐层步长2下采样
     │  • 输出特征：64通道，1/16分辨率
     │
     ├─ 分支2：Global Branch（全局上下文）
     │  └─ global_brach(nn.Module)
     │     • 多层级平均池化 + 融合
     │     • 输出：64通道全局特征
     │
     ├─ 融合：Fused = Global + Local
     │
     ├─ 双边网格生成：
     │  • self.linear: Conv2d(64, 96) → 12×8 = 96维双边网格
     │  • 形状：(B, 12, 8, H_low//16, W_low//16)
     │  • 这是改进HDRNet的关键！
     │
     └─ 应用阶段：
        ├─ self.guide_func(Guide2)：从高分辨率Y生成引导图
        ├─ self.slice_func(Slice)：采样双边网格系数
        ├─ self.transform_func(Transform)：应用变换矩阵
        ├─ self.adjustChromeU/V(adjustChrome)：色度通道微调
        └─ 输出：高分辨率YUV图像

✅ global_brach(nn.Module)
   • 自适应平均池化（1×1）提取全局特征
   • 多层级 Conv 融合（64→128→256→...）
   • 级联融合回归到64通道

✅ Guide2(nn.Module)
   • PointwiseNN 模式：逐像素神经网络
   • 输入：高分辨率Y通道 (1通道)
   • 输出：引导图 (1通道)
   • 结构：2层卷积 + Tanh激活

✅ Slice(nn.Module)
   • 功能：从双边网格采样系数
   • 使用 grid_sample 进行可微分采样

✅ Transform(nn.Module)
   • 应用双边网格变换
   • 逐通道处理 Y, U, V

✅ adjustChrome(nn.Module)
   • 色度微调网络
   • 输入：单通道 (U或V)
   • 输出：单通道调整值
   • 与插值结果残差相加
```

#### 改进HDRNet的关键点：
- ✅ **96维双边网格**（12×8）：比标准HDRNet更细致
- ✅ **全局分支**：多层级上下文聚合
- ✅ **色度微调**：单独的色度调整网络（adjustChrome）
- ✅ **YUV分离处理**：Y通道引导，UV通道微调

---

## 🔄 光流对齐机制（PWCNet）

### 文件：`PWCNet.py`

#### 两种实现：
```python
✅ PWCDCNet(nn.Module)
   • GPU版本（CUDA相关性层）
   • 使用 Correlation(pad_size=md, kernel_size=1, max_displacement=md)
   • 来源：NVIDIA官方PWC-Net

✅ PWCDCNetCPU(nn.Module)  
   • CPU兼容版本
   • 使用 Correlation_CPU(kernel_size=2*md+1)
   • 自动降级机制（见 test_device.py）

✅ 关键方法：
   • warp()：根据光流对图像进行变形
     ├─ 输入：图像 x (B, C, H, W)、光流 flo (B, 2, H, W)
     ├─ 使用 grid_sample 实现可微分采样
     └─ 返回：变形后的图像与有效掩码
   │
   • forward()：计算光流
     ├─ 多层级特征提取（conv1a-conv6b）
     ├─ 6层相关性层 → 流估计
     ├─ 级联上采样与精化
     └─ 最后使用 7层膨胀卷积（dilation）精化流

📍 在RefAT中的使用：
   • warp_x = warp(reference_image, optical_flow)
   • 与低曝光彩色图拼接后输入 DecomNet_attention
```

#### 关键参数：
```python
md = 4  # 最大位移（最大4像素搜索范围）
相关性维度：(2*4+1)² = 81 维特征向量
```

---

## 💾 损失函数定义（myLoss.py）

### 关键损失类：

```python
✅ YUV_Loss(nn.Module)
   • 分阶段加权：先关注Y，后关注UV
   • if loss_y > threshold_uv:
   •     loss = loss_y + 0.2*loss_u + 0.2*loss_v
   •   else:
   •     loss = loss_y + loss_u + loss_v

✅ Image_smooth_loss(nn.Module)
   • 加权TV（全变分）正则
   • 权重由标签梯度确定：w = exp(-TV_scale * |∇label|)

✅ Flow_smooth_loss(nn.Module)
   • 光流平滑约束（流不应在平坦区域变化）

✅ L_ref_exp(nn.Module)
   • 参考引导的曝光一致性
   • Patch-wise (16×16) 平均值匹配

✅ L_spa(nn.Module)
   • 空间梯度一致性
   • 4方向梯度核（左右上下）
   • 权重：exp(-10000*min(org_pool - 0.3, 0))

✅ L_exp(nn.Module)
   • 全局曝光约束
   • 使目标亮度接近设定的 mean_val

✅ color_space_loss(nn.Module)
   • 颜色空间一致性
   • 检测颜色反转（sign(-predict*label)）

✅ PSNR_loss(nn.Module)
   • MSE 基础，可选择 Y/UV/YUV 通道
```

---

## 📊 数据加载与处理（loadDataset.py）

### 数据集类：

```python
✅ myBilateralDataset(Dataset)
   • 输入：(LSR彩色, HSR单色, HSR单色真值, HSR彩色真值)
   • 处理：
     ├─ 随机曝光调整（0.5-1.5x）
     ├─ 单色图转灰度（加权RGB）
     ├─ 色彩+单色变换（增强、裁剪等）
     └─ 输出：{'mono', 'color', 'label', 'label_color'}

✅ myEnhanceBilateralDataset(Dataset)
   • 训练增强版：
     ├─ 随机曝光(0.9-1.2x)
     ├─ 随机色调调整(0.5-1.5x)
     └─ 噪声添加选项

✅ myTestBilateralDataset(Dataset)
   • 测试版本：固定变换，支持旋转增强

✅ myTestOverallBilateralDataset(Dataset)
   • 整体管道测试
   • 输入映射：(LSR, HSR_mono, HSR_color)

✅ RealCaptureDataset(Dataset)
   • 真实捕获数据：_L.png (LSR) 和 _R.png (HSR单色)

✅ RealCaptureGopDataset(Dataset)
   • 视频序列支持（Group of Pictures）
   • GopSize 参数控制帧间隔
```

---

## 🎯 测试/推理管道（test-3ref-clean.py）

### 主函数：
```python
test_new_bilateral_simulate(args)
├─ 导入三个模块：
│  ├─ DecomYUVScaleNetSplit (RefIE)
│  ├─ DecomNet_attention (RefAT)
│  └─ HDRNetwoBN (RefSR)
│
├─ GPU自动检测与降级
│  └─ is_gpu_supported_by_pytorch13() 检查计算能力
│
├─ 依次应用三个模块：
│  1. RefIE：LSR → 增强亮度
│  2. RefAT + PWCNet：光流对齐 + 外观迁移
│  3. RefSR：色度超分辨率
│
└─ 输出：HSR彩色图像
```

---

## 🏋️ 关键技术栈

| 技术 | 文件 | 说明 |
|------|------|------|
| **光流估计** | `PWCNet.py` | PWC-DC网络（金字塔相关性）|
| **Patch特征** | `ref_exposure_combine_clean.py` | MaxPool/AvgPool金字塔 |
| **注意力融合** | `test_flow_sample_refine_res_clean.py` | 学习掩码的加权融合 |
| **HDRNet改进** | `ref_SR_deshape_clean.py` | 双边网格+色度微调 |
| **YUV处理** | `image_utils.py` | rgb2yuv / yuv2rgb 转换 |
| **损失函数** | `myLoss.py` | 分阶段、加权、参考引导 |
| **数据增强** | `loadDataset.py` | 随机曝光、噪声、旋转 |

---

## 🔗 模块间数据流

```
输入数据：
  ├─ LSR彩色图 (480×640×3)
  ├─ HSR单色图 (960×1280×1)
  └─ HSR单色真值 (960×1280×1)

         ↓
    [RefIE]
  增强LSR亮度 YUV空间
         ↓ 增强图 (480×640×3_YUV)
         
         ↓
    [PWCNet]
  计算光流：HSR_mono → LSR_mono
         ↓ 光流 (480×640×2)
         
         ↓
  [Warp] 
  变形HSR单色为LSR分辨率
         ↓ 变形图 (480×640×3)
         
         ↓
    [RefAT]
  融合：[增强图, 变形图, LSR_mono] → 7通道输入
         ↓ 融合图 (480×640×3)
         
         ↓
    [RefSR]
  超分辨率：用HSR_mono的Y引导 U/V 上采样
         ↓ 最终输出 (960×1280×3_HSR)
```

---

## 📌 训练策略推测

根据代码结构，推测为 **分阶段训练** + **联合微调**：

```
阶段1：RefIE 单独训练
  • 损失：YUV_Loss + Image_smooth_loss + L_ref_exp

阶段2：RefAT 单独训练
  • 损失：L1Loss + Flow_smooth_loss + 注意力正则

阶段3：RefSR 单独训练
  • 损失：CharbonnierLoss + ...

阶段4：端到端联合微调
  • 损失组合：α₁*L_RefIE + α₂*L_RefAT + α₃*L_RefSR
  • 共同目标：最小化最终输出与真值的差异
```

---

## 🐛 已知问题与改进

### model_init.py 中的 bug：
```python
# ❌ 错误代码
print(module.__class__)  # module 未定义

# ✅ 应为
print(m.__class__)  # 正确的模块变量
```

---

## 📚 参考文献
- PWC-Net: Sun et al., "PWC-Net: End-to-End Learning of Optical Flow"
- HDRNet: Gharbi et al., "Deep Bilateral Learning for Real-Time Image Enhancement"
- 论文：Guo et al., "Low-light Color Imaging via Cross-Camera Synthesis", IEEE JSTSP 2022

---

## ✅ 总结

| 模块 | 文件 | 关键类 | 关键技术 |
|------|------|--------|---------|
| **RefIE** | ref_exposure_combine_clean.py | DecomYUVScaleNetSplit | Patch金字塔、YUV分解 |
| **RefAT** | test_flow_sample_refine_res_clean.py | DecomNet_attention | 光流对齐、注意力掩码 |
| **RefSR** | ref_SR_deshape_clean.py | HDRNetwoBN | 双边网格、色度微调 |
| **Flow** | PWCNet.py | PWCDCNet/PWCDCNetCPU | 金字塔相关性、变形采样 |
