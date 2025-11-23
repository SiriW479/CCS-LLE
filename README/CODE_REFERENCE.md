# CCS-LLE 代码映射速查表

## 📌 按功能快速查找

### 🔵 亮度增强 (Illumination Enhancement) - RefIE

```python
# 文件: ref_exposure_combine_clean.py

【全局亮度尺度估计】
class ScaleYUVBlock(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        # 核心参数
        self.conv0 = nn.Conv2d(2, channel, kernel_size*3, padding=4)  # 9×9核
        self.maxpool = nn.MaxPool2d(kernel_size*3, stride=4, padding=4)
        self.avgpool = nn.AvgPool2d(kernel_size*3, stride=4, padding=4)
        
        # 多层级融合
        self.conv1 = nn.Conv2d(channel*2, channel, kernel_size*3, padding=4)
        self.conv2 = nn.Conv2d(channel*2, channel, kernel_size*3, padding=4)
        self.conv3 = nn.Conv2d(channel, 3, 1)  # 输出3通道尺度
    
    def forward(self, x, ref_y):
        # x: [Y_LSR, ref_Y] 拼接 (B, 2, H, W)
        # 输出: 全局尺度 (B, 3, H, W) ✅
        
【细节增强网络】
class SingleDecomNetSplit(nn.Module):
    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        self.conv0 = nn.Conv2d(3, channel, kernel_size*3, padding=4)
        
        # 5层特征融合
        feature_conv = []
        for idx in range(layer_num):
            feature_conv.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size, padding=1, groups=2),
                nn.ReLU()
            ))
        self.conv = nn.ModuleList(feature_conv)
        
        self.conv1 = nn.Conv2d(channel, 3, kernel_size, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # 残差学习
        residual = x
        out = self.conv0(x)
        for idx in range(self.layer_num):
            out = self.conv[idx](out)
        out = self.conv1(out)
        out = self.tanh(out)
        return out + residual  # 残差连接

【完整管道】
class DecomYUVScaleNetSplit(nn.Module):
    def forward(self, x, ref_y, limit=False):
        # 步骤1: 全局尺度
        x_y = x[:, 0, :, :].unsqueeze(1)  # 提取Y通道
        global_scale = self.global_scale(x_y, ref_y)
        
        # 步骤2: 使用尺度缩放
        refine_x = x * global_scale
        if limit:
            refine_x[:, 0, :, :].clamp_(min=0, max=1)
            refine_x[:, 1:, :, :].clamp_(min=-0.5, max=0.5)
        
        # 步骤3: 细节增强
        final_output = self.enhancement(refine_x)
        return final_output, global_scale
```

**关键特性**:
- ✅ Patch-wise 金字塔 (9×9 → 3×3 stride 4)
- ✅ MaxPool + AvgPool 双路特征
- ✅ 3通道独立尺度 (Y/U/V)
- ✅ 残差学习避免过度处理

---

### 🟠 外观迁移 (Appearance Transfer) - RefAT

```python
# 文件: test_flow_sample_refine_res_clean.py

【光流引导的融合网络】
class DecomNet_attention(nn.Module):
    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        self.conv0 = nn.Conv2d(7, channel, kernel_size*3, padding=4)  # 9×9核
        
        # 5层注意力特征
        self.conv_l1 = BasicBlock(channel, channel, activation=nn.ReLU(inplace=True))
        self.conv_l2 = BasicBlock(channel, channel, activation=nn.ReLU(inplace=True))
        self.conv_l3 = BasicBlock(channel, channel, activation=nn.ReLU(inplace=True))
        self.conv_l4 = BasicBlock(channel, channel, activation=nn.ReLU(inplace=True))
        self.conv_l5 = BasicBlock(channel, channel, activation=nn.ReLU(inplace=True))
        
        # 输出6通道: 3通道图像 + 3通道掩码
        self.conv1 = nn.Conv2d(channel, 6, kernel_size, padding=1)
        self.sig = nn.Sigmoid()
    
    def forward(self, x, warp_x, ref_y, strong_mask=False, output_mask=False):
        # 输入
        # x: LSR彩色 (3ch)
        # warp_x: 光流对齐的参考 (3ch)
        # ref_y: 参考单色 (1ch)
        x = torch.cat((x, warp_x, ref_y), dim=1)  # 7ch拼接
        
        # 特征提取
        out = self.conv0(x)
        out = self.conv_l1(out)
        out = self.conv_l2(out)
        out = self.conv_l3(out)
        out = self.conv_l4(out)
        out = self.conv_l5(out)
        
        # 输出
        out = self.conv1(out)  # (B, 6, H, W)
        out = self.sig(out)  # Sigmoid 归一化
        
        # 分离图像和掩码
        img2 = out.clone()[:, 0:3, :, :]  # 融合后图像
        mask = out.clone()[:, 3:, :, :]   # 融合掩码
        
        # 调整UV到中心 (YUV中 U,V ∈ [-0.5, 0.5])
        img2[:, 1, :, :] -= 0.5
        img2[:, 2, :, :] -= 0.5
        
        # 可选: 增强掩码对比度
        if strong_mask:
            mask = 1 / (1 + torch.exp(-10 * (mask - 0.5)))
        
        # 融合: 掩码加权组合
        out_refine = img2 * mask + warp_x * (1 - mask)
        
        if output_mask:
            return out_refine, mask
        else:
            return out_refine

【光流计算】
# 文件: PWCNet.py
class PWCDCNet(nn.Module):
    def forward(self, x):  # x: [image1(3ch)|image2(3ch)] = 6ch
        C_channel = x.shape[1] // 2
        im1 = x[:, :C_channel, :, :]
        im2 = x[:, C_channel:, :, :]
        
        # 6层特征金字塔
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))  # 1/2
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        ...
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))  # 1/64
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))
        
        # 从粗到细估计光流
        corr6 = self.corr(c16, c26)  # 相关性体积 (B, 81, H/64, W/64)
        
        # 6层解码
        flow6 = ... # 估计流
        up_flow6 = self.deconv6(flow6)
        
        # 递归精化到第2层
        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        ...
        flow5 = ...
        
        # ... (第4,3层)
        
        # 最终第2层
        corr2 = self.corr(c12, warp2)
        flow2 = ...
        
        # 7层膨胀卷积精化
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        
        return flow2  # (B, 2, H/4, W/4)

【图像变形】
def warp(x, flo):
    """根据光流变形图像"""
    B, C, H, W = x.size()
    
    # 创建坐标网格
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)  # 列坐标
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)  # 行坐标
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy), 1).float()
    
    # 加上光流
    vgrid = grid + flo  # (B, 2, H, W)
    
    # 标准化到 [-1, 1]
    vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone()/max(W-1,1) - 1.0
    vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone()/max(H-1,1) - 1.0
    
    # 双线性插值采样
    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid.clone())
    
    # 计算有效掩码（处理边界）
    mask = torch.ones(x.size()).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid.clone())
    
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    
    return output * mask, mask

【完整数据流】
def test_new_bilateral_simulate(args):
    # 加载模型
    from ref_exposure_combine_clean import DecomYUVScaleNetSplit
    from test_flow_sample_refine_res_clean import DecomNet_attention
    
    # RefIE: 亮度增强
    color_enhanced, scale = ref_ie_net(color_image, mono_image)
    
    # RefAT: 外观迁移
    # 步骤1: 光流计算
    flow = pwc_net(torch.cat([mono_ref_expanded, mono_lsr], dim=1))
    
    # 步骤2: 图像变形
    warped_ref, mask = warp(color_ref, flow)
    
    # 步骤3: 融合
    color_transfer = ref_at_net(color_enhanced, warped_ref, mono_lsr)
    
    # RefSR: 色度超分
    final_output = ref_sr_net(color_transfer, mono_hires)
    
    return final_output
```

**关键点**:
- ✅ 7通道输入: [LSR_RGB(3) + warp_ref(3) + LSR_mono(1)]
- ✅ 掩码融合权重学习
- ✅ PWCNet金字塔光流 (6层)
- ✅ grid_sample 可微分采样

---

### 🟣 色度超分 (Super-Resolution) - RefSR

```python
# 文件: ref_SR_deshape_clean.py

【改进HDRNet架构】
class HDRNetwoBN(nn.Module):
    def __init__(self, inc=3, outc=3):
        # 特征提取 (Splat分支)
        splat_layers = []
        for i in range(4):
            if i == 0:
                splat_layers.append(
                    conv_block(inc, 8, kernel_size=3, stride=2)
                )
            else:
                splat_layers.append(
                    conv_block(8*(2**(i-1)), 8*(2**i), kernel_size=3, stride=2)
                )
        self.splat_conv = nn.Sequential(*splat_layers)  # 输出64ch, 1/16分辨率
        
        # 全局上下文分支
        self.global_brach = global_brach(64, 64, BN=False)
        
        # 本地特征
        local_layers = [
            conv_block(64, 64, activation=self.activation, is_BN=False),
            conv_block(64, 64, use_bias=False, activation=None, is_BN=False),
        ]
        self.local_conv = nn.Sequential(*local_layers)
        
        # 双边网格生成
        self.linear = nn.Conv2d(64, 96, kernel_size=1)  # 关键！96维
        
        # 应用阶段
        self.guide_func = Guide2()          # 生成引导图
        self.slice_func = Slice()           # 采样系数
        self.transform_func = Transform()   # 应用变换
        self.adjustChromeU = adjustChrome() # U通道微调
        self.adjustChromeV = adjustChrome() # V通道微调
    
    def forward(self, low_res_input, full_res_input):
        bs, _, _, _ = low_res_input.size()
        _, _, hh, hw = full_res_input.size()
        
        # 步骤1: 特征提取
        splat_fea = self.splat_conv(low_res_input)  # (B, 64, H/16, W/16)
        
        # 步骤2: 本地特征
        local_fea = self.local_conv(splat_fea)
        
        # 步骤3: 全局特征
        global_fea = self.global_brach(splat_fea)   # (B, 64, 1, 1)
        
        # 步骤4: 融合
        fused = self.activation(
            global_fea.view(-1, 64, 1, 1) + local_fea
        )
        fused = self.linear(fused)  # (B, 96, H/16, W/16)
        
        # 步骤5: 双边网格生成
        f_n, f_c, f_h, f_w = fused.size()
        bilateral_grid = fused.view(-1, 12, 8, f_h, f_w)  # 12×8 = 96
        
        # 步骤6: 引导图生成
        guidemap = self.guide_func(full_res_input)  # 高分辨率Y
        
        # 步骤7: 系数采样
        coeff = self.slice_func(bilateral_grid, guidemap)  # (B, 12, H, W)
        
        # 步骤8: 变换应用
        bufferYUV = self.transform_func(coeff, full_res_input)
        
        # 步骤9: 色度微调
        fake_res_input = f.interpolate(
            low_res_input, size=(hh, hw), mode='bilinear'
        )
        U = self.adjustChromeU(bufferYUV[:, 1, :, :].unsqueeze(1)) + \
            fake_res_input[:, 1, :, :].unsqueeze(1)
        V = self.adjustChromeV(bufferYUV[:, 2, :, :].unsqueeze(1)) + \
            fake_res_input[:, 2, :, :].unsqueeze(1)
        
        # 步骤10: 输出
        output = torch.cat([bufferYUV[:, 0, :, :].unsqueeze(1), U, V], dim=1)
        return output

【全局分支】
class global_brach(nn.Module):
    def __init__(self, inc=64, outc=64, BN=True):
        self.average_0 = nn.AdaptiveAvgPool2d((1,1))
        self.conv_1 = conv_block(inc, 2*inc, kernel_size=3, padding=1, stride=2)
        self.average_1 = nn.AdaptiveAvgPool2d((1,1))
        self.conv_2 = conv_block(2*inc, 4*inc, kernel_size=3, padding=1, stride=2)
        self.average_2 = nn.AdaptiveAvgPool2d((1,1))
        
        # 融合
        self.fuse_1 = conv_block(7*inc, 4*inc, kernel_size=1, padding=0)
        self.fuse_2 = conv_block(4*inc, 2*inc, kernel_size=1, padding=0)
        self.fuse_3 = conv_block(2*inc, 1*inc, kernel_size=1, padding=0)
    
    def forward(self, x):
        # 多层级特征提取
        a0 = self.average_0(x)  # (B, 64, 1, 1)
        
        x = self.conv_1(x)
        a1 = self.average_1(x)  # (B, 128, 1, 1)
        
        x = self.conv_2(x)
        a2 = self.average_2(x)  # (B, 256, 1, 1)
        
        # 拼接: (B, 448, 1, 1) = 64+128+256
        a = torch.cat((a0, a1, a2), dim=1)
        
        # 级联融合
        a = self.fuse_1(a)  # 448 → 256
        a = self.fuse_2(a)  # 256 → 128
        a = self.fuse_3(a)  # 128 → 64
        
        return a  # (B, 64, 1, 1)

【引导图生成】
class Guide2(nn.Module):
    def __init__(self, mode="PointwiseNN"):
        self.mode = "PointwiseNN"
        self.conv1 = conv_block(1, 16, kernel_size=3, stride=1, is_BN=False)
        self.conv2 = conv_block(16, 1, kernel_size=1, padding=0, 
                               activation=nn.Tanh())
    
    def forward(self, x):
        # 输入: 高分辨率Y通道
        guidemap = self.conv2(self.conv1(x))
        return guidemap

【系数采样】
class Slice(nn.Module):
    def forward(self, bilateral_grid, guidemap):
        # bilateral_grid: (B, 12, 8, H/16, W/16)
        # guidemap: (B, 1, H, W) 高分辨率
        
        N, _, H, W = guidemap.shape
        
        # 创建归一化坐标网格
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(N,1,1,1)
        yy = yy.view(1,1,H,W).repeat(N,1,1,1)
        xx = 2.0*xx/max(W-1,1) - 1.0
        yy = 2.0*yy/max(H-1,1) - 1.0
        grid = torch.cat((xx,yy), 1).float()
        
        # 拼接坐标和引导图
        guidemap_guide = torch.cat([grid, guidemap], dim=1)  # (B, 3, H, W)
        guidemap_guide = guidemap_guide.permute(0,2,3,1).contiguous()
        guidemap_guide = guidemap_guide.unsqueeze(1)  # (B, 1, H, W, 3)
        
        # 采样
        coeff = f.grid_sample(bilateral_grid, guidemap_guide)
        
        return coeff.squeeze(2)  # (B, 12, H, W)

【变换应用】
class Transform(nn.Module):
    def forward(self, coeff, full_res_input):
        # coeff: (B, 12, H, W)
        # full_res_input: (B, 1, H, W) 高分辨率Y
        
        Y = full_res_input * coeff[:, 3:4, :, :] + \
            torch.sum(coeff[:, 0:3, :, :], dim=1, keepdim=True)
        
        U = full_res_input * coeff[:, 7:8, :, :] + \
            torch.sum(coeff[:, 4:7, :, :], dim=1, keepdim=True)
        
        V = full_res_input * coeff[:, 11:12, :, :] + \
            torch.sum(coeff[:, 8:11, :, :], dim=1, keepdim=True)
        
        return torch.cat([Y, U, V], dim=1)

【色度微调】
class adjustChrome(nn.Module):
    def __init__(self):
        self.conv1 = conv_block(1, 16, kernel_size=1, padding=0, is_BN=False)
        self.conv2 = conv_block(16, 1, kernel_size=1, padding=0, 
                               activation=nn.Tanh())
    
    def forward(self, chromeInfo):
        chromemap = self.conv1(chromeInfo)
        chromemap = self.conv2(chromemap)  # 微调值
        return chromemap
```

**关键亮点**:
- ✅ 96维双边网格 (12×8)
- ✅ 全局分支多层级聚合
- ✅ Guide2 逐像素引导
- ✅ 变换应用到Y/U/V分别
- ✅ 色度独立微调网络

---

## 🎯 损失函数速查

```python
# 文件: myLoss.py

【分阶段YUV损失】
class YUV_Loss(nn.Module):
    def forward(self, predict, label):
        loss_y = self.loss(predict[:, 0:1], label[:, 0:1])
        loss_u = self.loss(predict[:, 1:2], label[:, 1:2])
        loss_v = self.loss(predict[:, 2:3], label[:, 2:3])
        
        if loss_y.item() > self.threshold_uv:
            # 前期: 主要关注Y
            total_loss = loss_y + 0.2*loss_u + 0.2*loss_v
        else:
            # 后期: 均衡关注
            total_loss = loss_y + loss_u + loss_v
        
        return torch.mean(total_loss)

【平滑约束】
class Image_smooth_loss(nn.Module):
    def forward(self, predicted, label):
        # 加权TV: w = exp(-λ|∇label|)
        predicted_grad_x, predicted_grad_y = self.gradients(predicted)
        label_grad_x, label_grad_y = self.gradients(label)
        
        w_x = torch.exp(-self.TV_scale * torch.abs(label_grad_x))
        w_y = torch.exp(-self.TV_scale * torch.abs(label_grad_y))
        
        error = ((w_x*torch.abs(predicted_grad_x)).mean() +
                 (w_y*torch.abs(predicted_grad_y)).mean())
        return error

【参考曝光一致性】
class L_ref_exp(nn.Module):
    def forward(self, x_y, ref_y):
        # Patch-wise平均值匹配
        x_mean = self.pool(x_y)      # 16×16 pool
        ref_mean = self.pool(ref_y)
        
        d = torch.mean(torch.abs(torch.pow(x_mean - ref_mean, self.lossN)))
        return d
```

---

## 📊 数据加载速查

```python
# 文件: loadDataset.py

【标准训练集】
class myBilateralDataset(Dataset):
    def __getitem__(self, idx):
        # 加载路径
        color = Image.open(self.left_pic_list[idx])      # LSR彩色
        mono = Image.open(self.right_pic_list[idx])      # LSR单色
        label = Image.open(self.left_label_pic_list[idx]) # HSR单色真值
        label_color = Image.open(self.right_label_pic_list[idx]) # HSR彩色真值
        
        # 转numpy并归一化到[0,1]
        color_image = np.array(color).transpose(2,0,1) / 255.0
        mono_source = np.array(mono).transpose(2,0,1) / 255.0
        
        # 转Tensor
        colorTensor = torch.from_numpy(color_image)
        monoTensor_ = torch.from_numpy(mono_source)
        
        # 单色转灰度 (加权平均)
        h, w = monoTensor_.shape[-2:]
        monoTensor = torch.zeros(1, h, w)
        monoTensor[0,:,:] = (monoTensor_[2,:,:] * 0.114 + 
                             monoTensor_[1,:,:] * 0.587 + 
                             monoTensor_[0,:,:] * 0.299)  # RGB权重
        
        return {
            'mono': monoTensor,
            'color': colorTensor,
            'label': mono_labelTensor,
            'label_color': color_labelTensor
        }

【增强训练集】
class myEnhanceBilateralDataset(Dataset):
    def __getitem__(self, idx):
        # 随机曝光调整
        color_adjust = np.random.uniform(0.5, 1.5)
        color_image *= color_adjust
        
        mono_adjust = np.random.uniform(0.9, 1.2)
        mono_source *= mono_adjust
        
        # ... (同上)
```

---

## 🔧 配置常数

```python
# RefIE 参数
ScaleYUVBlock:
    conv0_kernel = 9        # 大感受野
    pool_kernel = 9
    pool_stride = 4         # 下采样因子
    channel = 64            # 隐藏通道

# RefAT 参数
DecomNet_attention:
    conv0_kernel = 9
    layer_num = 5           # 5层BasicBlock
    channel = 64
    output_channel = 6      # 3图像 + 3掩码

# RefSR 参数
HDRNetwoBN:
    splat_layers = 4        # 1/16最终分辨率
    bilateral_grid = 12×8   # 96维
    global_branch = enabled # 关键改进

# PWCNet 参数
PWCDCNet:
    md = 4                  # 最大位移(像素)
    pyramid_levels = 6      # 6层金字塔
    corr_dim = 81           # (2*4+1)²
```

---

## ✅ 推理步骤检查清单

```
[ ] 1. 加载预训练模型
      ├─ RefIE: DecomYUVScaleNetSplit()
      ├─ RefAT: DecomNet_attention()
      ├─ RefSR: HDRNetwoBN()
      └─ Flow: PWCDCNet() 或 PWCDCNetCPU()

[ ] 2. 预处理输入
      ├─ LSR彩色: (B, 3, H, W) 归一化[0,1]
      ├─ LSR单色: (B, 1, H, W) 或 转灰度
      └─ RGB → YUV 转换

[ ] 3. RefIE 亮度增强
      └─ enhanced, scale = ref_ie(color_yuv, mono_yuv)

[ ] 4. RefAT 外观迁移
      ├─ 计算光流: flow = pwc_net([mono_up, mono])
      ├─ 变形参考: warp_ref = warp(ref_color, flow)
      └─ 融合: transfer = ref_at(enhanced, warp_ref, mono)

[ ] 5. RefSR 色度超分
      └─ final = ref_sr(transfer, hires_mono)

[ ] 6. 后处理输出
      └─ YUV → RGB 转换
      └─ 裁剪到 [0,1]
```
