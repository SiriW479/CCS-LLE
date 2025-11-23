# CCS-LLE 文档导航

欢迎使用 CCS-LLE 代码分析文档！本指南将帮助你快速定位所需的信息。

---

## 📚 文档结构

### 🎯 根据需求选择文档

#### 1️⃣ **想快速了解整个项目？**
👉 **[ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md)**
- ⏱️ 阅读时间: 15-20分钟
- 📋 内容:
  - 项目核心创新
  - 三大模块概览
  - 完整数据流
  - 关键技术总结
- 🎓 适合: 初学者、论文阅读者

---

#### 2️⃣ **需要详细的架构和工作流信息？**
👉 **[PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md)**
- ⏱️ 阅读时间: 30-40分钟
- 📋 内容:
  - 论文三大模块与代码的精确对应
  - 每个模块的详细分析 (RefIE/RefAT/RefSR)
  - 关键技术栈表格
  - 损失函数详解
  - 数据加载机制
- 🎓 适合: 想深入理解论文实现的研究者

---

#### 3️⃣ **需要详细的技术实现细节？**
👉 **[DETAILED_ARCHITECTURE.md](DETAILED_ARCHITECTURE.md)**
- ⏱️ 阅读时间: 45-60分钟
- 📋 内容:
  - 每个模块的完整工作流程图
  - 各类的参数详解
  - 关键设计决策的原因分析
  - 改进点的对比
  - 端到端数据流向图
- 🎓 适合: 想从事后续研究或改进的开发者

---

#### 4️⃣ **需要快速查找代码实现？**
👉 **[CODE_REFERENCE.md](CODE_REFERENCE.md)**
- ⏱️ 阅读时间: 即时查询
- 📋 内容:
  - 按功能分类的代码片段
  - 完整类定义及方法签名
  - 代码调用流程
  - 快速查找表
- 🎓 适合: 修改代码、集成接口、调试问题

---

## 🔍 按任务快速导航

### 任务: "理解RefIE模块"
1. 快速概览 → [ANALYSIS_SUMMARY.md - RefIE部分](ANALYSIS_SUMMARY.md#1️⃣-refie-reference-based-illumination-enhancement)
2. 详细工作流 → [PROJECT_ANALYSIS.md - RefIE部分](PROJECT_ANALYSIS.md#1️⃣-refieReference-based-Illumination-Enhancement)
3. 代码实现 → [DETAILED_ARCHITECTURE.md - 蓝色部分](DETAILED_ARCHITECTURE.md#🔵-refie-reference-based-illumination-enhancement)
4. 查找代码 → [CODE_REFERENCE.md - RefIE部分](CODE_REFERENCE.md#🔵-亮度增强-illumination-enhancement---refie)

### 任务: "修改光流模块参数"
1. 了解PWCNet → [DETAILED_ARCHITECTURE.md - PWCNet部分](DETAILED_ARCHITECTURE.md#🔴-光流对齐详解-pwcnetpy)
2. 查找关键代码 → [CODE_REFERENCE.md - 光流部分](CODE_REFERENCE.md#【光流计算】)
3. 参数查询 → [CODE_REFERENCE.md - 配置常数](CODE_REFERENCE.md#🔧-配置常数)

### 任务: "理解损失函数"
1. 快速了解 → [ANALYSIS_SUMMARY.md - 损失函数](ANALYSIS_SUMMARY.md#💾-损失函数设计)
2. 详细实现 → [PROJECT_ANALYSIS.md - 损失函数](PROJECT_ANALYSIS.md#💾-损失函数定义mylosspy)
3. 代码查询 → [CODE_REFERENCE.md - 损失函数](CODE_REFERENCE.md#🎯-损失函数速查)

### 任务: "运行推理"
1. 了解流程 → [ANALYSIS_SUMMARY.md - 推理过程](ANALYSIS_SUMMARY.md#🎮-推理过程)
2. 详细步骤 → [CODE_REFERENCE.md - 推理步骤](CODE_REFERENCE.md#✅-推理步骤检查清单)
3. 查找实现 → [CODE_REFERENCE.md - 完整数据流](CODE_REFERENCE.md#【完整数据流】)

---

## 📊 层级结构

```
CCS-LLE 项目分析文档
│
├─ 入门级 (初学者)
│  └─ ANALYSIS_SUMMARY.md
│     • 快速总览
│     • 核心概念
│     • 高层架构
│
├─ 中级 (研究人员)
│  ├─ PROJECT_ANALYSIS.md
│  │  • 详细技术分析
│  │  • 论文对应
│  │  • 完整设计
│  │
│  └─ DETAILED_ARCHITECTURE.md
│     • 实现细节
│     • 工作流程
│     • 设计决策
│
└─ 高级 (开发者)
   └─ CODE_REFERENCE.md
      • 代码片段
      • 快速查询
      • 参数配置
```

---

## 🎯 常见问题快速定位

| 问题 | 答案位置 |
|------|---------|
| 项目在做什么？ | [ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md#🎯-核心创新) |
| 三个模块分别是什么？ | [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md#🏗️-核心架构与模块对应) |
| 为什么用YUV而不是RGB？ | [DETAILED_ARCHITECTURE.md](DETAILED_ARCHITECTURE.md#为什么-yuv-而不是-rgb) |
| PWCNet怎样工作？ | [DETAILED_ARCHITECTURE.md](DETAILED_ARCHITECTURE.md#🔴-光流对齐详解-pwcnetpy) |
| 如何修改参数？ | [CODE_REFERENCE.md](CODE_REFERENCE.md#🔧-配置常数) |
| 怎样运行推理？ | [CODE_REFERENCE.md](CODE_REFERENCE.md#✅-推理步骤检查清单) |
| RefIE的关键类？ | [CODE_REFERENCE.md](CODE_REFERENCE.md#【全局亮度尺度估计】) |
| 怎样调试光流？ | [DETAILED_ARCHITECTURE.md](DETAILED_ARCHITECTURE.md#版本2-cpu版本-pwcdcnetcpu) |

---

## 📝 文档功能表

### 各文档的主要功能

```
┌──────────────────┬────────┬────────┬─────────┬──────────┐
│ 文档名称          │ 理论  │ 实现  │ 参数   │ 代码   │
├──────────────────┼────────┼────────┼─────────┼──────────┤
│ SUMMARY           │ ████  │ ██    │ -       │ -        │
│ PROJECT_ANALYSIS  │ ████  │ ████  │ ██      │ ██       │
│ DETAILED_ARCH     │ ████  │ ████  │ ███     │ ███      │
│ CODE_REFERENCE    │ ██    │ ███   │ ████    │ ████     │
└──────────────────┴────────┴────────┴─────────┴──────────┘

理论: ■ = 深度程度
实现: ■ = 实现细节
参数: ■ = 参数列表
代码: ■ = 代码示例
```

---

## 🧭 阅读路径建议

### 路径1️⃣: 快速上手 (1小时)
```
开始 → ANALYSIS_SUMMARY.md (20分钟)
    ↓
    → 选择感兴趣的模块
    ↓
    → CODE_REFERENCE.md 对应部分 (15分钟)
    ↓
    → 理解关键代码 (25分钟)
    ↓
完成 ✅
```

### 路径2️⃣: 深度学习 (3小时)
```
开始 → ANALYSIS_SUMMARY.md (20分钟)
    ↓
    → PROJECT_ANALYSIS.md 完整阅读 (45分钟)
    ↓
    → DETAILED_ARCHITECTURE.md 完整阅读 (45分钟)
    ↓
    → CODE_REFERENCE.md 对应查询 (30分钟)
    ↓
完成 ✅
```

### 路径3️⃣: 开发改进 (4小时)
```
开始 → ANALYSIS_SUMMARY.md (20分钟)
    ↓
    → PROJECT_ANALYSIS.md (45分钟)
    ↓
    → DETAILED_ARCHITECTURE.md (60分钟)
    ↓
    → CODE_REFERENCE.md 深入查询 (45分钟)
    ↓
    → 修改代码并测试 (60分钟)
    ↓
完成 ✅
```

---

## 💡 使用技巧

### 1. Markdown 导航
所有文档都使用 Markdown 格式，支持快速导航：
- 按 `Ctrl+F` 搜索关键词
- 点击目录链接快速跳转
- 使用emoji快速定位章节

### 2. 交叉引用
文档间相互链接，可快速跳转：
```markdown
[查看详细工作流](DETAILED_ARCHITECTURE.md#-refie-工作流)
[查看代码实现](CODE_REFERENCE.md#-亮度增强)
```

### 3. 代码查询
CODE_REFERENCE.md 提供：
- ✅ 完整类定义
- ✅ 方法签名
- ✅ 使用示例
- ✅ 参数说明

### 4. 快速查找表
每个文档都有快速查找表，可直接定位信息

---

## 🔗 外部资源

### 论文与项目
- **官方仓库**: https://github.com/peiyaoooo/CCS-LLE
- **论文**: "Low-Light Color Imaging via Cross-Camera Synthesis", IEEE JSTSP 2022
- **作者**: Guo, Peiyao et al.

### 相关技术
- **PWC-Net**: https://github.com/NVlabs/PWC-Net
- **HDRNet**: https://github.com/google/hdrnet
- **PyTorch**: https://pytorch.org

---

## ✅ 文档检查清单

在使用文档前，请确保：

- [ ] 已下载所有四个 MD 文件
- [ ] 文件编码为 UTF-8 (支持中文)
- [ ] 使用支持 Markdown 的编辑器 (VS Code, Typora等)
- [ ] 有基础的深度学习知识
- [ ] 熟悉 PyTorch 框架

---

## 📞 问题反馈

如有问题或建议，可：
1. 查阅文档中的"常见问题"部分
2. 在对应文档中搜索关键词
3. 参考代码注释和实现

---

## 📈 文档更新历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2025-11-20 | 初版发布，包含4个MD文件 |

---

## 🎓 推荐学习顺序

### 初次接触CCS-LLE
```
1. 阅读论文摘要和概述
2. 浏览 ANALYSIS_SUMMARY.md
3. 理解三个模块的基本功能
4. 查看关键代码实现
5. 运行测试脚本 (test-3ref-clean.py)
```

### 深度研究CCS-LLE
```
1. 完整阅读论文
2. 详细学习 PROJECT_ANALYSIS.md
3. 理解 DETAILED_ARCHITECTURE.md 中的工作流
4. 研究 CODE_REFERENCE.md 中的具体实现
5. 尝试修改参数和重新训练
```

### 改进或扩展CCS-LLE
```
1. 掌握上述所有内容
2. 分析代码瓶颈和改进空间
3. 设计改进方案
4. 修改代码进行实验
5. 对比结果并总结
```

---

**祝你学习愉快！🎉**

有任何问题，欢迎参考文档中的相关章节。
