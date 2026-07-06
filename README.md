# PatchCore-NG

基于 PatchCore（Memory Bank + kNN）的工业异常检测项目，提供 WPF 图形界面与命令行工具。

## 项目结构

```
PatchCoreNg/
├── src/PatchCoreNg.Core/   # 核心算法（训练、Coreset、推理、调参）
├── src/PatchCoreNg.App/    # WPF 图形界面
├── src/PatchCoreNg/        # 命令行工具
├── scripts/                # Python 脚本（ONNX 导出、示例数据）
├── models/                 # Backbone ONNX 与训练模型
├── config/                 # 超参数配置 JSON
└── data/                   # 示例数据
```

## 快速开始

### 1. 导出 Backbone ONNX

```bash
python scripts/export_backbone.py --all
```

### 2. 生成示例数据（可选）

```bash
python scripts/create_sample_data.py
```

### 3. 启动 WPF 界面

```bash
dotnet run --project src/PatchCoreNg.App
```

### 4. 命令行

```bash
# 训练
dotnet run --project src/PatchCoreNg -- train --data data/train/good --output models/patchcore_model.json

# 推理
dotnet run --project src/PatchCoreNg -- predict --model models/patchcore_model.json --input data/tune/ng --output output/predictions
```

## WPF 界面

界面分为两个 Tab：

- **训练与调参**：构建 Memory Bank，并用 OK/NG 样本自动搜索最佳阈值与 kNN
- **推理**：加载模型，对单张图像或目录批量检测，显示分数、OK/NG 标签与热力图

### 多套配置切换

窗口顶部提供 **配置切换栏**（训练与推理页均可见）：

| 操作 | 说明 |
|------|------|
| **当前配置** | 下拉选择已保存的配置，**选择后自动加载**全部参数（样本目录、模型路径、超参数等） |
| **保存** | 将当前参数写入当前配置 `{配置目录}/{名称}.json` |
| **另存为** | 以「另存为」输入框中的名称保存为新配置 |
| **新建** | 创建空白配置（内置默认值），填写后点保存 |
| **删除** | 删除选中的配置文件 |
| **刷新** | 重新扫描配置目录 |

- 每套配置独立保存：**样本目录**、**模型输出路径**、**推理模型路径**、Backbone、Coreset、kNN 等
- 修改未保存时，切换配置会提示是否保存；标题栏旁显示「未保存」标记
- 程序记住上次使用的配置（`config/.last_profile`），下次启动自动恢复

配置文件示例：`config/default.json`、`config/product_a.json`

在「训练与调参」页，**将鼠标悬停在各参数行或按钮上**可查看详细说明（ToolTip）。下方为相同内容的文档版，便于查阅与分享。

---

## 训练与调参 — 参数说明

### 1. 样本目录

| 参数 | 必填 | 说明 |
|------|------|------|
| **OK 训练目录** | 是 | 放置无缺陷的正常产品图像。PatchCore 会从中提取 patch 特征，经 Coreset 采样后写入 Memory Bank，作为后续异常检测的参照基准。建议覆盖典型正常外观变化，数量越多越好。 |
| **OK 调参目录** | 否 | 用于验证的正常样本，不参与 Memory Bank 构建。与 NG 调参目录一起，自动搜索使 OK/NG 区分效果最好的异常阈值和 kNN 邻居数。留空则使用 OK 训练目录评分（易偏乐观，建议单独准备验证 OK）。 |
| **NG 调参目录** | 推荐 | 放置已知缺陷/异常样本。系统对 OK 与 NG 样本分别打分，以 F1 最优为准自动确定异常阈值；勾选「自动搜索 kNN」时还会比较 1/3/5/9/15 等邻居数。未填写时跳过 OK/NG 调参，仅使用训练集 P95 作为阈值。 |

### 2. 模型与超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **模型输出** | — | 训练完成后保存的 `.json` 模型文件，包含 Memory Bank、Coreset 参数、调参后的异常阈值和 kNN 设置。推理页可直接加载此文件。 |
| **Backbone** | WideResNet-50 | 预训练 CNN，负责从图像提取多层 patch 特征。WideResNet-50 为 PatchCore 论文默认；ResNet-18 / MobileNet 更快；WideResNet-101 / ResNet-101 精度更高但更慢。切换 backbone 后须重新训练，且需先导出对应 ONNX。 |
| **ONNX 路径** | 自动关联 | Backbone 对应的 ONNX 特征提取模型路径。预设 backbone 自动关联 `models/` 目录下文件；选「自定义 ONNX」可手动指定。导出命令：`python scripts/export_backbone.py --backbone <名称>` |
| **输入尺寸** | 224 | 训练/推理前图像缩放边长（像素），须与 ONNX 导出尺寸一致。 |
| **Patch** | 3 | 局部邻域聚合窗口，须为奇数。每个 patch 与其周围特征做平均，使异常定位更稳定。 |
| **Coreset** | 0.1 | Coreset 采样比例（0~1），从全部 patch 中贪心选取代表性子集写入 Memory Bank。<br>• **0.1（10%）**：常用默认，精度与速度平衡<br>• **0.01~0.05**：更快、更省内存，适合大规模部署<br>• **越大**：Memory Bank 越大，训练/推理越慢，可能略增精度<br>比例越高，Coreset 采样耗时越长。 |
| **kNN 邻居** | 9 | 推理时将每个 patch 与 Memory Bank 中最近 k 个邻居比较，取平均距离作为异常分数。值越大分数越平滑。 |
| **自动搜索 kNN** | 开启 | 在 1/3/5/9/15 中结合 OK/NG 调参样本搜索 F1 最优值（需提供 NG 调参目录）。 |

#### 可选 Backbone

| 名称 | 特点 |
|------|------|
| WideResNet-50 | PatchCore 默认，精度与速度平衡 |
| WideResNet-101 | 更高精度，推理较慢 |
| ResNet-18 | 轻量快速 |
| ResNet-50 / ResNet-101 | 经典 backbone，工业常用 |
| EfficientNet-B0 | 效率较高 |
| MobileNet-V3-Large | 最快，适合边缘部署 |
| 自定义 ONNX | 手动指定 ONNX 路径 |

### 3. 配置管理（顶部切换栏）

| 参数 / 操作 | 说明 |
|-------------|------|
| **当前配置** | 下拉切换已保存的配置，选择后自动加载，无需再点「加载」 |
| **配置目录** | 超参数配置（`.json`）的保存位置，默认 `config/` |
| **另存为** | 输入新名称后点「另存为」，复制当前参数为新配置 |
| **保存 / 新建 / 删除** | 管理当前配置集合 |
| **恢复默认** | 将超参数恢复为内置默认值（在训练页「3. 其他」中） |

每套配置包含：样本目录、模型输出路径、推理模型路径、Backbone、Coreset、kNN 等。

### 4. 执行

| 操作 | 说明 |
|------|------|
| **训练并调参** | 依次执行：<br>1. 用 OK 训练目录构建 Memory Bank（Coreset 采样）<br>2. 若提供 NG 调参目录，用 OK/NG 样本自动搜索最优阈值与 kNN<br>3. 保存模型到「模型输出」路径<br>耗时取决于样本数量、Coreset 比例和 backbone 大小。 |

---

## 调参结果说明

训练并调参完成后，界面会显示：

- **F1 / 准确率 / 精确率 / 召回率**：基于 OK/NG 调参样本的评估指标
- **最佳阈值**：自动搜索得到的异常分数分界点
- **最佳 kNN**：若启用自动搜索，显示选中的邻居数

## 依赖

- .NET 8+ / .NET 10（WPF 应用）
- Python 3（导出 ONNX、生成示例数据）
- PyTorch、onnx（`scripts/export_backbone.py`）

## 参考

- PatchCore: [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)
