# 新手入门必看：3 个节点生成你的第一张图

[English Version](tutorial-0-easy-gen.md)

## 概述

这是最简单的入门方式。**QuantFunc Model AutoLoader** 节点会**自动下载**并配置模型——你只需从下拉菜单中选择，无需手动下载模型或填写路径。

整个工作流只有 **3 个节点**：

```
Model AutoLoader → Generate → Preview Image
```

![Easy Gen 工作流全貌](../assets/t0-workflow-overview.png)

> **工作流文件：** [`workflow_sample/QuantFunc-Easy-Gen.json`](../workflow_sample/QuantFunc-Easy-Gen.json)

> **想要更多控制？** 本教程是[教程 1（运行时量化）](tutorial-1-use-without-quantfunc-models_zh.md)的自动下载简化版。如果你需要手动指定本地模型、添加 LoRA、调整管线配置等高级操作，请参考教程 1。

## 前置条件

1. 已安装 ComfyUI-QuantFunc 插件（参见 [README](../README_zh.md)）
2. 已安装 CUDA 13.0+ 运行时及 cuDNN 9.x
3. 网络连接正常（首次使用需要自动下载模型）

## 步骤

### 第一步：导入工作流

在 ComfyUI 中导入 `workflow_sample/QuantFunc-Easy-Gen.json`。

你会看到 3 个节点已经连接好了——不需要任何额外操作。

### 第二步：配置 Model AutoLoader

在 **QuantFunc Model AutoLoader** 节点中：

| 参数 | 说明 |
|------|------|
| `model_series` | 选择模型系列（如 `QuantFunc/Z-Image-Series`） |
| `model_backend` | 量化后端：`svdq` 或 `lighting` |
| `device` | 你的 GPU（从列表中选择） |
| `data_source` | 下载源：`modelscope`（国内推荐）或 `huggingface` |
| `transformer` | 选择具体的 Transformer 权重（根据你的 GPU 变体自动列出） |

其他参数保持默认即可。

> **提示：** 选择 `model_series` 后，`transformer` 下拉菜单会自动列出该系列可用的权重文件。选择与你 GPU 匹配的变体（`50x-below` 适用 RTX 20/30/40，`50x-above` 适用 RTX 50）。

### 第三步：配置生成参数

在 **QuantFunc Generate** 节点中：

| 参数 | 建议值 |
|------|--------|
| `prompt` | 你的文本提示词（如 "A cute cat"） |
| `width` / `height` | `1024` x `1024` |
| `steps` | `8`（Lightning 蒸馏模型）或 `20`（完整模型） |
| `seed` | 任意数字，或选择 `randomize` 自动随机 |
| `guidance_scale` | `0`（Lightning 蒸馏模型）或 `3.5`（完整模型） |

### 第四步：运行

点击 **Queue Prompt**。首次运行时插件会自动下载模型（取决于网速），后续运行直接使用缓存。

生成的图像会显示在 **Preview Image** 节点中。

## 下一步

- 想用自己的本地模型？→ [教程 1：运行时量化](tutorial-1-use-without-quantfunc-models_zh.md)
- 想导出量化模型加速加载？→ [教程 2：导出运行时量化模型](tutorial-2-export-quantized-models_zh.md)
- 想了解 SVDQ 与 Lighting 的区别？→ [教程 3：下载并使用已导出的量化模型](tutorial-3-download-quantfunc-models_zh.md)
