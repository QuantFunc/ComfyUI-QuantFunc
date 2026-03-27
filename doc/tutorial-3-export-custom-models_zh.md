# 教程 3：将自定义模型与 LoRA 导出为 Lighting/SVDQ 模型

[English Version](tutorial-3-export-custom-models.md)

## 概述

如果你精心调试了模型参数、叠加了多个 LoRA，你可以将当前配置**一键导出**为预量化模型。导出后的模型：

- **LoRA 已烘焙**：无需每次运行时重新加载 LoRA
- **量化权重已保存**：跳过实时量化步骤，加载即用
- **可分享**：导出的模型可以分享给他人直接使用

## 使用场景

| 场景 | 说明 |
|------|------|
| LoRA 烘焙 | 将你调试好的 LoRA（含强度配置）永久融入模型 |
| 加速加载 | Lighting 实时量化后导出，下次直接加载量化权重 |
| 模型分发 | 将配置好的模型打包分享给团队成员 |
| 多 LoRA 合并 | 将多个 LoRA 合并为单一模型，简化工作流 |

## 第一步：导入导出工作流

在 ComfyUI 中导入 `workflow_sample/QuantFunc-Model-Export.json`。

## 第二步：配置 Model Loader

根据你的模型选择后端：

### 方案 A：从 FP16 模型导出（Lighting 后端）

```
QuantFunc Model Loader (lighting)
    → QuantFunc LoRA (LoRA 1)
        → QuantFunc LoRA (LoRA 2, 可选)
            → QuantFunc Export
```

Lighting 后端的 Model Loader 配置与[教程 1](tutorial-1-use-without-quantfunc-models_zh.md) 相同：

| 参数 | 设置 |
|------|------|
| `model_dir` | 你的 FP16 基础模型路径，例如 `/path/to/Qwen-Image-Edit-2511` |
| `transformer_path` | **留空** —— Lighting 会从 FP16 实时量化 |
| `model_backend` | `lighting` |
| `device` | GPU 编号（通常为 `0`） |
| `precision_config` | 逐层精度配置文件路径（详见[教程 1](tutorial-1-use-without-quantfunc-models_zh.md)） |
| `fused_mod` | Qwen 系列模型建议开启 `True`（与 `prequant_weights` 互斥） |
| `prequant_weights` | 预量化调制权重路径，低显存 GPU 推荐（与 `fused_mod` 互斥） |

> **调制层优化选择：** 24 GB+ 显存用 `fused_mod = True`（画质更好）；8-12 GB 显存用 `prequant_weights`（模型从 ~14 GB 降到 ~11 GB）。导出时的选择会保存到模型元数据中，加载时自动启用。详见[教程 1 的调制层优化说明](tutorial-1-use-without-quantfunc-models_zh.md)。

![Lighting 后端 Model Loader 配置](../assets/t1-step2-model-loader.png)

### 方案 B：从 SVDQ 模型导出（SVDQ 后端）

```
QuantFunc Model Loader (svdq)
    → QuantFunc LoRA (LoRA 1)
        → QuantFunc LoRA Config (合并策略)
            → QuantFunc Export
```

SVDQ 后端的 Model Loader 配置与[教程 2](tutorial-2-download-and-use-quantfunc-models_zh.md) 相同：

| 参数 | 设置 |
|------|------|
| `model_dir` | QuantFunc 模型目录路径，例如 `/path/to/QuantFunc-Model` |
| `transformer_path` | Transformer 权重路径，例如 `/path/to/QuantFunc-Model/transformer/model.safetensors`（也兼容旧版 nunchaku 的量化权重） |
| `model_backend` | `svdq` |
| `device` | GPU 编号（通常为 `0`） |

> SVDQ 后端导出时，如果叠加了 LoRA，必须添加 LoRA Config 节点。详见[教程 2 的 LoRA 配置说明](tutorial-2-download-and-use-quantfunc-models_zh.md)。

![SVDQ 后端 Model Loader 配置](../assets/t2-step3-import-workflow.png)

## 第三步：添加 LoRA（可选）

在 Model Loader 和 Export 之间插入 **QuantFunc LoRA** 节点：

```
Model Loader → LoRA (scale=0.8) → LoRA (scale=1.2) → Export
```

每个 LoRA 节点：
- `lora_path`：LoRA 文件路径
- `scale`：LoRA 强度（0.0-2.0）

> 你在这里设置的 LoRA 强度会被永久烘焙到导出的模型中。

![添加多个 LoRA 节点](../assets/t1-optional-add-lora.png)

## 第四步：配置 Export 节点

在 **QuantFunc Export** 节点中：

| 参数 | 说明 |
|------|------|
| `export_path` | 导出目录，例如 `/path/to/my-exported-model` |
| `export_mode` | `all` —— 导出完整模型（推荐，包含 VAE、tokenizer 等） |
| | `custom` —— 自定义选择导出组件 |

如果选择 `custom`，可以单独控制：

| 参数 | 说明 |
|------|------|
| `export_transformer` | 导出 Transformer（量化权重 + 烘焙的 LoRA） |
| `export_text_encoder` | 导出文本编码器 |
| `export_vision_encoder` | 导出视觉编码器 |

> **推荐使用 `all`**，这样导出的模型是完整的、独立的，可以直接作为 `model_dir` 使用。

![配置 Export 节点](../assets/t3-step4-export-config.png)

## 第五步：执行导出

点击 **Queue Prompt**。导出过程会：

1. 加载基础模型
2. 应用所有 LoRA（按配置的强度和合并策略）
3. 执行量化（如果是 Lighting 从 FP16 量化）
4. 将量化后的权重保存到指定目录

导出完成后，目录结构类似：

```
my-exported-model/
├── model_index.json
├── transformer/
│   └── *.safetensors    ← 量化权重（含烘焙的 LoRA）
├── vae/
├── tokenizer/
├── text_encoder/
└── scheduler/
```

![导出完成](../assets/t3-step5-export-done.png)

## 第六步：使用导出的模型

导出的模型可以用两种方式加载：

### 方式 A：作为完整模型（推荐，适用于 `all` 导出模式）

| 参数 | 设置 |
|------|------|
| `model_dir` | `/path/to/my-exported-model` |
| `transformer_path` | 留空或指向导出的 Transformer 权重 |
| `model_backend` | `lighting`（导出的量化权重作为预量化加载） |

### 方式 B：仅替换 Transformer 权重

| 参数 | 设置 |
|------|------|
| `model_dir` | 原始基础模型路径 |
| `transformer_path` | `/path/to/my-exported-model/transformer/model.safetensors` |
| `model_backend` | 与导出时相同 |

> 使用导出模型时**不需要**再添加之前的 LoRA 节点——LoRA 已经烘焙进去了。

![使用导出的模型](../assets/t1-step2-model-loader.png)

## 完整示例：从头到尾

假设你有：
- 基础模型：`/models/FLUX.1-dev/`
- 风格 LoRA：`/loras/anime-style.safetensors`（强度 0.8）
- 细节 LoRA：`/loras/detail-enhancer.safetensors`（强度 1.2）

**导出流程：**

```
Model Loader                    Export
  model_dir: /models/FLUX.1-dev/    export_path: /models/my-anime-flux/
  transformer_path: (空)             export_mode: all
  model_backend: lighting
      ↓
  LoRA (anime-style, scale=0.8)
      ↓
  LoRA (detail-enhancer, scale=1.2)
      ↓
  Export
```

**使用导出模型：**

```
Model Loader                    Generate
  model_dir: /models/my-anime-flux/   prompt: "1girl, anime style..."
  transformer_path: (空)               steps: 20
  model_backend: lighting              ...
      ↓
  Generate → Preview Image
```

无需 LoRA 节点，加载即用！

## 常见问题

**Q: 导出的模型可以再叠加新 LoRA 吗？**
A: SVDQ 后端导出的模型可以继续叠加新 LoRA（需要 LoRA Config 节点）。但 **Lighting 后端导出的模型目前不支持再叠加新 LoRA**——如果需要更换 LoRA 组合，请从原始 FP16 模型重新导出。

**Q: 导出需要多长时间？**
A: 取决于模型大小和后端。Lighting 从 FP16 导出需要量化时间（几分钟），SVDQ 导出相对快一些。

**Q: 导出的模型体积有多大？**
A: INT4 量化后的 Transformer 权重通常只有 FP16 的 1/4 左右。完整模型（含 VAE、tokenizer）的总大小取决于各组件。

**Q: `custom` 导出模式什么时候用？**
A: 当你只想更新 Transformer 权重（比如换了 LoRA 组合），而 VAE、tokenizer 等不变时，用 `custom` 只导出 Transformer 可以节省时间和空间。
