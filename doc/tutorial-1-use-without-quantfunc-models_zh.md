# 教程 1：运行时量化 —— 将 BF16/FP16 原模型量化为 4bit 加速推理

[English Version](tutorial-1-use-without-quantfunc-models.md)

## 概述

你**不需要**下载 QuantFunc 预量化模型也能使用本插件。**Lighting 后端**提供**运行时量化**能力 —— 在加载时将任意 **diffusers 格式**的 FP16 模型（例如 [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)）运行时量化并加速推理，无需预先转换。

![工作流全貌](../assets/t1t3-workflow-overview.png)

> **工作流文件（使用右侧 Lighting 组）：**
> - 文生图：[`workflow_sample/QuantFunc-Text-to-Image-Workflow.json`](../workflow_sample/QuantFunc-Text-to-Image-Workflow.json)
> - 图像编辑：[`workflow_sample/QuantFunc-Image-to-Image-Workflow.json`](../workflow_sample/QuantFunc-Image-to-Image-Workflow.json)

## 前置条件

1. 已安装 ComfyUI-QuantFunc 插件（参见 [README_zh.md](../README_zh.md)）
2. 已安装 CUDA 13.0+ 运行时及 cuDNN 9.x
3. 下载一个 diffusers 格式的模型到本地，例如：

```bash
# 使用 huggingface-cli 下载
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir /path/to/Qwen-Image-Edit-2511

# 或使用 git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen-Image-Edit-2511 /path/to/Qwen-Image-Edit-2511
```

> **什么是 diffusers 格式？** 目录下应包含 `model_index.json` 文件，以及 `transformer/`、`vae/`、`tokenizer/` 等子目录。

## 步骤

### 第一步：导入 Workflow

在 ComfyUI 中导入 `workflow_sample/QuantFunc-Text-to-Image-Workflow.json`。

工作流包含两组节点——左侧 SVDQ，右侧 Lighting。**删除或忽略左侧 SVDQ 组**，只使用右侧 Lighting 组。

![导入 Workflow 并选择 Lighting 组](../assets/t1-step1-import-workflow.png)

### 第二步：配置 Model Loader

在 **QuantFunc Model Loader** 节点中：

| 参数 | 设置 |
|------|------|
| `model_dir` | 你的基础模型路径，例如 `/path/to/Qwen-Image-Edit-2511` |
| `transformer_path` | **留空** —— Lighting 会从 FP16 运行时量化 |
| `model_backend` | 选择 `lighting` |
| `device` | GPU 编号（通常为 `0`） |
| `precision_config` | 逐层精度配置文件路径（见下方说明） |
| `fused_mod` | Qwen 系列模型建议开启 `True`（见下方说明） |
| `prequant_weights` | 预量化调制权重路径，低显存 GPU 推荐（见下方说明） |

![配置 Model Loader 节点](../assets/t1-step2-model-loader.png)

#### 关于 precision_config（逐层精度配置）

`precision_config` 是 Lighting 后端的核心配置，它定义了 Transformer 每一层使用的量化精度（如 INT4、INT8、FP8 等）。通过精细控制每层的精度，可以在**速度和画质之间取得最佳平衡**。

**QuantFunc 为各系列模型提供了官方推荐配置**，你可以直接下载使用。

> **注意：** precision_config 是**与模型架构绑定**的，不同模型系列的配置**不能混用**。请务必下载与你使用的模型对应的配置文件。

以 Qwen-Image-Edit 系列为例：

> 示例下载地址（Qwen-Image-Edit 系列）：
> https://www.modelscope.cn/models/QuantFunc/Qwen-Image-Edit-Series/file/view/master/precision-config

```bash
# 示例：下载 Qwen-Image-Edit 系列的 precision config
modelscope download --model QuantFunc/Qwen-Image-Edit-Series --include "precision-config/*" --local_dir /path/to/configs
```

其他模型系列请在 [QuantFunc ModelScope 主页](https://www.modelscope.cn/models/QuantFunc) 找到对应模型仓库下的 `precision-config` 目录。

下载后在 Model Loader 中设置：
```
precision_config = /path/to/configs/precision-config/your-model-config.json
```

> **如果不设置 precision_config**，Lighting 引擎会使用默认的全局量化精度。设置后可以获得更好的画质或更快的速度，取决于所选配置文件。

![配置 precision_config](../assets/t1-step2-precision-config.png)

#### 关于 fused_mod（融合 Modulation 内核）

`fused_mod` 启用后会将 modulation 层的 SiLU、GEMV、bias、split 等操作融合为单个 FP16 内核，减少显存访问开销，提升推理速度。

> **建议：使用 Qwen 系列模型（如 Qwen-Image-Edit-2511）时，将 `fused_mod` 设为 `True`。** Qwen 系列的 Transformer 结构非常适合此优化，开启后可获得明显的性能提升。

其他模型架构是否受益于此选项取决于其 modulation 层的实现方式，如不确定可先保持默认关闭。

#### 调制层优化：fused_mod vs prequant_weights（仅 QwenImage 系列）

QwenImage 系列提供两种**互斥**的调制层优化方案，根据你的显存情况选择其一：

| 你的显存 | 推荐方案 | 设置方式 |
|----------|----------|----------|
| **24 GB+**（RTX 4090 等） | `fused_mod = True` | 画质更好，模型约 14 GB |
| **8–12 GB**（RTX 3060 等） | `prequant_weights = 路径` | 模型约 11 GB，推理约 9 秒（无此优化需 20 秒+） |

> **注意：** 两个选项互斥 —— 设置了 `prequant_weights` 后，`fused_mod` 不生效。

**使用 prequant_weights：**

1. 从 [QuantFunc ModelScope](https://www.modelscope.cn/models/QuantFunc) 或 HuggingFace 下载对应模型的 `mod_weights.safetensors` 文件
2. 在 Model Loader 中设置：
```
prequant_weights = /path/to/mod_weights.safetensors
```

> 导出时选择的方案会保存到模型元数据中，加载导出模型时自动启用。

### 第三步：配置生成参数

在 **QuantFunc Generate** 节点中：

| 参数 | 建议值 |
|------|--------|
| `prompt` | 你的文本提示词 |
| `width` / `height` | `1024` x `1024`（或模型支持的尺寸） |
| `steps` | `20`（完整模型），`4`（Lightning 蒸馏模型） |
| `guidance_scale` | `3.5`（根据模型调整） |
| `seed` | 任意数字 |

![配置 Generate 节点参数](../assets/t1-step3-generate-params.png)

### 第四步：运行

点击 **Queue Prompt**。首次运行时 Lighting 引擎会对模型进行运行时量化（需要额外几十秒），后续运行会使用缓存加速。

![运行结果预览](../assets/t1-step4-run-result.png)

## 节点连接示意

```
QuantFunc Model Loader (lighting, FP16)
    → QuantFunc LoRA (可选，加载你的 LoRA)
        → QuantFunc Generate
            → Preview Image
```

## 可选：添加 LoRA

Lighting 后端支持**零成本 LoRA 叠加**。在 Model Loader 和 Generate 之间插入 **QuantFunc LoRA** 节点：

1. 设置 `lora_path` 为你的 LoRA `.safetensors` 文件路径
2. 调整 `scale`（默认 1.0）
3. 可以串联多个 LoRA 节点

> Lighting 后端叠加 LoRA **不需要** QuantFunc LoRA Config 节点。

![添加 LoRA 节点](../assets/t1-optional-add-lora.png)

## 可选：图像编辑模式

如果你使用的是图像编辑模型（如 Qwen-Image-Edit-2511）：

1. 导入 `workflow_sample/QuantFunc-Image-to-Image-Workflow.json`
2. 使用 **LoadImage** 节点加载参考图
3. 连接到 **QuantFunc Image List** 节点
4. 将 Image List 连接到 Generate 的 `ref_images` 输入
5. 在 prompt 中描述编辑内容（如 "把背景换成海滩"）

连接 `ref_images` 后，节点会自动切换为图像编辑模式。

![图像编辑模式工作流](../assets/t1-optional-image-edit.png)

## 可选：高级管线配置

如果遇到显存不足或想要调优，可添加 **QuantFunc Pipeline Config** 节点并连接到 Model Loader：

| 参数 | 说明 |
|------|------|
| `cpu_offload` | 显存不够时开启，将空闲模型卸载到 CPU |
| `layer_offload` | 极端低显存场景，逐层加载 Transformer |
| `tiled_vae` | 生成高分辨率图像时开启 |
| `attention_backend` | 通常保持 `auto` |
| `text_precision` | 文本编码器精度，`int4` 最省显存 |

> 大多数情况下不需要此节点——插件会自动优化。

![高级管线配置节点](../assets/t1-optional-pipeline-config.png)

## 常见问题

**Q: 首次加载很慢？**
A: 正常现象。Lighting 首次运行需要进行运行时量化，后续运行会使用缓存的量化模型。你也可以使用[教程 2](tutorial-2-export-quantized-models_zh.md) 将所有运行时量化的模型导出到磁盘，以后加载时完全跳过量化步骤。

**Q: 哪些模型可以用？**
A: 任何 diffusers 格式的模型。目前支持的架构请查看 [QuantFunc 官方文档](https://www.modelscope.cn/models/QuantFunc)。

**Q: 和下载预量化模型比有什么区别？**
A: 已导出的量化模型（包括 SVDQ）加载更快，因为跳过了运行时量化步骤。推理速度方面，SVDQ 与 Lighting 基本一致，在 RTX 50 以下的机器上 Lighting 甚至比 SVDQ 快约 20%。Lighting 没有 SVDQ 的 low-rank 计算开销，因此在 RTX 50 以下机器上推理更快。详见[教程 3](tutorial-3-download-quantfunc-models_zh.md)。
