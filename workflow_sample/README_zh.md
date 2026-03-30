# QuantFunc ComfyUI 工作流

[English](README.md)

## 1. 概述

本目录包含 QuantFunc ComfyUI 插件的示例工作流。将 `.json` 文件导入 ComfyUI 即可快速开始。

| 工作流 | 说明 |
|--------|------|
| `QuantFunc-Easy-Gen.json` | 新手入门 3 节点工作流，自动下载模型，无需手动配置 |
| `QuantFunc-Text-to-Image-Workflow.json` | 文生图（SVDQ + Lighting） |
| `QuantFunc-Image-to-Image-Workflow.json` | 参考图像编辑（QwenImage-Edit） |
| `QuantFunc-Model-Export.json` | 导出运行时量化模型（支持融合 LoRA） |

每个工作流包含**两组节点** —— 左侧 **SVDQ**（预量化权重），右侧 **Lighting**（运行时量化）。根据你的模型选择对应的组。

## 2. 节点说明

### 2.1 QuantFunc Model Loader（模型加载器）

入口节点，配置加载哪个模型以及如何加载。

| 参数 | 必填 | 说明 |
|------|------|------|
| `model_dir` | 是 | 基础模型目录路径（diffusers 格式，含 `model_index.json`） |
| `transformer_path` | 是* | 量化 Transformer 权重路径（`.safetensors`）。*Lighting 运行时量化从 FP16 时留空 |
| `model_backend` | 是 | `svdq`（预量化）或 `lighting`（运行时量化） |
| `device` | 是 | GPU 编号（0, 1, ...） |
| `precision_config` | 仅 Lighting | 逐层精度 JSON 配置文件路径 |
| `prequant_weights` | 仅 Lighting | 预量化 modulation 权重路径 |
| `scheduler_config` | 可选 | 自定义调度器 JSON 路径（Lightning 蒸馏模型） |
| `api_key` | 可选 | QuantFunc API 密钥 |
| `fused_mod` | 可选 | 启用融合 modulation（Lighting，实验性） |

**输出：** `QUANTFUNC_PIPELINE` —— 连接 LoRA 节点或直接连接 Generate。

### 2.2 QuantFunc LoRA（LoRA 加载器）

向管线追加 LoRA 适配器。**可链式连接** —— 串联多个 LoRA 节点即可叠加 LoRA。

| 参数 | 说明 |
|------|------|
| `lora_path` | LoRA `.safetensors` 文件路径 |
| `strength` | LoRA 强度倍数（默认 1.0） |

### 2.3 QuantFunc LoRA Config（LoRA 配置）

控制 LoRA 合并策略。**SVDQ 后端使用 LoRA 时必须添加此节点**（控制新 LoRA 与 SVDQ 权重中已有 LoRA 的合并方式）。

| 参数 | 说明 |
|------|------|
| `merge_method` | `auto`（推荐）、`concat` 或 `replace` |
| `max_rank` | 最大合并 LoRA 秩（-1 = 自动） |

### 2.4 QuantFunc Generate（图像生成）

核心推理节点。根据文本生成图像，或使用参考图编辑图像。

| 参数 | 说明 |
|------|------|
| `prompt` | 文本提示词 |
| `width` / `height` | 输出图像尺寸 |
| `steps` | 去噪步数（Lightning 用 4，完整模型 8-30） |
| `seed` | 随机种子（-1 = 随机） |
| `guidance_scale` | CFG 引导强度 |
| `negative_prompt` | 反向提示词（True CFG） |
| `true_cfg_scale` | True CFG 强度（>1.0 启用双流 CFG） |
| `ref_images` | 可选 —— 连接 QuantFunc Image List 进入编辑模式 |

**输出：** `IMAGE` —— 连接 Preview Image 或 Save Image。

> 连接 `ref_images` 后，节点自动切换为**图像编辑模式**（QwenImage-Edit 管线）。

### 2.5 QuantFunc Image List（参考图列表）

收集 1-10 张参考图用于编辑模式。将 `LoadImage` 节点连接到输入槽位。

### 2.6 QuantFunc Pipeline Config（管线配置）

高级配置（可选）。连接到 Model Loader 时覆盖 `auto_optimize` 默认值。

| 参数 | 说明 |
|------|------|
| `cpu_offload` | 空闲模型卸载到 CPU |
| `layer_offload` | 逐层 Transformer 卸载（更省显存，更慢） |
| `tiled_vae` | 分块 VAE 解码（高分辨率） |
| `attention_backend` | `auto`、`sage`、`flash` 或 `sdpa` |
| `precision` | `bf16` 或 `fp16` |
| `text_precision` | 文本编码器精度：`int4`、`int8`、`fp4`、`fp8`、`fp16` |
| `adaptive_offload` | `off`、`normal`、`aggressive` |

> 大多数用户不需要此节点 —— `auto_optimize` 会自动处理一切。

### 2.7 QuantFunc Export（模型导出）

将所有运行时量化的模型导出到磁盘（支持融合 LoRA），后续加载无需重新量化。

| 参数 | 说明 |
|------|------|
| `export_path` | 导出目录 |
| `export_models` | 导出组件：`transformer`、`text_encoder`、`all` |

## 3. 快速上手

### 3.1 文生图（SVDQ）

1. 在 ComfyUI 中导入 `QuantFunc-Text-to-Image-Workflow.json`
2. 设置 `model_dir` 为基础模型路径
3. 设置 `transformer_path` 为 SVDQ `.safetensors` 文件
4. 设置 `model_backend` = `svdq`
5. 输入提示词，点击 **Queue Prompt**

### 3.2 文生图（Lighting）

1. 同一工作流，使用**右侧 Lighting 组**
2. 设置 `model_dir` 为 FP16 基础模型路径
3. `transformer_path` 留空（或指向已导出的预量化权重）
4. 设置 `model_backend` = `lighting`
5. 设置 `precision_config` 为精度 JSON 配置路径
6. 按需添加 LoRA 节点（零成本合并）

### 3.3 图像编辑

1. 导入 `QuantFunc-Image-to-Image-Workflow.json`
2. 配置 Model Loader（同文生图）
3. 通过 `LoadImage` → `QuantFunc Image List` 加载参考图
4. 将 Image List 连接到 Generate 的 `ref_images` 输入
5. 输入编辑提示词（如"把天空变成日落"）

### 3.4 模型导出

1. 导入 `QuantFunc-Model-Export.json`
2. 配置 Model Loader（Lighting 后端 + LoRA）
3. 在 Export 节点设置 `export_path`
4. 点击 **Queue Prompt** —— 模型运行时量化并将所有量化权重保存
5. 后续使用：在 `transformer_path` 填入导出的权重路径，即时加载（无需重新量化）

## 4. 模型下载

预量化模型下载地址：
- **ModelScope**: https://www.modelscope.cn/models/QuantFunc
- **HuggingFace**: https://huggingface.co/QuantFunc

> 基础模型与 Transformer 权重必须使用**相同的 GPU 变体**（`50x-below` 适用 RTX 30/40，`50x-above` 适用 RTX 50）。
