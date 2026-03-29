# QuantFunc ComfyUI Workflows

[中文说明](README_zh.md)

## 1. Overview

This directory contains example workflows for the QuantFunc ComfyUI plugin. Import any `.json` file into ComfyUI to get started quickly.

| Workflow | Description |
|----------|-------------|
| `QuantFunc-Text-to-Image-Workflow.json` | Text-to-image generation (SVDQ + Lighting) |
| `QuantFunc-Image-to-Image-Workflow.json` | Reference-based image editing (QwenImage-Edit) |
| `QuantFunc-Model-Export.json` | Export runtime-quantized models with LoRA fusion support |

Each workflow file contains **two groups** side by side — one for **SVDQ** (pre-quantized weights) and one for **Lighting** (runtime quantization). Use the one that matches your model.

## 2. Node Reference

### 2.1 QuantFunc Model Loader

The entry point. Configures which model to load and how.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_dir` | Yes | Path to base model directory (diffusers format, contains `model_index.json`) |
| `transformer_path` | Yes* | Path to quantized transformer weights (`.safetensors`). *Leave empty for Lighting runtime quantization from FP16 |
| `model_backend` | Yes | `svdq` (pre-quantized) or `lighting` (runtime quantization) |
| `device` | Yes | GPU index (0, 1, ...) |
| `precision_config` | Lighting only | Path to per-layer precision JSON config |
| `prequant_weights` | Lighting only | Path to pre-quantized modulation weights |
| `scheduler_config` | Optional | Custom scheduler JSON path (for Lightning distilled models) |
| `api_key` | Optional | QuantFunc API key for authenticated models |
| `fused_mod` | Optional | Enable fused modulation (Lighting, experimental) |

**Output:** `QUANTFUNC_PIPELINE` — connect to LoRA nodes or directly to Generate.

### 2.2 QuantFunc LoRA

Appends a LoRA adapter to the pipeline. **Chainable** — connect multiple LoRA nodes in sequence to stack LoRAs.

| Parameter | Description |
|-----------|-------------|
| `lora_path` | Path to LoRA `.safetensors` file |
| `strength` | LoRA strength multiplier (default: 1.0) |

### 2.3 QuantFunc LoRA Config

Controls how LoRAs are merged. **Required when using LoRA with SVDQ backend** (controls merge strategy with the pre-existing LoRA in SVDQ weights).

| Parameter | Description |
|-----------|-------------|
| `merge_method` | `auto` (recommended), `concat`, or `replace` |
| `max_rank` | Maximum merged LoRA rank (-1 = auto) |

### 2.4 QuantFunc Generate

The core inference node. Produces images from text (or edits images with references).

| Parameter | Description |
|-----------|-------------|
| `prompt` | Text prompt for generation |
| `width` / `height` | Output image dimensions |
| `steps` | Number of denoising steps (4 for Lightning, 8-30 for full) |
| `seed` | Random seed (-1 = random) |
| `guidance_scale` | CFG guidance scale |
| `negative_prompt` | Negative prompt (for True CFG) |
| `true_cfg_scale` | True CFG scale (>1.0 enables dual-stream CFG) |
| `ref_images` | Optional — connect QuantFunc Image List for edit mode |

**Output:** `IMAGE` — connect to Preview Image or Save Image.

> When `ref_images` is connected, the node automatically switches to **image editing mode** (QwenImage-Edit pipeline).

### 2.5 QuantFunc Image List

Collects 1–10 reference images for editing mode. Connect `LoadImage` nodes to the input slots.

### 2.6 QuantFunc Pipeline Config

Advanced configuration (optional). Overrides `auto_optimize` defaults when connected to Model Loader.

| Parameter | Description |
|-----------|-------------|
| `cpu_offload` | Offload idle models to CPU |
| `layer_offload` | Per-layer transformer offload (less VRAM, slower) |
| `tiled_vae` | Tile-based VAE decoding for high-res |
| `attention_backend` | `auto`, `sage`, `flash`, or `sdpa` |
| `precision` | `bf16` or `fp16` |
| `text_precision` | Text encoder precision: `int4`, `int8`, `fp4`, `fp8`, `fp16` |
| `adaptive_offload` | `off`, `normal`, `aggressive` |

> Most users don't need this node — `auto_optimize` handles everything automatically.

### 2.7 QuantFunc Export

Exports all runtime-quantized models to disk (with LoRA fusion support) for instant future loading.

| Parameter | Description |
|-----------|-------------|
| `export_path` | Output directory for exported model |
| `export_models` | Components to export: `transformer`, `text_encoder`, `all` |

## 3. Quick Start

### 3.1 Text-to-Image (SVDQ)

1. Open `QuantFunc-Text-to-Image-Workflow.json` in ComfyUI
2. Set `model_dir` to your base model path
3. Set `transformer_path` to your SVDQ `.safetensors` file
4. Set `model_backend` = `svdq`
5. Enter a prompt and click **Queue Prompt**

### 3.2 Text-to-Image (Lighting)

1. Same workflow, use the **Lighting group** (right side)
2. Set `model_dir` to FP16 base model path
3. Leave `transformer_path` empty (or point to exported pre-quantized weights)
4. Set `model_backend` = `lighting`
5. Set `precision_config` to your precision JSON path
6. Add LoRA nodes as needed (merged at zero cost)

### 3.3 Image Editing

1. Open `QuantFunc-Image-to-Image-Workflow.json`
2. Configure Model Loader (same as text-to-image)
3. Load reference images via `LoadImage` → `QuantFunc Image List`
4. Connect Image List to Generate's `ref_images` input
5. Write an editing prompt (e.g., "Change the sky to sunset")

### 3.4 Model Export

1. Open `QuantFunc-Model-Export.json`
2. Configure Model Loader with Lighting backend + LoRAs
3. Set `export_path` in the Export node
4. Click **Queue Prompt** — model is runtime-quantized and all quantized weights are saved
5. Future runs: load exported model with `transformer_path` for instant startup (no re-quantization)

## 4. Model Download

Pre-quantized models available at:
- **ModelScope**: https://www.modelscope.cn/models/QuantFunc
- **HuggingFace**: https://huggingface.co/QuantFunc

> Base model and transformer weights must use the **same GPU variant** (`50x-below` for RTX 30/40, `50x-above` for RTX 50).
