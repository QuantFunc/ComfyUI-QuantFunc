# QuantFunc ComfyUI Workflows

[中文说明](README_zh.md)

## 1. Overview

This directory ships an **all-in-one workflow** that demonstrates every QuantFunc node, plus **focused per-feature workflows** for the newer generation modes. Import the one that matches your case.

| Workflow | Description |
|----------|-------------|
| `QuantFunc-Sample-WorkFlow-All-In-One.json` | One comprehensive workflow: **3 model-loading methods** × **text-to-image / image editing / model export**. In-canvas notes explain each group. |
| `QuantFunc-Ideogram4.json` | **Ideogram4** text-to-image — *Model Loader* → *Build Pipeline* → *Generate*, with the Ideogram-4 prompt builder for typography-aware prompts. |
| `QuantFunc-QwenImage-Layered.json` | **Layered (transparent RGBA)** generation — *Layered Config* sets the layer count; *Layer Viewer* + *Image List* preview each decomposed RGBA layer. |
| `QuantFunc-ControlNet.json` | **ControlNet** structure-guided generation — *ControlNet Auto Loader* + *Control Image* feed the conditioning into *Generate*. |

> Start from the all-in-one to learn the node graph, then use a per-feature workflow for day-to-day work.

## 2. What's Inside

Drag the canvas to the labelled group you need.

### 2.1 Model loading — three ways

| Method | Nodes | When to use |
|--------|-------|-------------|
| **Existing UNet / CLIP / VAE** | *Pick Diffusion Model* + *Pick CLIP* + *Pick VAE* → **Build Pipeline** | You already have separate component files (the ComfyUI-native wiring). |
| **Base / pre-quantized model** | *Model Loader* or *Model Auto Loader* (one-click download) | A diffusers base-model directory or a pre-quantized model. For a full-precision diffusers base, also attach a *Precision Config (Auto) Loader*. |
| **All-in-one checkpoint** | *Pick Checkpoint* → **Build Pipeline** | A single bundled checkpoint file (AIO). |

> A **Precision Config (Auto) Loader** is required when loading a **full-precision diffusers base model** — it supplies the per-layer precision map. Pre-quantized / checkpoint models don't need it.

### 2.2 Tasks — three samples

- **Text-to-Image** — pipeline → **Generate** → **Preview Image**.
- **Image Editing** — **Load Image** → **Image List** → **Generate** (reference-based / inpaint editing). Attach a `MASK` to the Image List for inpainting (white = regenerate, black = preserve).
- **Export** — the **Export** node. Choose the **checkpoint** format to export an all-in-one bundle (all components), or **diffusers** to export only vae / clip / transformer.

**LoRA:** attach a **LoRA Auto Loader** to any pipeline; chain several to stack LoRAs (zero-cost merge).

## 3. Node Reference (v2)

| Node | Role |
|------|------|
| **Pick Diffusion Model / Pick CLIP / Pick VAE / Pick Checkpoint** | Select component files (UNet / CLIP / VAE) or a single all-in-one checkpoint. |
| **Build Pipeline** | Assembles the picked components into a runnable pipeline, with per-component precision / backend control. |
| **Model Loader / Model Auto Loader** | Load a base or pre-quantized model directory directly (the Auto Loader adds one-click download + dropdown filtering by model series). |
| **Precision Config Loader / Precision Config Auto Loader** | Per-layer precision map — **required for a full-precision diffusers base model**. |
| **LoRA Auto Loader** | Append a LoRA adapter to the pipeline (chainable to stack multiple LoRAs). |
| **Generate** | Run inference — text-to-image, or reference-based editing when an Image List is connected. |
| **Image List** | Bundle 1–N reference image(s) and an optional inpaint mask for editing. |
| **Export** | Export a runtime-quantized model (checkpoint = AIO bundle, diffusers = individual components). |
| **Layered Config** | Set the layer count / options for **QwenImage Layered** transparent-RGBA generation. |
| **Layer Viewer** | Preview each decomposed RGBA layer produced by a layered generation. |
| **ControlNet Auto Loader** | One-click download + load of a ControlNet model (InstantX for QwenImage). |
| **Control Image** | Preprocess / pass a control image (edges, depth, pose, …) into the pipeline. |
| **Wan Combine Experts (A14B two-transformer)** | Combine the two single-file Wan2.2-A14B checkpoints (high-noise + low-noise experts) into one engine-loadable model dir (`model_dir` → *Model Loader*). |
| **Wan Combine Experts (Auto)** | Zero-typing front-end for the above: pick the high / low experts + a Wan shared dir from three dropdowns of your scanned files → outputs **MODEL / CLIP / VAE** straight into *Build Pipeline* (the two-expert combine is done internally + cached; no user-facing `model_dir`). |

## 4. Model Download

Pre-quantized models:
- **ModelScope**: https://www.modelscope.cn/models/QuantFunc
- **HuggingFace**: https://huggingface.co/QuantFunc

> Base model and transformer weights must use the **same GPU variant** (`50x-below` for RTX 30/40, `50x-above` for RTX 50).
