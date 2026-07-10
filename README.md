<div align="center" style="margin-top: 50px;">
  <img src="https://raw.githubusercontent.com/RealJonathanYip/ComfyUI-QuantFunc/main/assets/logo.webp" width="300" alt="QuantFunc Logo">
</div>

<p align="center">
  🌐 <a href="https://www.quantfunc.com/">Website</a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/QuantFunc">Hugging Face</a> &nbsp;|&nbsp;
  🤖 <a href="https://www.modelscope.cn/profile/QuantFunc">ModelScope</a> &nbsp;|&nbsp;
  💬 <a href="#wechat">WeChat (微信)</a> &nbsp;|&nbsp;
  🎮 <a href="https://discord.gg/jCp9TpFWcn">Discord</a>
</p>

# ComfyUI-QuantFunc

[中文说明](README_zh.md)

## 1. Introduction

ComfyUI plugin for **QuantFunc** — the fastest diffusion model inference engine. Run quantized text-to-image and image editing models at 2x–11x speed with zero Python model dependencies.

**Key features:**
- Native C++/CUDA acceleration via `libquantfunc.so` / `quantfunc.dll`
- SVDQ (offline quantization) + Lighting (runtime quantization) dual engine
- Zero-cost LoRA stacking
- Image editing with reference images
- Export runtime-quantized models with LoRA fusion support
- Auto-update from ModelScope

## Version History

| Plugin (`comfy`) | Engine (`lib`) | Summary |
|:---:|:---:|---|
| **0.0.06** *(current)* | **0.0.12** | New modes — Ideogram4 · Layered (RGBA) · ControlNet · img2img · VRAM-budget · FBCache · Klein auto-download · SHA-256 integrity check — details below |
| 0.0.02 | 0.0.07 | v2 loader architecture · inpainting · full GPU coverage · faster editing |
| 0.0.01 | 0.0.01 – 0.0.06 | Base release: runtime/offline quantization · model & LoRA loaders · reference-image editing · export · auto-update |

### What's New in 0.0.06 (engine 0.0.12)

**🧩 New generation modes** — each ships with a ready-to-run workflow in [`workflow_sample/`](workflow_sample/):
- **Ideogram4 text-to-image** — strong prompt & text/typography following; renders 1024² / 1536² / 2048² and runs on 8 GB VRAM (`QuantFunc-Ideogram4.json`).
- **Layered generation (QwenImage Layered)** — generate a scene decomposed into transparent **RGBA layers** in a single pass, accelerated by FBCache (`QuantFunc-QwenImage-Layered.json`).
- **ControlNet** — structure-guided generation (InstantX ControlNet for QwenImage) (`QuantFunc-ControlNet.json`).

**🎛️ New nodes & controls**
- **Wan Combine Experts (A14B two-transformer)** — combine the two single-file Wan2.2-A14B checkpoints (high-noise + low-noise experts) into one engine-loadable model dir: keys remapped to diffusers naming, fp8 dequantized to fp16, `boundary_ratio` inherited from the published model_index (t2v 0.875 / i2v 0.9). Wire its `model_dir` output into *Model Loader*. The ~56 GB stage caches per source set — set `QUANTFUNC_CACHE_DIR` (or the node's `output_dir`) to persist it across ComfyUI restarts.
- **Wan Combine Experts (Auto)** — a zero-typing front-end for the above: scans the ComfyUI model dirs (`diffusion_models`/`unet`/`checkpoints`/`diffusers`, recursively) + the QuantFunc model_cache and offers **three dropdowns** — `high_noise_expert` / `low_noise_expert` (your Wan single-file experts; the TI2V-5B is excluded) + `shared_components` (a Wan diffusers dir for vae / text_encoder / tokenizer / scheduler, A14B-family listed first) — plus `boundary_ratio` (0 = AUTO). It outputs **MODEL / CLIP / VAE** that wire **straight into `QuantFunc Build Pipeline`** (no `model_dir`, no separate Model Loader): the two experts are combined into the engine's two-transformer layout **internally and cached** — you never see a staging dir. (A ready diffusers A14B *directory* is already engine-loadable, so load it with the plain `QuantFunc Model Loader` instead.) Reopen the graph to rescan after adding files. *Note: the plugin↔staged-dir handoff is verified; the end-to-end two-expert video generation is your local ComfyUI GPU run.*
- **Wan single-file TI2V-5B (trio)** — load the ComfyUI single-file Wan2.2 TI2V-5B directly: wire **UNETLoader** (`wan2.2_ti2v_5B_*.safetensors`) + **CLIPLoader** (`umt5_xxl_*.safetensors`, fp16 or fp8-scaled) + **VAELoader** (`wan2.2_vae.safetensors`) straight into `QuantFunc Build Pipeline` — keys are remapped to diffusers naming and fp8 dequantized automatically (same machinery as Combine Experts), staged into a session cache under ComfyUI's temp dir. A 16-channel single file (an A14B expert) is refused with guidance — use *Wan Combine Experts* for the A14B pair.
- **`tiny_vae` fast-preview toggle** on *Build Pipeline* (Wan video only, **default OFF**) — swaps the full Wan VAE decoder for the tiny TAEHV decoder: much faster video decode (measured ~5× on Wan2.1/A14B @384×384 up to ~28× on Wan2.2-5B), output stays coherent but is **softer** — use for drafts/previews, disable for final quality. The taew variant is picked automatically (Wan2.1/A14B → `taew2_1`, Wan2.2-5B → `taew2_2`); place the weights at `<ComfyUI>/models/QuantFunc/taew/<variant>.safetensors` (missing file / non-Wan model / an engine build that predates tiny-VAE support all fail loud instead of silently falling back).
- **VRAM budget** dropdown on *Build Pipeline* — cap VRAM at create time; the engine then plans as if running on a smaller card.
- **FBCache acceleration** with separate **cond / uncond** thresholds (`fbcache` / `fbcache_uncond`).
- **Image-to-image** — `init_img` socket + `init_img_strength` on the *Generate* node.
- **Klein 4B / 9B** one-click auto-download (3-tier 50x / 40x / 30x-below) in the *Auto Loader*.
- **Ideogram-4 + Qwen-Image-Layered** precision configs auto-detected by the auto-loader.

**🔒 Engine integrity check (new)**
- On startup the plugin verifies the installed engine library's **SHA-256** against the official manifest published on ModelScope (`<version>/verify.json`), re-fetched on every launch (the local cache is used only when the network is unreachable). A corrupt, incomplete, or wrong-build binary **self-heals** — the official artifact is re-downloaded and swapped in **only if its hash matches** (verify-before-replace, so a bad download never replaces a working library). It never blocks node loading, and a locally-compiled build is left untouched.

**⚡ Engine improvements (0.0.12 vs 0.0.11)**
- **VRAM workspace budget** — run large models on tight cards by planning to a fixed budget.
- **Wider low-VRAM & RTX 20 (Turing) support** — Ideogram4 and layered generation now run on 8 GB and SM75.
- **RTX 50 (Blackwell) FP4 fast lane** for the new pipelines.

> The plugin auto-pulls the matching engine on startup: bumping `comfy` to **0.0.06** lets the updater fetch engine **0.0.12** from ModelScope.

### What's New in 0.0.02 (engine 0.0.07)

**🎯 Ease of Use**
- **v2 loaders** — separate `MODEL` / `CLIP` / `VAE` sockets feed a **Build Pipeline** node, so models wire up the ComfyUI-native way instead of one monolithic loader.
- **Universal format adapters** — load **diffusers / BFL (Flux) / nunchaku SVDQ / bundled-checkpoint / HF** layouts automatically, with no manual conversion.
- **Base Model Auto Loader** with one-click download; the plugin also auto-pulls the matching engine on first startup.

**🧩 Model Support**
- **SVDQ** (offline quantization) **+ Lighting** (runtime BF16/FP16 → 4-bit) dual engine.
- Pipelines: **Z-Image · QwenImage · QwenImage-Edit · Flux.2 Klein**.
- **Full GPU coverage** (engine 0.0.07): consumer **RTX 20 / 30 / 40 / 50-series**, datacenter **A100 / H100 / H200 / B100 / B200 / GB300**, workstation **RTX 6000 Ada / RTX PRO 6000 Blackwell** — across **CUDA 12 & 13**.

**⚡ Performance**
- **Consumer GPUs run native SASS** — *no first-run JIT compile stall* on 20/30/40/50-series (datacenter/workstation cards JIT once, then cache).
- Native **FP4 (NVFP4)** on Blackwell (SM120) — the fastest 4-bit path.
- **QFRAW raw staging** for reference images & masks skips the PNG/BMP encode (~80 ms saved per ref).
- **Multi-pipeline CPU↔GPU coexistence** — swap pipelines without a full reload; idle workers auto-free VRAM.

**✨ New Features**
- **Inpainting** — `MASK` input plus **Mask Config** and **Mask Scale By** nodes (white = regenerate, black = preserve), mirroring ComfyUI's SetLatentNoiseMask.
- **Build Pipeline** node (v2 assembly) with per-component precision control.
- Robust **worker-process architecture** — CPU↔GPU model swap + zombie-worker cleanup.

**🛡️ Stability & Security**
- Fixed a **`/dev/shm` RAM leak** — edit/inpaint staging files are now always cleaned up.
- **Zip-slip guard** on dependency-archive extraction.
- **IPC bound-check** on the worker → host image transfer.

> The plugin auto-pulls the matching engine on startup: bumping `comfy` to **0.0.02** lets the updater fetch engine **0.0.07** from ModelScope (older `comfy` stays capped at engine 0.0.06).

## 2. Installation

### 2.1 Method A: Clone from Git (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RealJonathanYip/ComfyUI-QuantFunc.git
```

The plugin will **automatically download** the latest compatible `libquantfunc.so` (Linux) or `quantfunc.dll` (Windows) from ModelScope on first startup. No manual binary download needed.

### 2.2 Method B: Manual Installation

1. Download or clone this repository into `ComfyUI/custom_nodes/`:

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-QuantFunc/
        ├── __init__.py
        ├── nodes.py
        ├── worker.py
        ├── auto_update.py
        └── bin/
            ├── linux/
            │   └── version.json
            └── windows/
                └── version.json
```

2. Start ComfyUI — the plugin auto-downloads the library binary on first run.

3. (Optional) To skip auto-download, manually place the binary:
   - **Linux:** Download `libquantfunc.so` → `bin/linux/`
   - **Windows:** Download `quantfunc.dll` → `bin/windows/`

### 2.3 System Requirements

| Requirement | Minimum |
|-------------|---------|
| **GPU** | NVIDIA RTX 20 series or newer (CC 7.5+) |
| **VRAM** | 8 GB |
| **Driver** | NVIDIA ≥ 560 |
| **CUDA Runtime** | 13.0+ |
| **cuDNN** | 9.x |
| **OS** | Linux (glibc 2.31+) or Windows 10/11 |
| **Python** | 3.9+ (ComfyUI's embedded Python) |

### 2.4 Runtime Dependencies

#### Linux

```bash
# CUDA 12 runtime libraries
sudo apt install cuda-libraries-12-8
# or individual packages:
sudo apt install libcublas-12-8 libcurand-12-8 libcusolver-12-8 libcusparse-12-8 libnvjitlink-12-8

# cuDNN 9
sudo apt install libcudnn9-cuda-12

# --- OR ---

# CUDA 13 runtime libraries
sudo apt install cuda-libraries-13-0
# or individual packages:
sudo apt install libcublas-13-0 libcurand-13-0 libcusolver-13-0 libcusparse-13-0 libnvjitlink-13-0

# cuDNN 9
sudo apt install libcudnn9-cuda-13
```

#### Windows

- **NVIDIA Driver** ≥ 560 (provides CUDA runtime DLLs)
- **Visual C++ Redistributable** 2015-2022 ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe))
- **cuDNN 9.x** ([download](https://developer.nvidia.com/cudnn))

### 2.5 ModelScope Dependency (for auto-update)

Auto-update requires `modelscope` Python package:

```bash
pip install modelscope
```

If `modelscope` is not installed, auto-update is silently skipped. You can manually download binaries from:
- https://www.modelscope.cn/models/QuantFunc/Plugin

To keep a locally-built engine library, create an empty `bin/<platform>/.dev_lib_lock`.
Auto-update then skips its integrity check, which would otherwise see the SHA
mismatch and re-download the release library over your build. Delete the marker to
restore normal updating.

### 2.6 Verify Installation

After starting ComfyUI, check the console for:

```
[QuantFunc] Checking for updates (plugin v0.0.01, lib v0.0.01)...
[QuantFunc] Library is up to date (v0.0.01)
```

If the library was not found:

```
[QuantFunc] No library found, checking ModelScope for download (plugin v0.0.01)...
[QuantFunc] Downloading libquantfunc.so v0.0.01 from ModelScope...
[QuantFunc] Updated libquantfunc.so to v0.0.01. Restart ComfyUI to use the new version.
```

## 3. Usage

See [doc/](doc/) for detailed tutorials and [workflow_sample/README.md](workflow_sample/README.md) for node reference.

### Quick Start for Beginners

The easiest way to get started — add a **Model Auto Loader**, pick a model series from the dropdown, wire it into **Build Pipeline → Generate**, and the plugin auto-downloads everything. No manual model downloads or path configuration needed.

> **[Quick Start & documentation index →](doc/README.md)**

### 3.1 Runtime Quantization: Quantize BF16/FP16 Models to 4bit for Accelerated Inference

The **Lighting backend** provides **runtime quantization** — it quantizes any diffusers-format BF16/FP16 model (e.g., [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)) to 4bit at load time for accelerated inference. Just point **Model Loader** at the FP16 model and leave `transformer_path` empty; the backend is auto-detected — no pre-quantized model download needed.

> **[Model Loading & Runtime Quantization →](doc/model-loading-and-apikey_zh.md)** (Chinese)

### 3.2 Export Runtime-Quantized Models (with LoRA Fusion Support)

The Lighting export saves all runtime-quantized models to disk, so you don't need to re-quantize on every startup. If you've also stacked LoRAs, they are permanently fused into the exported weights — no LoRA nodes needed, no re-quantization, load and go.

> **[Export Quantized Models →](doc/export-quantized-models.md)**

### 3.3 Download and Use Pre-exported Quantized Models

QuantFunc has pre-exported commonly used models (runtime-quantized and ready to use). Download them directly from [ModelScope](https://www.modelscope.cn/models/QuantFunc) or [HuggingFace](https://huggingface.co/QuantFunc) — same 2x–11x inference speedup as runtime quantization, but with faster loading since the quantization step is skipped.

> **[Model Loading & Downloads →](doc/model-loading-and-apikey_zh.md)** (Chinese)

### 3.4 Example Workflows

Import from [`workflow_sample/`](workflow_sample/):

| File | Use Case |
|------|----------|
| `QuantFunc-Sample-WorkFlow-All-In-One.json` | **All-in-one** — every node × 3 model-loading methods × text-to-image / editing / export |
| `QuantFunc-Ideogram4.json` | Ideogram4 text-to-image with prompt builder |
| `QuantFunc-QwenImage-Layered.json` | Layered (transparent RGBA) generation + layer viewer |
| `QuantFunc-ControlNet.json` | ControlNet structure-guided generation |

## 4. Troubleshooting

| Issue | Solution |
|-------|----------|
| Worker failed to start | Check CUDA driver ≥ 560, ensure CUDA runtime libs installed |
| DLL/SO not found | Check `bin/linux/` or `bin/windows/` contains the library; restart ComfyUI to trigger auto-download |
| No log output | Update to latest library version (requires stderr log support) |
| cuDNN BAD_PARAM | Delete cuDNN algo cache and retry |
| Noisy output | Ensure model backend matches transformer weights (svdq vs lighting) |
| Auto-update fails | Install `modelscope` package, or manually download from ModelScope |

## 5. License

See [QuantFunc Plugin License](https://www.modelscope.cn/models/QuantFunc/Plugin).

## Community

Join our community for support, updates, and discussions:

- 🎮 [Discord server](https://discord.gg/jCp9TpFWcn)
- 💬 Scan the QR code below to join our WeChat group:

<div align="center" id="wechat">
  <img src="assets/WeChat.jpg" alt="WeChat Group" width="300">
</div>
