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

## 1. Introduction

ComfyUI plugin for **QuantFunc** — the fastest diffusion model inference engine. Run quantized text-to-image and image editing models at 2x–11x speed with zero Python model dependencies.

**Key features:**
- Native C++/CUDA acceleration via `libquantfunc.so` / `quantfunc.dll`
- QuantFunc loaders for MiniMax-H3, LTX-2.5, Krea-2 and Qwen-Image-2.1 that work with ComfyUI's own text encoder, VAE and
  sampler nodes
- Changing quality, attention, caches or LoRAs reuses the loaded model: only a different model file loads again
- Image editing with reference images (Qwen-Image-2.1)
- The engine installs itself on Linux, checked against the release's published SHA-256 manifest

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

On Linux the plugin **installs its engine by itself** when ComfyUI starts (see [2.5](#25-engine-install-linux)); no
manual download is needed. On Windows, put `quantfunc.dll` in `bin/windows/` (the automatic install is Linux-only in this
release).

### 2.2 Method B: Manual Installation

1. Download or clone this repository into `ComfyUI/custom_nodes/`:

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-QuantFunc/
        ├── __init__.py                the loaders (MiniMax-H3, LTX-2, Krea-2, Qwen-Image-2.1) and QuantFunc Native LoRA
        ├── qf_engine.py               the engine bridge and the engine installer
        ├── qf_*_modelpatcher.py       one per model family
        ├── configs/                   one folder per model preset
        ├── example_workflows/
        └── bin/
            ├── linux/
            │   ├── version.json       this plugin's version: it picks the compatible engine
            │   └── <version>-<class>-cu<major>/   the installed engine (created by the plugin, see 2.5)
            └── windows/
                └── version.json
```

2. Start ComfyUI. On Linux the plugin installs the engine on the first start (see 2.5).

3. To run an engine you built or downloaded yourself instead:
   - **Linux:** put it at `bin/linux/libquantfunc.so` and create an empty `bin/linux/.dev_lib_lock` (see 2.5).
   - **Windows:** put `quantfunc.dll` in `bin/windows/`.

### 2.3 System Requirements

| Requirement | Minimum |
|-------------|---------|
| **GPU** | NVIDIA RTX 20 series or newer (CC 7.5+) |
| **VRAM** | 8 GB |
| **Driver** | NVIDIA ≥ 575 (CUDA 12 engine) or ≥ 580 (CUDA 13 engine) |
| **CUDA Runtime** | 12.9+ or 13.0+ (the engine matches PyTorch's CUDA version) |
| **cuDNN** | 9.x |
| **OS** | Linux (glibc 2.31+) or Windows 10/11 |
| **Python** | 3.9+ (ComfyUI's embedded Python) |

### 2.4 Runtime Dependencies

#### Linux

```bash
# CUDA 12 runtime libraries
sudo apt install cuda-libraries-12-9
# or individual packages:
sudo apt install libcublas-12-9 libcurand-12-9 libcusolver-12-9 libcusparse-12-9 libnvjitlink-12-9

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

- **NVIDIA Driver** ≥ 575 for a CUDA 12 engine, ≥ 580 for CUDA 13 (provides CUDA runtime DLLs)
- **Visual C++ Redistributable** 2015-2022 ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe))
- **cuDNN 9.x** ([download](https://developer.nvidia.com/cudnn))

### 2.5 Engine Install (Linux)

No extra Python package is needed: the installer uses Python's standard library. At every ComfyUI start, in the
background, it:

1. reads the release list on [ModelScope `QuantFunc/Plugin`](https://www.modelscope.cn/models/QuantFunc/Plugin) (HTTPS
   only) and picks the newest engine compatible with this plugin;
2. picks the engine for your setup: the host library for torch's CUDA version (12 or 13), and the kernel library for your
   GPU class, from the release's `sets.json`;
3. downloads both, checks each file's SHA-256 against the release's `verify.json` and that both come from one build, and
   installs them into `bin/linux/<version>-<class>-cu<major>/` — all or nothing.

Before every load the plugin hashes the installed files again: a file that changed on disk is not loaded, and it is
downloaded again. Offline, the installed engine stays in use. Several ComfyUI instances with different GPUs can share one
plugin folder; each uses the engine for its own GPU.

`verify.json` proves the files are the ones published in that same ModelScope repository. It is an integrity check, not a
signature: it does not protect against the repository itself being changed.

**Your own engine build:** put it at `bin/linux/libquantfunc.so` and create an empty `bin/linux/.dev_lib_lock`. The plugin
then loads exactly that file, and the installer does not download or change anything (one console line says so). Delete
the marker to go back to the installed engine. Without the marker a file there is not used: it is usually a copy left by
an earlier plugin version. (Developers can also point `QF_NATIVE_SO_PATH` at a library; the installer then keeps out too.)

### 2.6 Verify Installation

After ComfyUI starts, the console shows one of these lines when the installer installs, cannot install, fails or keeps
out; a start whose engine is already current (or, outside Linux, whose engine library is in place) prints nothing:

```
[qf_native] installed QuantFunc engine <version> for <class> GPUs, CUDA <major>
[qf_native] QuantFunc engine not installed: <why this machine cannot take one>
[qf_native] QuantFunc engine update failed (<reason>); the installed engine <version> stays in use
[qf_native] QuantFunc engine update failed (<reason>); no engine is installed
[qf_native] QuantFunc engine install skipped: bin/linux/.dev_lib_lock keeps the local build bin/linux/libquantfunc.so
```

A loader run during the first download stops with "still downloading": queue the prompt again after the `installed` line.

## 3. Usage

### Quick Start

Put a QuantFunc model file in ComfyUI's `models/diffusion_models/`, add the QuantFunc loader for it (**QuantFunc
MiniMax-H3**, **LTX-2**, **Krea-2** or **Qwen-Image-2.1 Loader**) where ComfyUI's diffusion-model loader would go, pick
the file and its `model_config` preset, and keep ComfyUI's own text encoder, VAE and sampler nodes. The Qwen-Image-2.1
workflows in [`example_workflows/`](example_workflows/) show the wiring.

### 3.1 QuantFunc Models

QuantFunc's 4-bit models are on [ModelScope](https://www.modelscope.cn/models/QuantFunc) and
[HuggingFace](https://huggingface.co/QuantFunc).

### 3.2 LoRA Format Conversion (Native Loaders)

The native loaders (Krea-2 / Qwen-Image-2.1 / LTX-2 / MiniMax-H3) adapt ONE LoRA format — diffusers/PEFT
canonical. Convert kohya / ai-toolkit LoRAs once with the bundled pure-Python tool (no
torch, no GPU, lossless byte-copy):

```bash
python3 scripts/qf_lora_convert.py --in my_kohya_lora.safetensors --out my_lora-diff.safetensors
```

> **[LoRA Format Converter →](doc/lora-convert.md)**

### 3.3 Example Workflows

Qwen-Image-2.1 native-loader workflows are in [`example_workflows/`](example_workflows/) — ComfyUI lists them under
**Templates → ComfyUI-QuantFunc**. They use the stock `CLIPLoader` (type `qwen_image`), `TextEncodeQwenImage21`, VAE and
`KSampler`; only the transformer loader is QuantFunc's. The VAE is RGBA, so `VAE Decode` + `Save Image` keep transparency.
The loader's `quality` choice (default `balance` on the MiniMax-H3, LTX-2, Krea-2 and Qwen-Image-2.1 loaders): `balance`
can be a little faster than `best_quality`; the subject and scene stay the same, but details such as poses, faces or small
objects can differ from `best_quality`. On GPUs that offer them, `fast` and `super_fast` are usually faster still and can
give a different variation of the same seed. `best_quality` gives the highest quality. The LTX-2 loader never offers
`super_fast` (a workflow saved with it runs `fast`, or `balance` on a GPU without `fast`). On GPUs without
`fast` and `super_fast`, the Qwen-Image-2.1 loader shows no `quality` choice and always gives the highest quality.

| File | Use Case |
|------|----------|
| `QuantFunc-QwenImage21-t2i.json` | text-to-image |
| `QuantFunc-QwenImage21-t2i-transparent.json` | text-to-image with a transparent background (RGBA PNG) |
| `QuantFunc-QwenImage21-edit.json` | image edit, one reference (`<image1>`) |
| `QuantFunc-QwenImage21-edit-multi-reference.json` | image edit, two references (the official example) |
| `QuantFunc-QwenImage21-edit-remove-background.json` | remove the background → transparent PNG |

## 4. Troubleshooting

| Issue | Solution |
|-------|----------|
| "no QuantFunc engine is installed" / "still downloading" | Linux: the console's `[qf_native]` line from the start says why (still downloading, not installable here, update failed); see 2.5 and 2.6 |
| Engine library not found (Windows) | Put `quantfunc.dll` in `bin/windows/` |
| Console shows only warnings | That is the default: the engine prints only warnings and errors |
| cuDNN BAD_PARAM | Delete cuDNN algo cache and retry |
| The engine cannot be downloaded (offline) | Put an engine build at `bin/linux/libquantfunc.so` with an empty `bin/linux/.dev_lib_lock` (2.5) |

## 5. License

See [QuantFunc Plugin License](https://www.modelscope.cn/models/QuantFunc/Plugin).

## Community

Join our community for support, updates, and discussions:

- 🎮 [Discord server](https://discord.gg/jCp9TpFWcn)
- 💬 Scan the QR code below to join our WeChat group:

<div align="center" id="wechat">
  <img src="assets/WeChat.jpg" alt="WeChat Group" width="300">
</div>
