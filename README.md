<div align="center" style="margin-top: 50px;">
  <img src="https://raw.githubusercontent.com/QuantFunc/ComfyUI-QuantFunc/main/assets/logo.webp" width="300" alt="QuantFunc Logo">
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
- Changing quality_enhance, attention, caches or LoRAs reuses the loaded model: only a different model file, or the
  pinned_memory switch, loads it again
- Image editing with reference images (Qwen-Image-2.1)
- The engine installs itself on Linux and Windows, checked against the release's published SHA-256 manifest

## Quick Start

1. Clone into your ComfyUI custom nodes folder:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/QuantFunc/ComfyUI-QuantFunc.git
   ```

2. Restart ComfyUI. The plugin installs the compatible engine for your system, CUDA version and GPU architecture.
3. Install the required models and import an [example workflow](example_workflows/). For image-conditioned examples, copy the [included reference images](example_workflows/assets/README.md) into `ComfyUI/input`.

**Requirements:** NVIDIA GPU, Linux or Windows, and a compatible CUDA 12/13 environment. See the [installation and usage guide](docs/USER_GUIDE.md) for supported GPUs, dependencies, manual installation and troubleshooting.

## MiniMax-H3 Workflows

<table>
  <tr>
    <td align="center"><img src="example_workflows/assets/minimax_h3_fl2va_beach_reference.png" width="280" alt="FL2VA beach reference"><br><strong>First / last frame</strong></td>
    <td align="center"><img src="example_workflows/assets/ComfyUI_temp_tupev_00031_.png" width="280" alt="Ref2VA armor reference"><br><strong>Reference image</strong></td>
  </tr>
</table>

Reference images included with the example workflows.

| Workflow | Use |
|---|---|
| [FL2VA](example_workflows/QuantFunc-MiniMaxH3-fl2va.json) | First/last-frame video with audio |
| [FL2VA · double sampling](example_workflows/QuantFunc-MiniMaxH3-fl2va-double-sampling.json) | Two-stage generation and latent upscale; image inputs are bypassed by default |
| [Ref2VA](example_workflows/QuantFunc-MiniMaxH3-ref2va.json) | Reference-image video with audio |
| [Ref2VA · double sampling](example_workflows/QuantFunc-MiniMaxH3-ref2va-double-sampling.json) | Reference-image generation with two-stage sampling and latent upscale |

[All example workflows →](example_workflows/)

## Recent Updates

### Engine 0.0.18 · Plugin 0.0.08

- **PC Engine architecture:** refactored to a standalone single-Transformer engine with a zero-CUDA host library (`libquantfunc.so` / `quantfunc.dll`) paired with single-architecture CUDA 12/13 kernel plugins (`sm75`, `sm86`, `sm89`, `sm120a`) across Linux and Windows, with SHA-256 verification and strict `-12` pairing.
- **Blackwell (SM120) quality & 8 GB VRAM stability:** fixed diagonal blur artifacts on RTX 50-series GPUs (MiniMax-H3) while preserving full throughput and 8 GB VRAM headroom across MiniMax-H3, LTX-2.5, Krea-2 and Qwen-Image-2.1.
- **Built-in attention tiers & faster startup/loading:** exposed `auto`, `int8`, `i4xi8`, `int4`, `fp16` across all four loaders, with faster model scanning, cached VRAM estimates, prefetching, and zero-sync stream handoff.
- **Bug fixes & workflow updates:** fixed `QF:PL_PERM` / clone 2x VRAM inflation / cold autoload / MiniMax-H3 3D audio packing / Qwen-Image-2.1 `CFG=1.0` & RGBA transparent alpha issues; unified all 12 example workflows to `res_multistep` with `sol_min_tokens=0` on LTX-2.5 and MiniMax-H3.

### Engine 0.0.17 · Plugin 0.0.07

- **Built-in attention backends:** architecture-matched attention backends (`auto`, `int8`, `i4xi8`, `int4`, `fp16`) for Linux and Windows.
- **SM75 performance:** faster inference on Turing GPUs.
- **VRAM management:** more efficient memory use and model movement for faster inference.
- **MiniMax-H3 workflows:** updated single- and double-sampling examples for better image quality and speed.

### Engine 0.0.13 · Plugin 0.0.07

- Added quality-enhancement and pinned-memory controls.
- Reuse loaded models when changing settings or LoRAs.
- Architecture-specific engines, simpler LoRA handling and quieter logging.

## Community

[Discord](https://discord.gg/jCp9TpFWcn) · [Website](https://www.quantfunc.com/) · [License](https://www.modelscope.cn/models/QuantFunc/Plugin)

<div align="center" id="wechat">
  <img src="assets/WeChat.jpg" width="220" alt="WeChat Group">
</div>
