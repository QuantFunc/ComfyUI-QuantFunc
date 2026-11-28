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

### Engine 0.0.19 · Plugin 0.0.09

- **Clean shutdown & pinned host-memory safety:** added `atexit` and GC cleanup for cached and pending native handles so CUDA resources and pinned host memory release cleanly on exit, and bounded large Linux pinned host allocations into chunks to eliminate multi-GiB host memory retention across repeated LTX-2.5 and MiniMax-H3 runs.
- **Backward compatibility:** fully compatible with existing Plugin `0.0.08` installs (which automatically receive Engine `0.0.19`).

### Engine 0.0.18 · Plugin 0.0.08

- **Compact engine download:** shrunk per-GPU runtime libraries from 95–174 MiB (`0.0.17`) down to 4.0–5.5 MiB on Linux and 14.0–28.3 MiB on Windows (`14.0–17.6 MiB` on RTX 40/50 series).
- **Faster startup & loading:** ~0.19 s cold init and 0.54–1.49 s NVMe model loads (down from 13.3–23.5 s on MiniMax-H3 in `0.0.17`; 2.18× faster than ComfyUI's native loader on LTX-2.5), plus 1.8 ms LoRA hot-swap.
- **Linear & attention speedup + accuracy:** 1.15×–1.56× median faster than `ComfyUI-Kitchen` (`int8_convrot`); built-in attention is 1.61×–5.31× median faster than PyTorch SDPA with `0.9989–0.9999` (`int8`) / `0.9878–0.9990` (`i4xi8`) / `0.9506–0.9900` (`int4`) cosine similarity.
- **Blackwell (SM120) & 8 GB VRAM stability:** fixed RTX 50-series diagonal blur on MiniMax-H3 (visual parity with RTX 4090) with 0 OOM across 8 GB GPUs.
- **Bug fixes & workflows:** added LTX-2.5 `ref_audio` and per-frame video/audio `denoise_mask` support; fixed Windows locked-DLL updates, clone VRAM inflation, H3 audio packing, and Qwen-Image-2.1 `CFG=1.0` / RGBA alpha; unified all 12 workflows to `res_multistep`.

### Engine 0.0.17 · Plugin 0.0.07

- **Built-in attention backends:** architecture-matched attention backends (`auto`, `int8`, `i4xi8`, `int4`, `fp16`) for Linux and Windows.
- **SM75 performance:** faster inference on Turing GPUs.
- **VRAM management:** more efficient memory use and model movement for faster inference.
- **MiniMax-H3 workflows:** updated single- and double-sampling examples for better image quality and speed.

## Community

[Discord](https://discord.gg/jCp9TpFWcn) · [Website](https://www.quantfunc.com/) · [License](https://www.modelscope.cn/models/QuantFunc/Plugin)

<div align="center" id="wechat">
  <img src="assets/WeChat.jpg" width="220" alt="WeChat Group">
</div>
