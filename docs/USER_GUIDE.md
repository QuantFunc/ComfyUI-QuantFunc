# Installation and Usage

[Back to the homepage](../README.md)

## 2. Installation

### 2.1 Method A: Clone from Git (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/QuantFunc/ComfyUI-QuantFunc.git
```

On Linux and Windows the plugin **installs its engine by itself** when ComfyUI starts (see
[2.5](#25-engine-install-linux-and-windows)); no manual download is needed.

### 2.2 Method B: Manual Installation

1. Download or clone this repository into `ComfyUI/custom_nodes/`:

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-QuantFunc/
        ├── __init__.py                the loaders (MiniMax-H3, LTX-2, Krea-2, Qwen-Image-2.1), QuantFunc Native LoRA and QuantFunc MiniMax-H3 Latent Upscale
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

2. Start ComfyUI. The plugin installs the engine on the first start (see 2.5).

3. To run an engine you built or downloaded yourself instead:
   - **Linux:** put it at `bin/linux/libquantfunc.so` and create an empty `bin/linux/.dev_lib_lock` (see 2.5).
   - **Windows:** put it at `bin/windows/quantfunc.dll` and create an empty `bin/windows/.dev_lib_lock` (see 2.5).

### 2.3 System Requirements

| Requirement | Minimum |
|-------------|---------|
| **GPU** | NVIDIA, on an x86_64 host. Linux: compute capability 7.5 / 8.0 / 8.6 / 8.9 / 9.0 / 10.0 / 10.3 / 12.0 (RTX 20/30/40/50, T4, A100, A10/A40, L4/L40, H100/H200, B200, B300, RTX PRO 6000 Blackwell). Windows: 7.5 / 8.6 / 8.9 / 12.0 (RTX 20/30/40/50, T4, A10/A40, L4/L40, RTX PRO 6000 Blackwell). Not published: 8.7, 11.0, 12.1 and Grace (aarch64) systems such as GB200/GB300 |
| **VRAM** | 8 GB |
| **Driver** | NVIDIA ≥ 575 (CUDA 12 engine) or ≥ 580 (CUDA 13 engine) |
| **CUDA Runtime** | PyTorch built for CUDA 12.6 or newer (CUDA 12 engine) or CUDA 13.0 or newer (CUDA 13 engine); the engine matches PyTorch's CUDA major |
| **cuDNN** | 9.x |
| **OS** | Linux (glibc 2.31+) or Windows 10/11 |
| **Python** | 3.9+ (ComfyUI's embedded Python) |
| **Engine build** (0.0.13, Linux) | CUDA 13.0 + cuDNN 9.13 (CUDA 13 engine), CUDA 12.9.2 + cuDNN 9.3 (CUDA 12 engine); gcc 11.4; OpenCV / OpenSSL / curl linked statically |

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

### 2.5 Engine Install (Linux and Windows)

No extra Python package is needed: the installer uses Python's standard library. At every ComfyUI start, in the
background, it:

1. reads the release list on [ModelScope `QuantFunc/Plugin`](https://www.modelscope.cn/models/QuantFunc/Plugin) (HTTPS
   only) and picks the newest engine compatible with this plugin;
2. picks the engine for your setup, for torch's CUDA version (12 or 13) and your GPU's architecture (the GPU ComfyUI runs
   on), from the release's `sets.json`. An engine holds exactly one architecture; an architecture the release does not
   publish is refused with its SM number, never given another's. On Linux that is a host library plus the kernel library
   of your architecture; on Windows one DLL;
3. downloads it, checks each file's SHA-256 against the release's `verify.json` (on Linux also that host and kernel come
   from one build), and installs it into `bin/<linux|windows>/<version>-<architecture>-cu<major>/` — all or nothing.
   A new version goes into a new folder. A version republished under the same number is downloaded again into
   the same folder and replaces the files there; on Windows, while a second running ComfyUI still has that DLL
   loaded, the replacement has to wait until that instance exits.

Before every load the plugin hashes the installed files again: a file that changed on disk is not loaded, and it is
downloaded again. Offline, the installed engine stays in use. Several ComfyUI instances with different GPUs can share one
plugin folder; each uses the engine for its own GPU. One ComfyUI serves one GPU architecture: a pipeline on a GPU of
another architecture is refused. To use both, run one ComfyUI per GPU architecture, each started with
`CUDA_VISIBLE_DEVICES` set to its GPU.

`verify.json` proves the files are the ones published in that same ModelScope repository. It is an integrity check, not a
signature: it does not protect against the repository itself being changed.

**Your own engine build:** put it at `bin/linux/libquantfunc.so` (Windows: `bin/windows/quantfunc.dll`) and create an
empty `.dev_lib_lock` in the same folder. The plugin then loads exactly that file, and the installer does not download or
change anything (one console line says so). Delete the marker to go back to the installed engine. Without the marker a
file there is not used: it is usually a copy left by an earlier plugin version. (Developers can also point
`QF_NATIVE_SO_PATH` at a library; the installer then keeps out too.)

### 2.6 Verify Installation

After ComfyUI starts, the console shows one of these lines when the installer installs, cannot install, fails or keeps
out; a start whose engine is already current prints nothing:

```
[qf_native] installed QuantFunc engine <version> for <class> GPUs, CUDA <major>
[qf_native] QuantFunc engine not installed: <why this machine cannot take one>
[qf_native] QuantFunc engine update failed (<reason>); the installed engine <version> stays in use
[qf_native] QuantFunc engine update failed (<reason>); no engine is installed
[qf_native] QuantFunc engine install skipped: bin/linux/.dev_lib_lock keeps the local build bin/linux/libquantfunc.so
```

A loader run during the first download stops with "still downloading": queue the prompt again after the `installed` line.

### 2.7 Upgrading from 0.0.05 / 0.0.06

- The old node set is retired: `QuantFuncGenerate`, `QuantFuncModelLoader`, `QuantFuncBuildPipeline` and the
  format-adapter nodes. Each model family now has its own loader (**QuantFunc MiniMax-H3**, **LTX-2**, **Krea-2** and
  **Qwen-Image-2.1 Loader**), which feeds ComfyUI's stock **KSampler** and ComfyUI's own text encoder and VAE nodes.
- A workflow saved with the old nodes opens with those nodes missing: replace them with the loader for its model and a
  stock KSampler. The workflows in [`example_workflows/`](../example_workflows/) show the wiring.
- The first ComfyUI start after the automatic update can show no QuantFunc nodes at all. Restart ComfyUI once more.

## 3. Usage

### Quick Start

Put a QuantFunc model file in ComfyUI's `models/diffusion_models/`, add the QuantFunc loader for it (**QuantFunc
MiniMax-H3**, **LTX-2**, **Krea-2** or **Qwen-Image-2.1 Loader**) where ComfyUI's diffusion-model loader would go, pick
the file, and keep ComfyUI's own text encoder, VAE and sampler nodes. The Qwen-Image-2.1 workflows in
[`example_workflows/`](../example_workflows/) show the wiring.

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

> **[LoRA Format Converter →](../doc/lora-convert.md)**

### 3.3 Example Workflows

Qwen-Image-2.1 native-loader workflows are in [`example_workflows/`](../example_workflows/) — ComfyUI lists them under
**Templates → ComfyUI-QuantFunc**. They use the stock `CLIPLoader` (type `qwen_image`), `TextEncodeQwenImage21`, VAE and
`KSampler`; only the transformer loader is QuantFunc's. The VAE is RGBA, so `VAE Decode` + `Save Image` keep transparency.
The loaders' `quality_enhance` switch (MiniMax-H3, LTX-2, Krea-2 and Qwen-Image-2.1; default OFF): OFF is faster; the
subject and scene stay the same, but details such as poses, faces or small objects can differ. ON gives the highest
quality. Changing it never reloads the model.
The loaders' `pinned_memory` switch: ON is faster on graphics cards with little VRAM. It is ON by default on the LTX-2.5
loader (LTX-2.5 moves the most model data on such cards) and OFF by default on the MiniMax-H3, Krea-2 and Qwen-Image-2.1
loaders. When enough RAM is free, the model is kept in locked system memory, which the rest of the PC cannot use while
the model is loaded; on a PC with little RAM this can make the system unstable. Changing it reloads the model. It applies
to the whole ComfyUI session: once any loader has turned it on (the LTX-2.5 loader does by default), it stays on for every
model until ComfyUI restarts.
The MiniMax-H3 and LTX-2.5 loaders accept two-stage (double-sampling) workflows as they are. The MiniMax-H3
`audio_enhance` switch is not supported in two-stage (double-sampling) workflows: it is ignored there.
Several loaders that name the same transformer file share one copy of the model in memory. `quality_enhance`,
`attention_backend`, `sol_tau`, `step_cache`, `block_cache` and `audio_enhance` apply per loader, at each run, so a
two-pass workflow can use a fast loader for the first pass and a quality loader for the second without loading the model
twice. A different transformer file or `pinned_memory` setting loads a second copy.

| File | Use Case |
|------|----------|
| `QuantFunc-QwenImage21-t2i.json` | text-to-image |
| `QuantFunc-QwenImage21-t2i-transparent.json` | text-to-image with a transparent background (RGBA PNG) |
| `QuantFunc-QwenImage21-edit.json` | image edit, one reference (`<image1>`) |
| `QuantFunc-QwenImage21-edit-multi-reference.json` | image edit, two references (the official example) |
| `QuantFunc-QwenImage21-edit-remove-background.json` | remove the background → transparent PNG |
| `QuantFunc-Krea2-t2i.json` | Krea-2 text-to-image |
| `QuantFunc-Krea2-t2i-double-sampling.json` | Krea-2 text-to-image in two passes: a fast loader at 1024x1024, then 1.5x latent upscale and a quality loader at 1536x1536 (one shared model) |
| `QuantFunc-LTX25-t2v.json` | LTX-2.5 text-to-video with audio, the official two-stage pipeline (half size, then 2x latent upsampler and refine) |
| `QuantFunc-MiniMaxH3-fl2va.json` | MiniMax-H3 first/last frame to video with audio |
| `QuantFunc-MiniMaxH3-fl2va-double-sampling.json` | video with audio in two stages on one 8-step sampling schedule: 3 steps at 768x768, then 1.5x nearest-exact latent upscale (no decode / encode) and the remaining 5 steps at 1152x1152; first/last reference image nodes are bypassed by default |
| `QuantFunc-MiniMaxH3-ref2va.json` | MiniMax-H3 reference images to video with audio |
| `QuantFunc-MiniMaxH3-ref2va-double-sampling.json` | reference-image video with audio: 3 steps at 768x768, then 1.5x nearest-exact latent upscale and 5 steps at 1152x1152 |

### 3.4 VRAM

A QuantFunc model takes part in ComfyUI's own memory management like any other model: ComfyUI sees its full size and
what it holds on the GPU, frees VRAM for it before it loads, and asks it for VRAM back when another model needs room.
The engine then gives back the memory it is not using and reports what it freed. When ComfyUI itself runs short of VRAM
during a run, the engine gives up its idle memory the same way. When the card cannot hold all of a model's weights,
the engine streams the rest from system memory during the run instead of failing; it runs out of memory only when one
step's working set does not fit on the card.

## 4. Troubleshooting

| Issue | Solution |
|-------|----------|
| "no QuantFunc engine is installed" / "still downloading" | The console's `[qf_native]` line from the start says why (still downloading, not installable here, update failed); see 2.5 and 2.6 |
| Engine library not found | Check the console's `[qf_native]` install line (2.6); the engine installs on the next start |
| Console shows only warnings | That is the default: the engine prints only warnings and errors |
| cuDNN BAD_PARAM | Delete cuDNN algo cache and retry |
| The engine cannot be downloaded (offline) | Put an engine build at `bin/linux/libquantfunc.so` (Windows: `bin/windows/quantfunc.dll`) with an empty `.dev_lib_lock` beside it (2.5) |
