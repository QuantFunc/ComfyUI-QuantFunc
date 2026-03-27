<div align="center" style="margin-top: 50px;">
  <img src="https://raw.githubusercontent.com/QuantFunc/ComfyUI-QuantFunc/main/assets/logo.webp" width="300" alt="QuantFunc Logo">
</div>

# ComfyUI-QuantFunc

[中文说明](README_zh.md)

## 1. Introduction

ComfyUI plugin for **QuantFunc** — the fastest diffusion model inference engine. Run quantized text-to-image and image editing models at 2x–11x speed with zero Python model dependencies.

**Key features:**
- Native C++/CUDA acceleration via `libquantfunc.so` / `quantfunc.dll`
- SVDQ + Lighting dual engine support
- Zero-cost LoRA stacking
- Image editing with reference images
- Model export with LoRA baked in
- Auto-update from ModelScope

## 2. Installation

### 2.1 Method A: Clone from Git (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/QuantFunc/ComfyUI-QuantFunc.git
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

See [workflow_sample/README.md](workflow_sample/README.md) for detailed node reference and quick start guides.

### 3.1 Basic Flow

```
ModelLoader → (LoRA) → (LoRA Config) → Generate → PreviewImage
```

1. **QuantFunc Model Loader** — set model path, backend, device
2. **QuantFunc LoRA** (optional) — chain one or more LoRA adapters
3. **QuantFunc LoRA Config** (optional, required for SVDQ + LoRA) — merge strategy
4. **QuantFunc Generate** — enter prompt, dimensions, steps → outputs IMAGE

### 3.2 Example Workflows

Import from `workflow_sample/`:

| File | Use Case |
|------|----------|
| `QuantFunc-Text-to-Image-Workflow.json` | Text-to-image (SVDQ + Lighting side by side) |
| `QuantFunc-Image-to-Image-Workflow.json` | Image editing with reference images |
| `QuantFunc-Model-Export.json` | Export quantized model with LoRA |

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
