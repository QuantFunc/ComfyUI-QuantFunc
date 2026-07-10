<div align="center" style="margin-top: 50px;">
  <img src="https://raw.githubusercontent.com/RealJonathanYip/ComfyUI-QuantFunc/main/assets/logo.webp" width="300" alt="QuantFunc Logo">
</div>

<p align="center">
  🌐 <a href="https://www.quantfunc.com/">官网</a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/QuantFunc">Hugging Face</a> &nbsp;|&nbsp;
  🤖 <a href="https://www.modelscope.cn/profile/QuantFunc">ModelScope</a> &nbsp;|&nbsp;
  💬 <a href="#wechat">微信群</a> &nbsp;|&nbsp;
  🎮 <a href="https://discord.gg/jCp9TpFWcn">Discord</a>
</p>

# ComfyUI-QuantFunc

[English](README.md)

## 1. 简介

**QuantFunc** 的 ComfyUI 插件 —— 最快的扩散模型推理引擎。以 2x-11x 加速运行量化文生图和图像编辑模型，零 Python 模型依赖。

**核心特性：**
- 通过 `libquantfunc.so` / `quantfunc.dll` 原生 C++/CUDA 加速
- SVDQ（离线量化）+ Lighting（运行时量化）双引擎
- 零成本 LoRA 叠加
- 参考图像编辑
- 导出运行时量化模型（支持融合 LoRA）
- 从 ModelScope 自动更新

## 版本历史

| 插件 (`comfy`) | 引擎 (`lib`) | 概要 |
|:---:|:---:|---|
| **0.0.06** *(当前)* | **0.0.12** | 全新模式 —— Ideogram4 · 分层（RGBA）· ControlNet · 图生图 · 显存预算 · FBCache · Klein 自动下载 · SHA-256 完整性校验 —— 详见下方 |
| 0.0.02 | 0.0.07 | v2 加载架构 · 局部重绘 · 全显卡覆盖 · 编辑提速 |
| 0.0.01 | 0.0.01 – 0.0.06 | 基础版本：运行时/离线量化 · 模型与 LoRA 加载 · 参考图编辑 · 导出 · 自动更新 |

### 0.0.06（引擎 0.0.12）新增内容

**🧩 全新生成模式** —— 每个都在 [`workflow_sample/`](workflow_sample/) 中附带可直接运行的工作流：
- **Ideogram4 文生图** —— 强提示词与文字 / 排版还原；支持 1024² / 1536² / 2048²，8 GB 显存即可运行（`QuantFunc-Ideogram4.json`）。
- **分层生成（QwenImage Layered）** —— 单次生成即可将画面拆解为透明 **RGBA 图层**，并由 FBCache 加速（`QuantFunc-QwenImage-Layered.json`）。
- **ControlNet** —— 结构引导生成（QwenImage InstantX ControlNet）（`QuantFunc-ControlNet.json`）。

**🎛️ 新增节点与控制**
- **Wan 单文件 TI2V-5B（三件套）** —— 直接加载 ComfyUI 单文件版 Wan2.2 TI2V-5B：把 **UNETLoader**（`wan2.2_ti2v_5B_*.safetensors`）+ **CLIPLoader**（`umt5_xxl_*.safetensors`，fp16 或 fp8-scaled）+ **VAELoader**（`wan2.2_vae.safetensors`）接进 `QuantFunc Build Pipeline` 即可 —— 键名自动重映射为 diffusers 命名、fp8 自动反量化（与 Combine Experts 同一套机制），暂存在 ComfyUI 临时目录的会话缓存中。16 通道的单文件（A14B 专家）会被明确拒绝并给出指引 —— A14B 成对模型请用 *Wan Combine Experts*。
- **`tiny_vae` 快速预览开关**（*Build Pipeline*，仅 Wan 视频，**默认关闭**）—— 将完整 Wan VAE 解码器替换为轻量 TAEHV 解码器：视频解码大幅加速（实测 Wan2.1/A14B @384×384 约 5×，Wan2.2-5B 最高约 28×），画面保持连贯但**偏软** —— 适合草稿/预览，最终出片请关闭。taew 变体自动选择（Wan2.1/A14B → `taew2_1`，Wan2.2-5B → `taew2_2`）；权重放置于 `<ComfyUI>/models/QuantFunc/taew/<variant>.safetensors`（权重缺失 / 非 Wan 模型 / 引擎库版本过旧均会明确报错，绝不静默回退）。
- **显存预算**下拉（*Build Pipeline*）—— 在创建时限制显存上限，引擎据此按更小的显卡来规划运行。
- **FBCache 加速** —— 支持独立的 **cond / uncond** 阈值（`fbcache` / `fbcache_uncond`）。
- **图生图** —— *Generate* 节点新增 `init_img` 接口与 `init_img_strength`。
- **Klein 4B / 9B** 一键自动下载（3 档 50x / 40x / 30x-below），见 *Auto Loader*。
- **Ideogram-4 + Qwen-Image-Layered** 精度配置由自动加载器自动识别。

**🔒 引擎完整性校验（新增）**
- 插件启动时会用 **SHA-256** 把已安装的引擎库与 ModelScope 上发布的官方清单（`<版本>/verify.json`）比对，**每次启动都重新拉取**（仅在网络不可达时才读本地缓存）。损坏、不完整或装错版本的二进制会**自愈** —— **仅当重新下载的官方产物哈希匹配时才替换**（verify-before-replace，坏下载绝不会覆盖能用的库）。全程不阻塞节点加载；本地自行编译的库保持不动。

**⚡ 引擎改进（0.0.12 相较 0.0.11）**
- **显存预算（workspace budget）** —— 通过设定固定预算，让大模型在显存紧张的显卡上运行。
- **更广的低显存与 RTX 20（Turing）支持** —— Ideogram4 与分层生成现可在 8 GB、SM75 上运行。
- **RTX 50（Blackwell）FP4 快速通道** —— 为新增流水线启用。

> 插件启动时自动拉取匹配的引擎：把 `comfy` 升到 **0.0.06** 即可让更新器从 ModelScope 拉取引擎 **0.0.12**。

### 0.0.02（引擎 0.0.07）新增内容

**🎯 易用性**
- **v2 加载器** —— 独立 `MODEL` / `CLIP` / `VAE` 接口接入 **Build Pipeline** 节点，模型按 ComfyUI 原生方式连线，不再是单一的整体加载器。
- **通用格式适配** —— 自动识别 **diffusers / BFL（Flux）/ nunchaku SVDQ / 全家桶 / HF** 多种布局，无需手动转换。
- **基础模型自动加载器**（一键下载）；插件首次启动还会自动拉取匹配的引擎。

**🧩 模型支持**
- **SVDQ**（离线量化）**+ Lighting**（运行时 BF16/FP16 → 4bit）双引擎。
- 支持管线：**Z-Image · QwenImage · QwenImage-Edit · Flux.2 Klein**。
- **全显卡覆盖**（引擎 0.0.07）：消费级 **RTX 20 / 30 / 40 / 50 系**、数据中心 **A100 / H100 / H200 / B100 / B200 / GB300**、工作站 **RTX 6000 Ada / RTX PRO 6000 Blackwell** —— 覆盖 **CUDA 12 与 13**。

**⚡ 性能**
- **消费级显卡原生 SASS** —— 20/30/40/50 系*首次运行零 JIT 编译卡顿*（数据中心/工作站显卡仅首次 JIT，之后缓存）。
- Blackwell（SM120）原生 **FP4（NVFP4）** —— 最快的 4bit 路径。
- 参考图/蒙版用 **QFRAW 原始格式暂存**，跳过 PNG/BMP 编码（每张参考图省 ~80ms）。
- **多 pipeline CPU↔GPU 共存** —— 切换管线无需整体重载；空闲 worker 自动释放显存。

**✨ 新功能**
- **局部重绘** —— `MASK` 输入 + **Mask Config** 与 **Mask Scale By** 节点（白色重绘、黑色保留），对齐 ComfyUI 的 SetLatentNoiseMask。
- **Build Pipeline** 节点（v2 组装），支持逐组件精度控制。
- 健壮的 **worker 进程架构** —— CPU↔GPU 模型交换 + 僵尸 worker 清理。

**🛡️ 稳定性与安全**
- 修复 **`/dev/shm` 内存泄漏** —— 编辑/重绘的暂存文件现在总会被清理。
- 依赖压缩包解压增加 **zip-slip 防护**。
- worker → 主进程图像传输增加 **IPC 边界检查**。

> 插件启动时自动拉取匹配的引擎：把 `comfy` 升到 **0.0.02** 才能让更新器从 ModelScope 拉取引擎 **0.0.07**（更低的 `comfy` 版本会被限制在引擎 0.0.06）。

## 2. 安装

### 2.1 方式 A：Git 克隆（推荐）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RealJonathanYip/ComfyUI-QuantFunc.git
```

插件首次启动时会**自动从 ModelScope 下载**最新兼容的 `libquantfunc.so`（Linux）或 `quantfunc.dll`（Windows），无需手动下载。

### 2.2 方式 B：手动安装

1. 下载或克隆此仓库到 `ComfyUI/custom_nodes/`：

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

2. 启动 ComfyUI —— 插件会在首次运行时自动下载库文件。

3.（可选）跳过自动下载，手动放置二进制文件：
   - **Linux:** 下载 `libquantfunc.so` → `bin/linux/`
   - **Windows:** 下载 `quantfunc.dll` → `bin/windows/`

### 2.3 系统要求

| 要求 | 最低配置 |
|------|----------|
| **GPU** | NVIDIA RTX 20 系列或更新（CC 7.5+） |
| **显存** | 8 GB |
| **驱动** | NVIDIA ≥ 560 |
| **CUDA 运行时** | 13.0+ |
| **cuDNN** | 9.x |
| **操作系统** | Linux (glibc 2.31+) 或 Windows 10/11 |
| **Python** | 3.9+（ComfyUI 内置 Python） |

### 2.4 运行时依赖

#### Linux

```bash
# CUDA 12 运行时库
sudo apt install cuda-libraries-12-8
# 或单独安装：
sudo apt install libcublas-12-8 libcurand-12-8 libcusolver-12-8 libcusparse-12-8 libnvjitlink-12-8

# cuDNN 9
sudo apt install libcudnn9-cuda-12

# --- 或者 ---

# CUDA 13 运行时库
sudo apt install cuda-libraries-13-0
# 或单独安装：
sudo apt install libcublas-13-0 libcurand-13-0 libcusolver-13-0 libcusparse-13-0 libnvjitlink-13-0

# cuDNN 9
sudo apt install libcudnn9-cuda-13
```

#### Windows

- **NVIDIA 驱动** ≥ 560（提供 CUDA 运行时 DLL）
- **Visual C++ Redistributable** 2015-2022（[下载](https://aka.ms/vs/17/release/vc_redist.x64.exe)）
- **cuDNN 9.x**（[从 NVIDIA 下载](https://developer.nvidia.com/cudnn)）

### 2.5 ModelScope 依赖（用于自动更新）

自动更新需要 `modelscope` Python 包：

```bash
pip install modelscope
```

如果未安装 `modelscope`，自动更新会静默跳过。你可以手动从以下地址下载二进制文件：
- https://www.modelscope.cn/models/QuantFunc/Plugin

### 2.6 验证安装

启动 ComfyUI 后，检查控制台输出：

```
[QuantFunc] Checking for updates (plugin v0.0.01, lib v0.0.01)...
[QuantFunc] Library is up to date (v0.0.01)
```

如果库文件不存在：

```
[QuantFunc] No library found, checking ModelScope for download (plugin v0.0.01)...
[QuantFunc] Downloading libquantfunc.so v0.0.01 from ModelScope...
[QuantFunc] Updated libquantfunc.so to v0.0.01. Restart ComfyUI to use the new version.
```

## 3. 使用方法

详细教程见 [doc/](doc/)，节点说明见 [workflow_sample/README_zh.md](workflow_sample/README_zh.md)。

### 新手快速上手

最简单的上手方式——加一个 **Model Auto Loader**，从下拉菜单选择模型系列，接到 **Build Pipeline → Generate**，插件自动下载一切，点击生成即可。无需手动下载模型或填写路径。

> **[快速上手与文档索引 →](doc/README.md)**

### 3.1 运行时量化：将 BF16/FP16 原模型量化为 4bit 加速推理

**Lighting 后端**提供**运行时量化**能力 —— 在加载时将任意 diffusers 格式的 BF16/FP16 原模型（如 [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)）量化为 4bit 并加速推理。只需把 **Model Loader** 指向 FP16 原模型、`transformer_path` 留空即可（后端自动检测），无需下载预量化模型。

> **[模型加载与运行时量化 →](doc/model-loading-and-apikey_zh.md)**

### 3.2 导出运行时量化模型（支持融合 LoRA）

Lighting 导出功能将运行时量化产生的所有量化模型持久化到磁盘，避免每次启动都重新量化。如果叠加了 LoRA，LoRA 也会被永久融入导出的权重 —— 无需 LoRA 节点，无需重新量化，加载即用。

> **[导出量化模型（含 LoRA 融合）→](doc/export-quantized-models_zh.md)**

### 3.3 下载并使用已导出的量化模型

QuantFunc 已将常用模型提前进行运行时量化并导出，你可以直接从 [ModelScope](https://www.modelscope.cn/models/QuantFunc) 或 [HuggingFace](https://huggingface.co/QuantFunc) 下载这些**已导出的量化模型**，加载即用，无需自行量化。与运行时量化一样，这些模型同样能达到 2x-11x 推理加速，且跳过了量化步骤，加载更快。

> **[模型加载与下载 →](doc/model-loading-and-apikey_zh.md)**

### 3.4 示例工作流

从 [`workflow_sample/`](workflow_sample/) 导入：

| 文件 | 用途 |
|------|------|
| `QuantFunc-Sample-WorkFlow-All-In-One.json` | **全功能合集** —— 每个节点 × 3 种模型加载方式 × 文生图 / 编辑 / 导出 |
| `QuantFunc-Ideogram4.json` | Ideogram4 文生图 + 提示词构建 |
| `QuantFunc-QwenImage-Layered.json` | 分层（透明 RGBA）生成 + 图层查看 |
| `QuantFunc-ControlNet.json` | ControlNet 结构引导生成 |

## 4. 常见问题

| 问题 | 解决方案 |
|------|----------|
| Worker 启动失败 | 检查 CUDA 驱动 ≥ 560，确保已安装 CUDA 运行时库 |
| 找不到 DLL/SO | 检查 `bin/linux/` 或 `bin/windows/` 是否包含库文件；重启 ComfyUI 触发自动下载 |
| 无日志输出 | 更新到最新库版本（需支持 stderr 日志） |
| cuDNN BAD_PARAM | 删除 cuDNN 算法缓存后重试 |
| 输出全噪声 | 确认 model_backend 与 Transformer 权重匹配（svdq vs lighting） |
| 自动更新失败 | 安装 `modelscope` 包，或从 ModelScope 手动下载 |

## 5. 许可证

见 [QuantFunc Plugin 许可证](https://www.modelscope.cn/models/QuantFunc/Plugin)。

## 社区

加入我们的社区获取支持、更新与交流:

- 🎮 [Discord 服务器](https://discord.gg/jCp9TpFWcn)
- 💬 扫描下方二维码加入我们的微信群:

<div align="center" id="wechat">
  <img src="assets/WeChat.jpg" alt="WeChat Group" width="300">
</div>
