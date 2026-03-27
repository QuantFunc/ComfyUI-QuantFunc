<div align="center" style="margin-top: 50px;">
  <img src="https://raw.githubusercontent.com/QuantFunc/ComfyUI-QuantFunc/main/assets/logo.webp" width="300" alt="QuantFunc Logo">
</div>

# ComfyUI-QuantFunc

[English](README.md)

## 1. 简介

**QuantFunc** 的 ComfyUI 插件 —— 最快的扩散模型推理引擎。以 2x-11x 加速运行量化文生图和图像编辑模型，零 Python 模型依赖。

**核心特性：**
- 通过 `libquantfunc.so` / `quantfunc.dll` 原生 C++/CUDA 加速
- SVDQ + Lighting 双引擎支持
- 零成本 LoRA 叠加
- 参考图像编辑
- 模型导出（可烘焙 LoRA）
- 从 ModelScope 自动更新

## 2. 安装

### 2.1 方式 A：Git 克隆（推荐）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/QuantFunc/ComfyUI-QuantFunc.git
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

详细节点说明和快速上手指南见 [workflow_sample/README_zh.md](workflow_sample/README_zh.md)。

### 3.1 基本连接

```
ModelLoader → (LoRA) → (LoRA Config) → Generate → PreviewImage
```

1. **QuantFunc Model Loader** —— 设置模型路径、后端、设备
2. **QuantFunc LoRA**（可选）—— 链式添加一个或多个 LoRA
3. **QuantFunc LoRA Config**（可选，SVDQ + LoRA 时必须）—— 合并策略
4. **QuantFunc Generate** —— 输入提示词、尺寸、步数 → 输出 IMAGE

### 3.2 示例工作流

从 `workflow_sample/` 导入：

| 文件 | 用途 |
|------|------|
| `QuantFunc-Text-to-Image-Workflow.json` | 文生图（SVDQ + Lighting 并排对比） |
| `QuantFunc-Image-to-Image-Workflow.json` | 参考图像编辑 |
| `QuantFunc-Model-Export.json` | 导出量化模型（含 LoRA） |

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
