# LoRA 格式转换工具 `qf_lora_convert.py` / LoRA Format Converter

QuantFunc 原生 loader(Krea-2 / LTX-2 / MiniMax-H3)的 LoRA 路径**只适配一种最主流格式**:
**diffusers / PEFT 规范形**(`<模块路径>.lora_A.weight` / `<模块路径>.lora_B.weight`,可带 `.alpha`)。
其它训练器产出的格式(kohya sd-scripts、ai-toolkit 等下划线命名)用本工具**一次性转换**成规范形即可加载。

The native loaders adapt ONE mainstream LoRA format — diffusers/PEFT canonical. Convert
anything else once with this tool.

## 为什么这么设计 / Why

- 引擎按 checkpoint 自身的模块名(即 diffusers 命名)注册模块,规范形 LoRA 的 key 与引擎模块**天生同名**,
  无需任何逐家族映射表——新模型家族零配置即可用 LoRA。
- 各种训练器格式的差异(前缀、下划线拍平、`lora_down/up` 拼写、alpha)只是**键名编码**问题,
  放在一个纯 Python 工具里一次解决,比在 C++ 引擎里维护 N 张映射表简洁、可靠得多。
- 转换是**无损字节拷贝**:不加载模型、不用 torch、不用 GPU,张量数据与 dtype 原样保留,只改键名。

## 用法 / Usage

```bash
cd ComfyUI/custom_nodes/ComfyUI-QuantFunc/scripts

# kohya / ai-toolkit → diffusers 规范形(常规用法)
python3 qf_lora_convert.py \
    --in  /path/to/my_kohya_lora.safetensors \
    --out /path/to/my_kohya_lora-diff.safetensors

# 强烈推荐:给出目标模型文件,用它的键名做【精确】逆映射(任何家族 100% 无歧义)
python3 qf_lora_convert.py \
    --in  my_kohya_lora.safetensors \
    --out my_kohya_lora-diff.safetensors \
    --model /path/to/ComfyUI/models/diffusion_models/krea2-turbo-quantfunc-int4-r32.safetensors

# 自检:对任意 diffusers 形 LoRA 做「合成 kohya 孪生 → 转回」的 round-trip,
# 断言模块集合与张量字节完全一致
python3 qf_lora_convert.py --self-test /path/to/some_diffusers_lora.safetensors
```

转换后把 `-diff.safetensors` 放进 `models/loras/`,在 **QuantFunc Native LoRA** 节点里选它即可。

## 行为细节 / Behavior

| 情况 | 行为 |
|---|---|
| 已是 diffusers/PEFT 形 | 归一(`lora_down/up` → `lora_A/B`)后原样输出 |
| kohya / ai-toolkit(`lora_unet_*` 下划线键) | 键名重建为点分模块路径;`--model` 时用目标 checkpoint 的真实模块表做精确逆映射,否则用内置词表 |
| 文本编码器 LoRA(`lora_te*`) | 丢弃(原生 loader 只驱动 transformer) |
| LyCORIS(LoHa/LoKr) | **响亮拒绝**——它是因子分解不是 (A,B) 低秩对,改名转不动;请先合并/重导出为标准 LoRA |
| 两个源键映射到同一目标键 | 拒绝(绝不猜) |

插件端的 **QuantFunc Native LoRA** 节点会嗅探文件头:遇到外来格式会直接报错并打印上面这条转换命令,
不会静默半合(half-apply)。

## 验证记录 / Verification

`--self-test` 在本机 5/5 真实 LoRA 上通过(krea2 × 2、LTX-2.5 × 2、LTX-2.3-22b 的 3320 模块 AV 巨件),
模块集合与张量字节逐一相同;转换产物喂给引擎与原生 diffusers 文件**逐模块合并数一致**(263/263),
同种子出图收敛在 svdq 前向自噪包络内。
