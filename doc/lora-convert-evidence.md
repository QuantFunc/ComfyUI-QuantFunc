# qf_lora_convert — committed verification evidence (P2, 2026-08-30)

Command per LoRA: python3 scripts/qf_lora_convert.py --self-test <file>
(round-trip: real diffusers LoRA -> synthesized kohya twin -> convert back;
 asserts module-set identity + tensor byte identity. Machine output verbatim.)

```
== krea2_comic_book ==
  module-set: 528 orig / 528 recon  match=True | tensor-bytes: 528 identical, 0 differ
SELF-TEST: PASS
== krea2_retro_anime_akira ==
  module-set: 512 orig / 512 recon  match=True | tensor-bytes: 512 identical, 0 differ
SELF-TEST: PASS
== ltx-2.5_ai_girl_fictional ==
  module-set: 3264 orig / 3264 recon  match=True | tensor-bytes: 3264 identical, 0 differ
SELF-TEST: PASS
== ltx25_Skeletor_lora_v1_LR32 ==
  module-set: 3264 orig / 3264 recon  match=True | tensor-bytes: 3264 identical, 0 differ
SELF-TEST: PASS
== ltx-2.3-22b-distilled-lora-384-1.1 ==
  module-set: 3320 orig / 3320 recon  match=True | tensor-bytes: 3320 identical, 0 differ
SELF-TEST: PASS
```

## krea2 263-module coverage probe (checkpoint-vs-LoRA module sets)
```
LoRA modules: 264  checkpoint-matched: 264  unmatched: []
(engine run log 2026-08-30 00:41: 'LoRA: applied 263/263 target modules' — comfy_gen.log)
```

## ComfyUI end-to-end, one family at a time (2026-09-24, issue #715)

The user's flow, verbatim: run `scripts/qf_lora_convert.py` **from the plugin directory** with ComfyUI's own python, write the
output into ComfyUI's `models/loras`, load it with the **QuantFunc Native LoRA** node after the family's native loader. Every step
of one family runs on ONE ComfyUI server and ONE pipeline (the Native LoRA node applies each set in place), same seed.

Setup: 远程-linux-c (RTX 5090, SM120, 32 GB), an isolated ComfyUI 0.37.0 on port 18815 started with `--cache-classic` (the loader
stays cached across every step, the sampler re-runs whenever its MODEL changes, so "off" and "on" are real re-renders), this
plugin tree at 6eb81ca (branch lora-convert-krea2, extracted with `git archive`; `bin/*/config.json` excluded), engine `libquantfunc.so` md5 f78ba101a5070123980b56a2702d8007 (QuantFunc #714+#715 on main 3fe5b9cbd, tree 5ff7f9bd2, sm_120a, `QF_NATIVE_SO_PATH`; the worker's own
`[qf_native] engine lib: … md5=` line was checked against it on every run). Runs go
through the QuantFunc test harness (engine `comfyui`, cases `lora/comfy-native-lora-e2e-image` and
`video/comfy-native-lora-e2e-video`), which wires each LoRA set as QuantFuncNativeLoRA nodes where a user puts them.

Conversion (from the plugin directory, ComfyUI's python 3.12.4, converter md5 8f0fee8db723 = this commit's):
```
cwd=/root/qf715-e2e/comfy-base/custom_nodes/ComfyUI-QuantFunc converter=8f0fee8db723 python=3.12.4
[qf_lora_convert] WARNING: converting kohya keys WITHOUT --model — the underscore->dot reconstruction falls back to a built-in vocabulary that is corpus-fitted and may miss a family's novel module names. STRONGLY recommended: pass --model <target-checkpoint.safetensors> for an exact, vocabulary-free inverse.
[qf_lora_convert] Krea2-漫画风格ogipote-ep63.safetensors: source=kohya (Krea-2 BFL names -> engine names)  wrote 792 tensors, dropped 0 text-encoder key(s)  -> ../../models/loras/Krea2-漫画风格ogipote-ep63-diff.safetensors
  md5 838b682668ff7759fb803f82f44d7850  Krea2-漫画风格ogipote-ep63-diff.safetensors
[qf_lora_convert] Krea2-炭笔素描.safetensors: source=diffusers (Krea-2 BFL names -> engine names)  wrote 512 tensors, dropped 0 text-encoder key(s)  -> ../../models/loras/Krea2-炭笔素描-diff.safetensors
  md5 6d9931956cb41bafe8e55a7d90cda328  Krea2-炭笔素描-diff.safetensors
[qf_lora_convert] FrameRush-Minimax-V2_c2-st1500.safetensors: source=diffusers  wrote 200 tensors, dropped 0 text-encoder key(s)  -> ../../models/loras/FrameRush-Minimax-V2_c2-st1500-diff.safetensors
  md5 59418b9a7f38f21e44ddbab936b96897  FrameRush-Minimax-V2_c2-st1500-diff.safetensors
[qf_lora_convert] Qwen2.1_Anime_consistency.safetensors: source=diffusers  wrote 448 tensors, dropped 0 text-encoder key(s)  -> ../../models/loras/Qwen2.1_Anime_consistency-diff.safetensors
  md5 91de6c7de8866cf58609a35fdf377da9  Qwen2.1_Anime_consistency-diff.safetensors
```

Results (drift = RMSE of the LoRA render vs the no-LoRA render at the same seed; "off"/"on" = the render is pixel-identical to the
earlier one — compared on decoded pixels, because ComfyUI embeds the workflow, incl. the LoRA file name, in every PNG):

| family | community file (source format) | raw file in the Native LoRA node | converted file | modules applied | drift vs no-LoRA | off / on | run id |
|---|---|---|---|---|---|---|---|
| Krea-2 Turbo int4 r128 | `Krea2-炭笔素描` (BFL / ai-toolkit names: `diffusion_model.txtfusion.*`, `first` …) | refused by the engine: 448 applied + 64 NOT APPLIED (no-module `txtfusion.*`), naming this script | `Krea2-炭笔素描-diff.safetensors` 6d9931956cb41bafe8e55a7d90cda328 | 256/256 (512 keys, 0 NOT APPLIED) | 0.2488 | — | qf715-comfye2e-krea2-c-0924-1900 |
| Krea-2 Turbo int4 r128 | `Krea2-漫画风格ogipote-ep63` (kohya, `lora_unet_*`) | refused by the node, printing this script's command | `Krea2-漫画风格ogipote-ep63-diff.safetensors` 838b682668ff7759fb803f82f44d7850 | 264/264 (528 A/B + 264 alpha, 0 NOT APPLIED) | 0.2197 | off: `[]` == first render; on: == its first render | qf715-comfye2e-krea2-c-0924-1900 |
| Qwen-Image-2.1 int4 r128 | `Qwen2.1_Anime_consistency` (PEFT adapter-name `.lora_A.default.weight`) | applies as is (448/448) | `Qwen2.1_Anime_consistency-diff.safetensors` 91de6c7de8866cf58609a35fdf377da9 | 224/224 (448 keys); pixel-identical to the raw file | 0.1338 | off/on identical | qf715-comfye2e-qi21-c-0924-1902 |
| MiniMax-H3 t2va 512²×22 | `FrameRush-Minimax-V2_c2-st1500` (PEFT adapter-name `.default`); `H3_Combat_V2` (diffusers) | FrameRush applies as is (200/200); Combat 416/416 | `FrameRush-Minimax-V2_c2-st1500-diff.safetensors` 59418b9a7f38f21e44ddbab936b96897 | 100/100 (200 keys); all 22 frames pixel-identical to the raw file | FrameRush 0.1317; Combat 0.1750 | off/on identical (22 frames) | qf715-comfye2e-h3-c-0924-1903 |
| LTX-2.5 22B int4 768×512×49 | `LTX-2.3-22b-AV-LoRA-talking-head-v1`, `Singularity LTX-2.3 OmniCine Preview v0.1` (diffusers) | 2304/2304 and 3264/3264 (no conversion needed) | — | 1152/1152, 1632/1632 | talking-head 0.0866; OmniCine 0.1070 | off/on identical (49 frames) | qf715-comfye2e-ltx25-c-0924-1905 |

Findings from this pass:
- In the LoRA collections on the test boxes (`/datasets/ComfyUI/models/loras`, `/home/waas/model_cache/models/lora`), Krea-2 is
  the only family with non-diffusers community LoRAs (BFL / ai-toolkit names, kohya). Both raw forms are
  refused with the converter named; the converted files apply in full. Exactness of the conversion: on engine builds that differ
  only by the removal of the engine's own Krea-2 aliases, the converted files render byte-identically to the raw files the aliased
  engine mapped (same seed; ctypes runs lora715r4-lora-krea2-root-runtime-swap-e-0924-1538 vs lora715r8-…-e-0924-1642: charcoal
  9bf7984d, Radiance ee270c86, ogipote 05e4a252). This commit's converter writes the same tensors plus the source metadata: the
  final renders above are pixel-identical to a pre-flight of the same graphs with the 2f63e54 output (qf715-comfye2e-*-0924-183x).
- H3 and LTX community LoRAs on the box are already diffusers (`lora_A/B`); they load directly.
- Two community files use the PEFT adapter-name form (`.lora_A.default.weight`: FrameRush-Minimax-V2 for H3,
  Qwen2.1_Anime_consistency for QI-2.1). The engine reads that form, but this converter used to DROP every such key and stop with
  "nothing to write" (fixed in 2f63e54); the converted files render pixel-identically to the raw ones.
- LyCORIS (LoHa/LoKr), DoRA and OFT/BOFT are not supported: the node, the engine and this converter all say so, and none of
  them sends the user to the others (6eb81ca: the converter used to drop DoRA magnitude vectors and write the direction alone,
  rc 0 — not the trained LoRA). Any other key the converter cannot map is refused, naming it; the source `__metadata__` is kept
  (the engine reads alpha / rank from `lora_adapter_metadata`).
- LTX-2.5: this pass found an engine bug (QuantFunc #729) — on SM120 every LTX render with a LoRA failed ("a user LoRA is bound on a slot whose
  kernel does not carry the term … Consult userLoRAFusable()"); QuantFunc 41cda66c6 fixes it (LTX takes the unfused gate / gelu
  chain for such a slot). With no LoRA, the output is byte-identical: the LTX `[]` render's 49 frames have the same
  pixels before and after the fix (bb769a9a1f89).
