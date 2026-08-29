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
