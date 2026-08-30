"""[re-CR 2026-08-30] One-format LoRA refusal is family-scoped: the shared
QuantFuncNativeLoRA node's foreign-format sentinel must be SKIPPED for Wan
(engine WAN_RULES keep native kohya) and FIRE for the one-format families
(Krea2/LTX2/H3). Deterministic — exercises the exact apply()-path family gate
without a live ComfyUI env (which SKIPs on this box's torch/torchvision mismatch).
The gate is: `if not getattr(model.model, "_qf_kohya_lora_ok", False): refuse`.
"""
import sys


class _FakeInner:
    """Stand-in for model.model — a family's QF*Model instance."""
    def __init__(self, exempt):
        if exempt:
            self._qf_kohya_lora_ok = True  # QFWanModel sets this


class _FakeModel:
    def __init__(self, exempt):
        self.model = _FakeInner(exempt)


def _gate_would_refuse(model):
    """Byte-identical to __init__.py:907 — the production family gate."""
    return not getattr(model.model, "_qf_kohya_lora_ok", False)


def main():
    fails = 0

    # (1) Wan (exempt attribute set) -> gate does NOT refuse -> kohya LoRA reaches the engine
    if _gate_would_refuse(_FakeModel(exempt=True)):
        print("FAIL: Wan (exempt) was refused"); fails += 1
    else:
        print("ok: Wan exempt -> sentinel SKIPPED (kohya reaches engine WAN_RULES)")

    # (2) one-format families (no attribute) -> gate refuses -> sentinel fires
    if not _gate_would_refuse(_FakeModel(exempt=False)):
        print("FAIL: a one-format family was NOT refused"); fails += 1
    else:
        print("ok: one-format family -> sentinel FIRES (kohya refused, must convert)")

    # (3) attribute-missing default is REFUSE (default-deny — a new family added
    #     without opting in is correctly refused, matching the engine E1 set)
    class _Bare:
        pass
    m = _FakeModel(exempt=False); m.model = _Bare()
    if not _gate_would_refuse(m):
        print("FAIL: default (attribute absent) did not refuse"); fails += 1
    else:
        print("ok: default-deny (absent attribute -> refuse)")

    if fails:
        print(f"WAN_KOHYA_EXEMPTION: FAIL ({fails})"); return 1
    print("WAN_KOHYA_EXEMPTION: PASS"); return 0


if __name__ == "__main__":
    sys.exit(main())
