"""Sol settings and legacy migration. Stdlib only; no engine/Comfy imports."""
from dataclasses import dataclass
import math
import struct


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def legacy_tau(value):
    """The engine's capped Acklam law, including both FP32 boundaries."""
    p = min(1.0 - 1e-4, max(1e-4, 1.0 - f32(value)))
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    if p < 0.02425 or p > 0.97575:
        q = math.sqrt(-2.0 * math.log(p if p < 0.02425 else 1.0 - p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        if p > 0.97575:
            x = -x
    else:
        q = p - 0.5
        r = q*q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    return f32(x)


def _number(value, default, name):
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Sol {name} must be a finite number")
    return float(value)


def _boolean(value, default, name):
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Sol {name} must be boolean")
    return value


@dataclass(frozen=True)
class Options:
    enabled: bool = False
    tau: float = f32(1.3)
    window_enabled: bool = True
    start_percent: float = 0.2
    end_percent: float = 0.9
    min_tokens: int = 12288
    sink_conditioning: str = "exact_kv_and_rows"

    def native(self, sampling=None):
        if not self.enabled:
            return None
        window = None
        if self.window_enabled:
            if sampling is None or not callable(getattr(sampling, "percent_to_sigma", None)):
                raise ValueError("Sol sampling window requires the model's percent_to_sigma mapping")
            start = float(sampling.percent_to_sigma(self.start_percent))
            end = float(sampling.percent_to_sigma(self.end_percent))
            if not math.isfinite(start) or not math.isfinite(end) or start < end:
                raise ValueError("Sol model sampling returned invalid sigma window bounds")
            window = {"sigma_start": start, "sigma_end": end}
        return {"enabled": True, "tau": self.tau, "window": window,
                "min_tokens": self.min_tokens, "sink_conditioning": self.sink_conditioning}


def from_loader(*, sol_version=None, sol_tau=None, sol_enabled=None,
                sol_window_enabled=None, sol_start_percent=None, sol_end_percent=None,
                sol_min_tokens=None, sol_sink_conditioning=None):
    new_values = (sol_enabled, sol_window_enabled, sol_start_percent, sol_end_percent,
                  sol_min_tokens, sol_sink_conditioning)
    if isinstance(sol_version, bool) or sol_version not in (None, 1, 2):
        raise ValueError("Unsupported Sol parameter version")
    if sol_version != 2:
        if any(v is not None for v in new_values):
            raise ValueError("New Sol options require sol_version=2; legacy sol_tau is a keep ratio")
        ratio = _number(sol_tau, 1.0, "legacy keep ratio")
        # The old client used `float(value or 1.0)` and this double off epsilon.
        if ratio == 0.0 or abs(ratio - 1.0) <= 1e-6:
            return Options()
        if not 0.0 < ratio < 1.0:
            raise ValueError("Legacy Sol keep ratio must be between 0 and 1")
        return Options(True, legacy_tau(ratio), False, 0.0, 1.0, 0, "off")
    tau = _number(sol_tau, 1.3, "tau")
    start = _number(sol_start_percent, 0.2, "start percent")
    end = _number(sol_end_percent, 0.9, "end percent")
    tokens = _number(sol_min_tokens, 12288, "minimum tokens")
    sink = "exact_kv_and_rows" if sol_sink_conditioning is None else sol_sink_conditioning
    if not -4.0 <= tau <= 4.0:
        raise ValueError("Sol tau must be between -4 and 4")
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("Sol percentages must satisfy 0 <= start <= end <= 1")
    if not tokens.is_integer() or not 0 <= tokens <= 1 << 20:
        raise ValueError("Sol minimum tokens must be an integer between 0 and 1048576")
    if sink not in ("off", "exact_kv", "exact_kv_and_rows"):
        raise ValueError("Unknown Sol conditioning sink mode")
    return Options(_boolean(sol_enabled, False, "enabled"), f32(tau),
                   _boolean(sol_window_enabled, True, "window enabled"), start, end, int(tokens), sink)
