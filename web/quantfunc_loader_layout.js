// Loader layout (user 2026-09-28 「要输入的都在上方 开关的统一在下方」): on every QuantFunc loader the value inputs come first
// and the switches last, and the MiniMax-H3 loader no longer has its partial-denoise switch (double sampling always
// works). ComfyUI restores a saved node's widget values by POSITION, so a workflow saved with the previous order would put
// each value into the wrong widget. This remaps such a workflow once, on load, and stamps every loader node with the
// layout it is saved with, so a remapped or new node is never remapped again.
import { app } from "../../scripts/app.js";

const LAYOUT = 3;                  // node.properties.qf_layout of a loader saved with the current order
// Earlier saved orders the loaders may meet, with the type each number / switch / dropdown value had: a saved workflow
// is remapped only when its values have exactly one of these shapes (the current LTX order has as many values as the
// previous one, and a Krea-2 / Qwen-Image-2.1 order did not change at all). The release-candidate loaders had a `quality`
// dropdown where quality_enhance is now: best_quality maps to ON and any other value to OFF, as for an old API prompt.
const V = (name, type) => [name, type];
const EARLIER = {
  QuantFuncLTXLoader: [
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"),
     V("step_cache", "number"), V("block_cache", "number"), V("quality_enhance", "boolean"), V("pinned_memory", "boolean")],
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"), V("quality", "string"),
     V("step_cache", "number"), V("block_cache", "number")],
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"), V("quality_enhance", "boolean"),
     V("step_cache", "number"), V("block_cache", "number"), V("pinned_memory", "boolean")],
  ],
  QuantFuncH3Loader: [
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"),
     V("step_cache", "number"), V("block_cache", "number"), V("quality_enhance", "boolean"),
     V("audio_enhance", "boolean"), V("pinned_memory", "boolean")],
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"), V("quality", "string"),
     V("audio_enhance", "boolean"), V("step_cache", "number"), V("block_cache", "number"),
     V("allow_partial_denoise", "boolean")],
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"), V("quality_enhance", "boolean"),
     V("audio_enhance", "boolean"), V("step_cache", "number"), V("block_cache", "number"),
     V("allow_partial_denoise", "boolean"), V("pinned_memory", "boolean")],
  ],
  QuantFuncKrea2Loader: [
    [V("transformer"), V("model_config"), V("attention_backend"), V("quality_enhance", "boolean"), V("pinned_memory", "boolean")],
    [V("transformer"), V("model_config"), V("attention_backend"), V("quality", "string")]],
  QuantFuncQwenImage21Loader: [
    [V("transformer"), V("model_config"), V("attention_backend"), V("quality_enhance", "boolean"), V("pinned_memory", "boolean")],
    [V("transformer"), V("model_config"), V("attention_backend"), V("quality", "string")]],
};

const isLoader = (name) => typeof name === "string" && name.startsWith("QuantFunc") && name.endsWith("Loader");
const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const SOL_DEFAULTS = { sol_version: 2, sol_enabled: false, sol_tau: 1.3, sol_window_enabled: true,
  sol_start_percent: 0.2, sol_end_percent: 0.9, sol_min_tokens: 12288, sol_sink_conditioning: "exact_kv_and_rows" };

// Same capped inverse-normal approximation and FP32 boundaries as qf_sol.legacy_tau.
export function legacyTau(ratio) {
  const p = Math.min(1 - 1e-4, Math.max(1e-4, 1 - Math.fround(ratio)));
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
    138.3577518672690, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const poly = (coeff, x) => coeff.reduce((v, k) => v * x + k, 0);
  let x;
  if (p < 0.02425 || p > 0.97575) {
    const q = Math.sqrt(-2 * Math.log(p < 0.02425 ? p : 1 - p));
    x = poly(c, q) / (poly(d, q) * q + 1);
    if (p > 0.97575) x = -x;
  } else {
    const q = p - 0.5, r = q * q;
    x = poly(a, r) * q / (poly(b, r) * r + 1);
  }
  return Math.fround(x);
}

function migrateSol(byName, propertyVersion) {
  const widgetVersion = byName.sol_version;
  for (const version of [propertyVersion, widgetVersion]) {
    if (version !== undefined && version !== 1 && version !== 2) throw new Error("Unsupported Sol parameter version");
  }
  if (propertyVersion !== undefined && widgetVersion !== undefined && propertyVersion !== widgetVersion)
    throw new Error("Conflicting Sol parameter versions");
  if ((propertyVersion ?? widgetVersion) === 2) return { ...SOL_DEFAULTS, ...byName, sol_version: 2 };
  if (Object.keys(SOL_DEFAULTS).some(k => k !== "sol_version" && k !== "sol_tau" && own(byName, k)))
    throw new Error("New Sol options require a saved version marker");
  const ratio = byName.sol_tau ?? 1;
  if (typeof ratio !== "number" || !Number.isFinite(ratio)) throw new Error("Invalid legacy Sol keep ratio");
  const enabled = ratio !== 0 && Math.abs(ratio - 1) > 1e-6;
  if (enabled && !(ratio > 0 && ratio < 1)) throw new Error("Legacy Sol keep ratio must be between 0 and 1");
  return { ...byName, ...SOL_DEFAULTS, ...(enabled ? { sol_enabled: true, sol_tau: legacyTau(ratio),
    sol_window_enabled: false, sol_start_percent: 0, sol_end_percent: 1, sol_min_tokens: 0,
    sol_sink_conditioning: "off" } : {}) };
}

function migratedInfo(comfyClass, widgets, info) {
  if (!info || !own(EARLIER, comfyClass)) return null;
  const serialized = widgets.filter(w => w.serialize !== false);
  const values = info.widgets_values;
  let byName = info.widgets_values_named;
  if (byName && typeof byName === "object" && !Array.isArray(byName)) {
    byName = { ...byName };
    const rawVersion = Array.isArray(values) && values.length === serialized.length
      ? values[serialized.findIndex(w => w.name === "sol_version")] : undefined;
    if (rawVersion !== undefined && byName.sol_version !== undefined && rawVersion !== byName.sol_version)
      throw new Error("Conflicting saved Sol widgets");
    if (rawVersion !== undefined) byName.sol_version = rawVersion;
  } else {
    if (values === undefined) return null;
    const earlier = EARLIER[comfyClass].find((order) => Array.isArray(values) && values.length === order.length
      && order.every(([, type], i) => !type || typeof values[i] === type));
    if (earlier) byName = Object.fromEntries(earlier.map(([name], i) => [name, values[i]]));
    else if (Array.isArray(values) && values.length === serialized.length)
      byName = Object.fromEntries(serialized.map((w, i) => [w.name, values[i]]));
    else throw new Error("Unknown QuantFunc loader layout; cannot safely migrate Sol settings");
  }
  if (typeof byName.quality === "string" && !own(byName, "quality_enhance")) {
    byName.quality_enhance = byName.quality === "best_quality";
  }
  byName = migrateSol(byName, info.properties?.qf_sol_version);
  // Update both formats: another extension must not restore the old ratio from the named map.
  const current = Object.fromEntries(serialized.map(w => [w.name, own(byName, w.name) ? byName[w.name] : w.value]));
  return { ...info, properties: { ...info.properties, qf_layout: LAYOUT, qf_sol_version: 2 },
    widgets_values: serialized.map(w => current[w.name]), widgets_values_named: current };
}

export function remappedValues(comfyClass, widgets, info) {
  return migratedInfo(comfyClass, widgets, info)?.widgets_values ?? null;
}

app.registerExtension({
  name: "QuantFunc.LoaderLayout",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isLoader(nodeData?.name)) return;
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const migrated = migratedInfo(this.comfyClass, this.widgets ?? [], info);
      const out = configure.call(this, migrated ?? info);
      this.properties.qf_layout = LAYOUT;
      if (migrated) this.properties.qf_sol_version = 2;
      return out;
    };
  },
  nodeCreated(node) {
    if (isLoader(node.comfyClass)) node.properties.qf_layout = LAYOUT;
    if (own(EARLIER, node.comfyClass)) node.properties.qf_sol_version = 2;
  },
});
