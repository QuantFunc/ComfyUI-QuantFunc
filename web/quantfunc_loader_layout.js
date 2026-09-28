// Loader layout (user 2026-09-28 「要输入的都在上方 开关的统一在下方」): on every QuantFunc loader the value inputs come first
// and the switches last, and the MiniMax-H3 loader no longer has its partial-denoise switch (double sampling always
// works). ComfyUI restores a saved node's widget values by POSITION, so a workflow saved with the previous order would put
// each value into the wrong widget. This remaps such a workflow once, on load, and stamps every loader node with the
// layout it is saved with, so a remapped or new node is never remapped again.
import { app } from "../../scripts/app.js";

const LAYOUT = 2;                  // node.properties.qf_layout of a loader saved with the current order
// Earlier saved orders the loaders may meet, with the type each number / switch / dropdown value had: a saved workflow
// is remapped only when its values have exactly one of these shapes (the current LTX order has as many values as the
// previous one, and a Krea-2 / Qwen-Image-2.1 order did not change at all). The release-candidate loaders had a `quality`
// dropdown where quality_enhance is now: best_quality maps to ON and any other value to OFF, as for an old API prompt.
const V = (name, type) => [name, type];
const EARLIER = {
  QuantFuncLTXLoader: [
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"), V("quality", "string"),
     V("step_cache", "number"), V("block_cache", "number")],
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"), V("quality_enhance", "boolean"),
     V("step_cache", "number"), V("block_cache", "number"), V("pinned_memory", "boolean")],
  ],
  QuantFuncH3Loader: [
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"), V("quality", "string"),
     V("audio_enhance", "boolean"), V("step_cache", "number"), V("block_cache", "number"),
     V("allow_partial_denoise", "boolean")],
    [V("transformer"), V("model_config"), V("attention_backend"), V("sol_tau", "number"), V("quality_enhance", "boolean"),
     V("audio_enhance", "boolean"), V("step_cache", "number"), V("block_cache", "number"),
     V("allow_partial_denoise", "boolean"), V("pinned_memory", "boolean")],
  ],
  QuantFuncKrea2Loader: [[V("transformer"), V("model_config"), V("attention_backend"), V("quality", "string")]],
  QuantFuncQwenImage21Loader: [[V("transformer"), V("model_config"), V("attention_backend"), V("quality", "string")]],
};

const isLoader = (name) => typeof name === "string" && name.startsWith("QuantFunc") && name.endsWith("Loader");

// The saved values in the node's current order, or null when the node needs no remap. Values saved by name (newer
// ComfyUI frontends save both) are placed by name; otherwise an earlier positional order is recognised by its shape.
// A value the node no longer has (the removed switch) is dropped; a widget with no saved value keeps its default.
export function remappedValues(comfyClass, widgets, info) {
  if (!info || Number(info.properties?.qf_layout) >= LAYOUT) return null;
  let byName = info.widgets_values_named;
  if (byName && typeof byName === "object") {
    byName = { ...byName };
  } else {
    const values = info.widgets_values;
    const earlier = (EARLIER[comfyClass] ?? []).find((order) => Array.isArray(values) && values.length === order.length
      && order.every(([, type], i) => !type || typeof values[i] === type));
    if (!earlier) return null;
    byName = Object.fromEntries(earlier.map(([name], i) => [name, values[i]]));
  }
  if (typeof byName.quality === "string" && !Object.prototype.hasOwnProperty.call(byName, "quality_enhance")) {
    byName.quality_enhance = byName.quality === "best_quality";
  }
  return widgets.filter((w) => w.serialize !== false)
    .map((w) => (Object.prototype.hasOwnProperty.call(byName, w.name) ? byName[w.name] : w.value));
}

app.registerExtension({
  name: "QuantFunc.LoaderLayout",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isLoader(nodeData?.name)) return;
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const values = remappedValues(this.comfyClass, this.widgets ?? [], info);
      const out = configure.call(this, values ? { ...info, widgets_values: values } : info);
      this.properties.qf_layout = LAYOUT;
      return out;
    };
  },
  nodeCreated(node) {
    if (isLoader(node.comfyClass)) node.properties.qf_layout = LAYOUT;
  },
});
