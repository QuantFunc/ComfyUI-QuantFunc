// The API key field of the four QuantFunc loaders (user 2026-09-27): a valid key typed here wins, and config.json is
// then not read. An empty field leaves the key to config.json, as before.
//
// ComfyUI saves every widget value into the workflow and every prompt input into each image and video it saves, and it
// has no secret widget for custom nodes. So this field keeps the key out of both:
//  * it is a password input that is never saved into a workflow (serialize = false): saved and exported workflows,
//    copy and paste, the browser's autosave and the workflow inside every image carry no key;
//  * a queued prompt carries only a reference: the key is POSTed to /quantfunc/api_key and stays in the ComfyUI
//    process (qf_api_key.py), so the prompt inside every image, the history and "Export (API)" carry no key either.
// The key is remembered in this browser, as ComfyUI keeps its own comfy.org API key, and never on the server's disk.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOADERS = new Set(["QuantFuncLTXLoader", "QuantFuncH3Loader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader"]);
const REMEMBERED = "QuantFunc.api_key";   // localStorage item
const FIELD_HEIGHT = 26;                  // px: one text line (the canvas adds the widget's margin around it)

app.registerExtension({
  name: "QuantFunc.ApiKey",
  nodeCreated(node) {
    if (!LOADERS.has(node.comfyClass)) return;
    const input = document.createElement("input");
    input.type = "password";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = "API key (empty: the key in config.json)";
    input.title = "QuantFunc API key. Remembered in this browser, never saved into a workflow or an image. " +
                  "Empty: the key in config.json.";
    input.setAttribute("aria-label", "QuantFunc API key");
    input.value = localStorage.getItem(REMEMBERED) ?? "";
    input.addEventListener("change", () => {
      const key = input.value.trim();
      if (key) localStorage.setItem(REMEMBERED, key);
      else localStorage.removeItem(REMEMBERED);
    });
    let widget;
    // The canvas draws a DOM widget's element inside the widget's margin on each side, so the row is the field plus both.
    const rowHeight = () => FIELD_HEIGHT + 2 * (widget?.margin ?? 0);
    widget = node.addDOMWidget("api_key", "password", input, {
      getValue: () => input.value,
      setValue: (value) => { input.value = value ?? ""; },
      getMinHeight: rowHeight,
      getMaxHeight: rowHeight,
    });
    widget.serialize = false;   // never in a saved workflow
    widget.serializeValue = async () => {   // what a queued prompt carries: "" or a reference, never the key
      const key = input.value.trim();
      if (!key) return "";
      const res = await api.fetchApi("/quantfunc/api_key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      if (!res.ok) throw new Error(`QuantFunc: the API key could not be handed to ComfyUI (HTTP ${res.status}); nothing was queued.`);
      return (await res.json()).ref;
    };
  },
});
