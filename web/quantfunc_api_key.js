// The API key field of the QuantFunc loaders (user 2026-09-27): a valid key typed here wins, and config.json is then
// not read. An empty field leaves the key to config.json, as before. The field goes on exactly the QuantFunc nodes that
// declare the hidden `api_key` input (qf_api_key.add_api_key_input), so a field never exists that the loader ignores.
//
// ComfyUI saves every widget value into the workflow and every prompt input into each image and video it saves, and it
// has no secret widget for custom nodes. So this field keeps the key out of both:
//  * it is a password input that is never saved into a workflow (serialize = false): saved and exported workflows,
//    copy and paste, the browser's autosave and the workflow inside every image carry no key;
//  * a queued prompt carries only a reference: the key is POSTed to /quantfunc/api_key and stays in the ComfyUI
//    process (qf_api_key.py), so the prompt inside every image, the history and "Export (API)" carry no key either.
// The key is remembered in this browser only (localStorage), never on the server's disk.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WITH_FIELD = new Set();             // node classes that get the field (beforeRegisterNodeDef fills it)
const REMEMBERED = "QuantFunc.api_key";   // localStorage item
const FIELD_HEIGHT = 26;                  // px: one text line (the canvas adds the widget's margin around it)

app.registerExtension({
  name: "QuantFunc.ApiKey",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name?.startsWith("QuantFunc") && nodeData?.input?.hidden?.api_key) WITH_FIELD.add(nodeData.name);
  },
  nodeCreated(node) {
    if (!WITH_FIELD.has(node.comfyClass)) return;
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
