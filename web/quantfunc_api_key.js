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
// The key is remembered in this browser only (localStorage), never on the server's disk, and every new field shows it
// again, shortened (user 2026-09-27: 部分明文): qf_ + its first and last 4 characters. Being edited, the field holds the
// whole key, masked.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WITH_FIELD = new Set();             // node classes that get the field (beforeRegisterNodeDef fills it)
const REMEMBERED = "QuantFunc.api_key";   // localStorage item
const FIELD_HEIGHT = 26;                  // px: one text line (the canvas adds the widget's margin around it)
const SHOWN_END = 4;                      // characters of the key shown at each end while the field is not edited

// The key as a field not being edited shows it: the prefix and SHOWN_END characters at each end, the rest elided. Text
// too short to elide shows only the prefix, so the field never shows a whole key.
function shortened(key) {
  const head = key.startsWith("qf_") ? "qf_" : "";
  const body = key.slice(head.length);
  if (!key) return "";
  return body.length > 2 * SHOWN_END ? `${head}${body.slice(0, SHOWN_END)}…${body.slice(-SHOWN_END)}` : `${head}…`;
}

app.registerExtension({
  name: "QuantFunc.ApiKey",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name?.startsWith("QuantFunc") && nodeData?.input?.hidden?.api_key) WITH_FIELD.add(nodeData.name);
  },
  nodeCreated(node) {
    if (!WITH_FIELD.has(node.comfyClass)) return;
    let key = localStorage.getItem(REMEMBERED) ?? "";   // the whole key; the field shows it only masked, while edited
    let editing = false;
    const input = document.createElement("input");
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = "API key (empty: the key in config.json)";
    input.title = "QuantFunc API key, shown shortened; click to change it. Remembered in this browser, never saved into " +
                  "a workflow or an image. Empty: the key in config.json.";
    input.setAttribute("aria-label", "QuantFunc API key");
    const show = () => {
      input.type = editing ? "password" : "text";
      input.value = editing ? key : shortened(key);
    };
    input.addEventListener("focus", () => { editing = true; show(); input.select(); });
    input.addEventListener("input", () => {   // every keystroke: a prompt queued while still editing sends what was typed
      key = input.value;
      if (key.trim()) localStorage.setItem(REMEMBERED, key.trim());
      else localStorage.removeItem(REMEMBERED);
    });
    input.addEventListener("blur", () => { editing = false; key = key.trim(); show(); });
    show();
    let widget;
    // The canvas draws a DOM widget's element inside the widget's margin on each side, so the row is the field plus both.
    const rowHeight = () => FIELD_HEIGHT + 2 * (widget?.margin ?? 0);
    widget = node.addDOMWidget("api_key", "password", input, {
      getValue: () => key,
      setValue: (value) => { key = value ?? ""; show(); },
      getMinHeight: rowHeight,
      getMaxHeight: rowHeight,
    });
    widget.serialize = false;   // never in a saved workflow
    widget.serializeValue = async () => {   // what a queued prompt carries: "" or a reference, never the key
      const whole = key.trim();
      if (!whole) return "";
      const res = await api.fetchApi("/quantfunc/api_key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: whole }),
      });
      if (!res.ok) throw new Error(`QuantFunc: the API key could not be handed to ComfyUI (HTTP ${res.status}); nothing was queued.`);
      return (await res.json()).ref;
    };
  },
});
