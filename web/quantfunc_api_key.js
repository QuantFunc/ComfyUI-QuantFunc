// The API key field of the QuantFunc loaders (user 2026-09-27): a key typed here wins and is saved to config.json; a
// field with nothing typed uses config.json's key and shows it, shortened. The field goes on exactly the QuantFunc nodes
// that declare the hidden `api_key` input (qf_api_key.add_api_key_input), so a field never exists that the loader ignores.
//
// ComfyUI saves every widget value into the workflow and every prompt input into each image and video it saves, and it
// has no secret widget for custom nodes. So this field keeps the key out of both:
//  * it is a password input that is never saved into a workflow (serialize = false): saved and exported workflows,
//    copy and paste, the browser's autosave and the workflow inside every image carry no key;
//  * a queued prompt carries only a reference: the key is POSTed to /quantfunc/api_key and stays in the ComfyUI
//    process (qf_api_key.py), so the prompt inside every image, the history and "Export (API)" carry no key either.
// The browser keeps no copy: config.json is where the key persists, and the server only ever sends it back shortened
// (user 2026-09-27: 部分明文): qf_ + its first and last 4 characters.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WITH_FIELD = new Set();             // node classes that get the field (beforeRegisterNodeDef fills it)
const ROUTE = "/quantfunc/api_key";
const FIELD_HEIGHT = 26;                  // px: one text line (the canvas adds the widget's margin around it)
const SHOWN_END = 4;                      // characters of a typed key shown at each end while the field is not edited

// A typed key as the field shows it while not edited: the prefix and SHOWN_END characters at each end, the rest elided.
// Text too short to elide shows only the prefix, so the field never shows a whole key.
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
    let typed = "";     // the key typed into this field ("": the loader uses config.json's key)
    let saved = "";     // config.json's key as the server shows it (shortened)
    let editing = false;
    const input = document.createElement("input");
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = "API key (none in config.json)";
    input.title = "QuantFunc API key, shown shortened; click to enter a new one. It is saved to config.json, never into " +
                  "a workflow or an image.";
    input.setAttribute("aria-label", "QuantFunc API key");
    const show = () => {
      input.type = editing ? "password" : "text";
      if (!editing) input.value = typed ? shortened(typed) : saved;
    };
    // Hand a typed key to ComfyUI: it keeps it behind a reference (returned) and saves a well-formed one to config.json.
    const post = async (key) => {
      const res = await api.fetchApi(ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      if (!res.ok) throw new Error(`QuantFunc: the API key could not be handed to ComfyUI (HTTP ${res.status}); nothing was queued.`);
      const answer = await res.json();
      saved = answer.shown ?? saved;
      return answer.ref;
    };
    input.addEventListener("focus", () => { editing = true; typed = ""; input.value = ""; show(); });
    input.addEventListener("input", () => { typed = input.value.trim(); });   // a prompt queued mid-edit sends it
    input.addEventListener("blur", () => {
      editing = false;
      show();
      if (typed) post(typed).then(show, () => {});   // a refused POST fails the queue loudly instead
    });
    let widget;
    // The canvas draws a DOM widget's element inside the widget's margin on each side, so the row is the field plus both.
    const rowHeight = () => FIELD_HEIGHT + 2 * (widget?.margin ?? 0);
    widget = node.addDOMWidget("api_key", "password", input, {
      getValue: () => typed,
      setValue: (value) => { typed = value ?? ""; show(); },
      getMinHeight: rowHeight,
      getMaxHeight: rowHeight,
    });
    widget.serialize = false;   // never in a saved workflow
    widget.serializeValue = async () => (typed ? post(typed) : "");   // "" or a reference, never the key
    api.fetchApi(ROUTE).then((res) => (res.ok ? res.json() : {})).then((answer) => { saved = answer.shown ?? ""; show(); },
                                                                      () => {});
    show();
  },
});
