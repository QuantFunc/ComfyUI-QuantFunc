// The API key row of the QuantFunc loaders (user 2026-09-27): a key typed here wins and is saved to config.json; a row
// with nothing typed uses config.json's key and shows it, shortened. It is ComfyUI's native text widget (user 2026-09-28:
// label on the left, value on the right, like every other row), placed with the value inputs above the switches, on
// exactly the QuantFunc nodes that declare the hidden `api_key` input (qf_api_key.add_api_key_input).
//
// ComfyUI saves every widget value into the workflow and every prompt input into each image and video it saves, and it
// has no secret widget for custom nodes. So the row keeps the key out of both:
//  * it is never saved into a workflow (serialize = false): saved and exported workflows, copy and paste, undo and the
//    workflow inside every image carry no key;
//  * the widget's value is only ever the SHORTENED key (qf_ + its first and last 4 characters): the whole key lives in
//    this script, never on the widget. Clicking the row opens ComfyUI's value dialog EMPTY and masked, and the typed
//    key goes straight here; in the Vue node view the inline input holds it (masked) only while it is being typed;
//  * a queued prompt carries only a reference: the key is POSTed to /quantfunc/api_key and stays in the ComfyUI
//    process (qf_api_key.py), so the prompt inside every image, the history and "Export (API)" carry no key either.
// The browser keeps no copy: config.json is where the key persists, and the server only ever sends it back shortened.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WITH_FIELD = new Set();             // node classes that get the row (beforeRegisterNodeDef fills it)
const ROWS = new Set();                   // each live row's refresh: a key saved from one row is every row's config.json key
const ROUTE = "/quantfunc/api_key";
const SHOWN_END = 4;                      // characters of a typed key shown at each end
const SWITCH_TYPES = new Set(["toggle", "boolean"]);   // the BOOLEAN widgets the row goes above

// A typed key as the row shows it: the prefix and SHOWN_END characters at each end, the rest elided. Text too short to
// elide shows only the prefix, so the row never shows a whole key.
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
    let typed = "";     // the key typed in ("": the loader uses config.json's key); never on the widget
    let saved = "";     // config.json's key as the server shows it (shortened)
    const widget = node.addWidget("text", "api_key", "", () => {}, {});
    widget.tooltip = "QuantFunc API key, shown shortened; click to enter a new one. It is saved to config.json, never " +
                     "into a workflow or an image. Empty: the key in config.json.";
    const show = () => {
      widget.value = typed ? shortened(typed) : saved;
      node.setDirtyCanvas?.(true, true);
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
      if (answer.shown !== undefined) for (const refresh of ROWS) refresh(answer.shown);
      return answer.ref;
    };
    const refresh = (shown) => { saved = shown; show(); };
    ROWS.add(refresh);
    const onRemoved = node.onRemoved;
    node.onRemoved = function () { ROWS.delete(refresh); return onRemoved?.apply(this, arguments); };
    // A finished entry: the text the user typed (the shortened form itself, left as it was, is no new key).
    const commit = (text) => {
      if (text === null || text === undefined) return;             // the dialog was cancelled
      const whole = String(text).trim();
      if (whole.includes("…")) { show(); return; }
      typed = whole;
      show();
      if (typed) post(typed).then(show, () => {});                 // a refused POST fails the queue loudly instead
    };
    // Canvas: ComfyUI's own value dialog, opened EMPTY and masked, so the whole key never passes through widget.value.
    widget.onClick = ({ e, canvas }) => {
      canvas.prompt("API key", "", commit, e)?.querySelector?.("input.value")?.setAttribute("type", "password");
    };
    // Vue node view: the inline input writes widget.value and calls this on every keystroke, so the entry is committed
    // (and the row shortened) when that input loses focus.
    widget.callback = (value) => {
      const input = document.activeElement;
      if (input instanceof HTMLInputElement && input.closest?.(`[data-node-id="${node.id}"]`)) {
        typed = String(value ?? "").trim();
        if (!input.dataset.qfApiKey) {
          input.dataset.qfApiKey = "1";
          input.type = "password";                    // masked while it is typed
          input.addEventListener("blur", () => {
            delete input.dataset.qfApiKey;
            input.type = "text";
            commit(input.value);
          }, { once: true });
        }
        return;
      }
      commit(value);
    };
    widget.serialize = false;   // never in a saved workflow
    widget.serializeValue = async () => (typed ? post(typed) : "");   // "" or a reference, never the key
    // With the value inputs: just above the first switch.
    const first = node.widgets.findIndex((w) => w !== widget && SWITCH_TYPES.has(w.type));
    if (first >= 0) node.widgets.splice(first, 0, ...node.widgets.splice(node.widgets.indexOf(widget), 1));
    api.fetchApi(ROUTE).then((res) => (res.ok ? res.json() : {})).then((answer) => { saved = answer.shown ?? ""; show(); },
                                                                      () => {});
    show();
  },
});
