#!/usr/bin/env python3
"""Contract and mutation death-rules for all shipped example_workflows/*.json.

Validates every shipped workflow against the plugin's real __init__.py loader INPUT_TYPES(),
qf_sol.from_loader(), configs/*/qf_native.json file_hints, graph topology invariants,
model/VAE/LoRA/sampler contracts, MarkdownNote parity, and security hygiene.

Pure CPU, stdlib + dependency-free plugin modules (qf_sol, qf_file_hints); zero GPU/CUDA,
zero SKIP arms. Auto-discovered by tests/run_plugin_tests.py.

Run:  CUDA_VISIBLE_DEVICES="" python3 tests/example_workflows_contract_test.py
"""
import ast
import copy
import json
import math
from pathlib import Path
import re
import sys
import unittest

PLUGIN = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = PLUGIN / "example_workflows"
sys.path.insert(0, str(PLUGIN))
from qf_file_hints import name_matches_hints  # noqa: E402
import qf_sol  # noqa: E402

EXPECTED_WORKFLOWS = {
    "QuantFunc-Krea2-t2i.json",
    "QuantFunc-Krea2-t2i-double-sampling.json",
    "QuantFunc-LTX25-t2v.json",
    "QuantFunc-MiniMaxH3-fl2va.json",
    "QuantFunc-MiniMaxH3-fl2va-double-sampling.json",
    "QuantFunc-MiniMaxH3-ref2va.json",
    "QuantFunc-MiniMaxH3-ref2va-double-sampling.json",
    "QuantFunc-QwenImage21-edit.json",
    "QuantFunc-QwenImage21-edit-multi-reference.json",
    "QuantFunc-QwenImage21-edit-remove-background.json",
    "QuantFunc-QwenImage21-t2i.json",
    "QuantFunc-QwenImage21-t2i-transparent.json",
}

QI21_LORA_FILE = "qwen_image_21_lightning_v2_rl_4steps_lora_rank64.safetensors"
H3_VIDEO_VAE_FILE = "minimax_h3_video_vae_int8_convrot.safetensors"
H3_AUDIO_VAE_FILE = "minimax_h3_audio_vae_fp32.safetensors"
_PATH_WIDGET_NODES = {
    "CLIPLoader": 0,
    "VAELoader": 0,
    "QuantFuncNativeLoRA": 0,
    "LatentUpscaleModelLoader": 0,
    "LoadImage": 0,
    "SaveImage": 0,
    "SaveVideo": 0,
}
_PATH_NAMED_KEYS = {
    "transformer", "model_config", "vae_name", "clip_name",
    "lora_name", "model_name", "image", "filename_prefix",
}
_ASSET_EXT_RE = re.compile(r"\.(?:safetensors|ckpt|gguf|pth?|bin|onnx|png|jpe?g|webp)$", re.I)
_QF_KEY_RE = re.compile(r"\bqf_[0-9a-f]{64}\b")


def _is_strict_int(val):
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_number(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def _no_duplicate_keys_hook(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"Duplicate JSON key: {k!r}")
        out[k] = v
    return out


def _load_presets_by_family():
    presets = {}
    hints = {}
    for mf in sorted((PLUGIN / "configs").glob("*/qf_native.json")):
        cfg_name = mf.parent.name
        data = json.loads(mf.read_text(encoding="utf-8"))
        fam = str(data["family"])
        presets.setdefault(fam, []).append(cfg_name)
        hints[cfg_name] = (data.get("file_hints") or {}).get("transformer1") or []
    return presets, hints


def _eval_ast_spec(node, constants, family_presets):
    """Resolve an INPUT_TYPES widget spec AST node from __init__.py."""
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        if fn == "_transformer_choices":
            return ("COMBO_FILE", {"suffix": ".safetensors"})
        if fn == "_attn_backend_input":
            return (list(constants["_ATTN_BACKENDS"]), {"default": "auto"})
        if fn == "_model_config_input":
            fam = ast.literal_eval(node.args[0])
            return (list(family_presets[fam]), {"default": family_presets[fam][0]})
        if fn == "_pinned_memory_input":
            default_on = ast.literal_eval(node.args[0])
            return ("BOOLEAN", {"default": default_on})
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        first = _eval_ast_spec(node.elts[0], constants, family_presets)
        second = _eval_ast_spec(node.elts[1], constants, family_presets)
        return (first, second)
    return ast.literal_eval(node)


def _normalize_widget_spec(raw_spec):
    kind_or_choices, opts = raw_spec
    opts = opts if isinstance(opts, dict) else {}
    if kind_or_choices == "COMBO_FILE":
        return {"kind": "COMBO_FILE", "suffix": opts.get("suffix", ".safetensors")}
    if isinstance(kind_or_choices, list):
        return {"kind": "COMBO", "choices": list(kind_or_choices)}
    return {
        "kind": str(kind_or_choices),
        "min": opts.get("min"),
        "max": opts.get("max"),
    }


def load_loader_schemas():
    """Parse QuantFunc*Loader INPUT_TYPES() directly from __init__.py and configs/*/qf_native.json."""
    tree = ast.parse((PLUGIN / "__init__.py").read_text(encoding="utf-8"))
    family_presets, hints = _load_presets_by_family()

    # Collect module-level and top-level `if _IMPORT_OK:` constant bindings needed by INPUT_TYPES.
    constants = {}
    stmts = list(tree.body)
    for top in tree.body:
        if isinstance(top, ast.If) and isinstance(top.test, ast.Name) and top.test.id == "_IMPORT_OK":
            stmts.extend(top.body)
    for stmt in stmts:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name.startswith("_"):
                try:
                    constants[name] = _eval_ast_spec(stmt.value, constants, family_presets)
                except Exception:
                    continue

    schemas = {}
    for cls_node in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if not (cls_node.name.startswith("QuantFunc") and cls_node.name.endswith("Loader")):
            continue
        family = None
        for item in cls_node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                if item.targets[0].id == "QF_FAMILY":
                    family = ast.literal_eval(item.value)
        fn = next(m for m in cls_node.body if isinstance(m, ast.FunctionDef) and m.name == "INPUT_TYPES")
        ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return)).value
        widgets = []
        for part_key, part_dict in zip(ret.keys, ret.values):
            if isinstance(part_key, ast.Constant) and part_key.value in ("required", "optional"):
                for k_node, v_node in zip(part_dict.keys, part_dict.values):
                    wname = k_node.value
                    wspec = _normalize_widget_spec(_eval_ast_spec(v_node, constants, family_presets))
                    widgets.append((wname, wspec))
        schemas[cls_node.name] = {
            "family": family,
            "widgets": widgets,
            "hints": hints,
        }
    return schemas


def _check_safe_relpath(val, label):
    if not isinstance(val, str):
        raise ValueError(f"{label}: expected string path, got {type(val).__name__}")
    if "\\" in val:
        raise ValueError(f"{label}: backslash not allowed in workflow asset path {val!r}")
    if val.startswith("/") or re.match(r"^[A-Za-z]:", val):
        raise ValueError(f"{label}: absolute path not allowed in {val!r}")
    if ".." in val.split("/"):
        raise ValueError(f"{label}: path traversal '..' not allowed in {val!r}")


def _check_hygiene(name, raw_text, doc):
    if _QF_KEY_RE.search(raw_text):
        raise ValueError(f"{name}: forbidden literal QuantFunc API key detected")

    def _walk_keys(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() == "api_key":
                    raise ValueError(f"{name}: forbidden 'api_key' field at {path}.{k}")
                _walk_keys(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _walk_keys(item, f"{path}[{idx}]")

    _walk_keys(doc, "$")

    for node in doc["nodes"]:
        nid = node.get("id")
        ntype = node.get("type") or ""
        if ntype == "QuantFuncApiKey" or "api_key" in ntype.lower():
            raise ValueError(f"{name}: forbidden API-key node id={nid} type={ntype!r}")
        for inp in node.get("inputs") or []:
            if str(inp.get("name", "")).lower() == "api_key":
                raise ValueError(f"{name}: forbidden 'api_key' input slot on node id={nid}")
        if ntype == "MarkdownNote":
            continue
        wv = node.get("widgets_values")
        if isinstance(wv, list):
            if ntype in _PATH_WIDGET_NODES:
                idx = _PATH_WIDGET_NODES[ntype]
                if idx < len(wv) and isinstance(wv[idx], str):
                    _check_safe_relpath(wv[idx], f"{name}: node {nid} ({ntype}).widgets_values[{idx}]")
            for idx, val in enumerate(wv):
                if isinstance(val, str) and (_ASSET_EXT_RE.search(val) or ".." in val.replace("\\", "/").split("/")):
                    _check_safe_relpath(val, f"{name}: node {nid} ({ntype}).widgets_values[{idx}]")
        wvn = node.get("widgets_values_named")
        if isinstance(wvn, dict):
            for k, val in wvn.items():
                if isinstance(val, str) and (
                    k in _PATH_NAMED_KEYS
                    or _ASSET_EXT_RE.search(val)
                    or ".." in val.replace("\\", "/").split("/")
                ):
                    _check_safe_relpath(val, f"{name}: node {nid} ({ntype}).widgets_values_named[{k!r}]")


def _check_topology(name, doc):
    nodes = doc.get("nodes")
    links = doc.get("links")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{name}: 'nodes' must be a non-empty list")
    if not isinstance(links, list):
        raise ValueError(f"{name}: 'links' must be a list")

    nodes_by_id = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError(f"{name}: node entry must be an object")
        nid = node.get("id")
        if not _is_strict_int(nid):
            raise ValueError(f"{name}: node id must be strict int, got {nid!r}")
        if nid in nodes_by_id:
            raise ValueError(f"{name}: duplicate node id {nid}")
        nodes_by_id[nid] = node

    last_node_id = doc.get("last_node_id")
    last_link_id = doc.get("last_link_id")
    if not _is_strict_int(last_node_id) or last_node_id < max(nodes_by_id):
        raise ValueError(f"{name}: last_node_id={last_node_id!r} < max(node_ids)={max(nodes_by_id)}")

    links_by_id = {}
    connected = set()
    for link in links:
        if not isinstance(link, list) or len(link) != 6:
            raise ValueError(f"{name}: malformed link entry {link!r}")
        lid, src_id, src_slot, dst_id, dst_slot, ltype = link
        if not all(_is_strict_int(x) for x in (lid, src_id, src_slot, dst_id, dst_slot)) or not isinstance(ltype, str):
            raise ValueError(f"{name}: invalid link field types in {link!r}")
        if lid in links_by_id:
            raise ValueError(f"{name}: duplicate link id {lid}")
        links_by_id[lid] = link
        if src_id not in nodes_by_id or dst_id not in nodes_by_id:
            raise ValueError(f"{name}: dangling link {lid} ({src_id} -> {dst_id})")
        connected.add(src_id)
        connected.add(dst_id)

        src_outputs = nodes_by_id[src_id].get("outputs") or []
        if not (0 <= src_slot < len(src_outputs)):
            raise ValueError(f"{name}: link {lid} invalid src_slot={src_slot} on node {src_id}")
        if lid not in (src_outputs[src_slot].get("links") or []):
            raise ValueError(f"{name}: link {lid} missing from node {src_id}.outputs[{src_slot}].links")
        if src_outputs[src_slot].get("type") != ltype:
            raise ValueError(
                f"{name}: link {lid} src slot type mismatch "
                f"({src_outputs[src_slot].get('type')!r} != {ltype!r})"
            )

        dst_inputs = nodes_by_id[dst_id].get("inputs") or []
        if not (0 <= dst_slot < len(dst_inputs)):
            raise ValueError(f"{name}: link {lid} invalid dst_slot={dst_slot} on node {dst_id}")
        if dst_inputs[dst_slot].get("link") != lid:
            raise ValueError(f"{name}: link {lid} dst node {dst_id}.inputs[{dst_slot}].link mismatch")
        if dst_inputs[dst_slot].get("type") != ltype:
            raise ValueError(
                f"{name}: link {lid} dst slot type mismatch "
                f"({dst_inputs[dst_slot].get('type')!r} != {ltype!r})"
            )

    max_lid = max(links_by_id) if links_by_id else 0
    if not _is_strict_int(last_link_id) or last_link_id < max_lid:
        raise ValueError(f"{name}: last_link_id={last_link_id!r} < max(link_ids)={max_lid}")

    for nid, node in nodes_by_id.items():
        for s_idx, out in enumerate(node.get("outputs") or []):
            for lid in (out.get("links") or []):
                if lid not in links_by_id or links_by_id[lid][1] != nid or links_by_id[lid][2] != s_idx:
                    raise ValueError(f"{name}: node {nid}.outputs[{s_idx}] dangling/mismatched ref {lid}")
        for i_idx, inp in enumerate(node.get("inputs") or []):
            lid = inp.get("link")
            if lid is not None:
                if lid not in links_by_id or links_by_id[lid][3] != nid or links_by_id[lid][4] != i_idx:
                    raise ValueError(f"{name}: node {nid}.inputs[{i_idx}] dangling/mismatched ref {lid}")
        if node.get("type") != "MarkdownNote" and nid not in connected:
            raise ValueError(f"{name}: orphan node id={nid} type={node.get('type')!r}")


def _validate_loader_node(name, node, loader_schemas):
    nid = node["id"]
    ntype = node["type"]
    if ntype not in loader_schemas:
        raise ValueError(f"{name}: unknown QuantFunc loader class {ntype!r} on node {nid}")
    schema = loader_schemas[ntype]
    props = node.get("properties")
    if not isinstance(props, dict):
        raise ValueError(f"{name}: node {nid} ({ntype}) missing properties dict")
    if props.get("qf_layout") != 3 or isinstance(props.get("qf_layout"), bool):
        raise ValueError(f"{name}: node {nid} ({ntype}) properties.qf_layout={props.get('qf_layout')!r} != 3")
    if props.get("qf_sol_version") != 2 or isinstance(props.get("qf_sol_version"), bool):
        raise ValueError(f"{name}: node {nid} ({ntype}) properties.qf_sol_version={props.get('qf_sol_version')!r} != 2")
    if props.get("Node name for S&R") != ntype:
        raise ValueError(f"{name}: node {nid} ({ntype}) S&R={props.get('Node name for S&R')!r} != {ntype!r}")

    wv = node.get("widgets_values")
    expected_widgets = schema["widgets"]
    if not isinstance(wv, list) or len(wv) != len(expected_widgets):
        raise ValueError(
            f"{name}: node {nid} ({ntype}) widgets_values len={len(wv) if isinstance(wv, list) else None} "
            f"!= {len(expected_widgets)}"
        )

    by_name = {}
    for (wname, spec), val in zip(expected_widgets, wv):
        kind = spec["kind"]
        label = f"{name}: node {nid} ({ntype}).{wname}"
        if kind == "BOOLEAN":
            if not isinstance(val, bool):
                raise ValueError(f"{label}={val!r} must be strict bool")
        elif kind == "INT":
            if not _is_strict_int(val):
                raise ValueError(f"{label}={val!r} must be strict int")
            if spec["min"] is not None and val < spec["min"]:
                raise ValueError(f"{label}={val!r} < min {spec['min']}")
            if spec["max"] is not None and val > spec["max"]:
                raise ValueError(f"{label}={val!r} > max {spec['max']}")
        elif kind == "FLOAT":
            if not _is_finite_number(val):
                raise ValueError(f"{label}={val!r} must be finite float")
            if spec["min"] is not None and val < spec["min"]:
                raise ValueError(f"{label}={val!r} < min {spec['min']}")
            if spec["max"] is not None and val > spec["max"]:
                raise ValueError(f"{label}={val!r} > max {spec['max']}")
        elif kind == "COMBO":
            if not isinstance(val, str) or val not in spec["choices"]:
                raise ValueError(f"{label}={val!r} not in choices {spec['choices']}")
        elif kind == "COMBO_FILE":
            if not isinstance(val, str) or not val.lower().endswith(spec["suffix"]):
                raise ValueError(f"{label}={val!r} must end with {spec['suffix']}")
            _check_safe_relpath(val, label)
        else:
            if not isinstance(val, str):
                raise ValueError(f"{label}={val!r} must be string")
        by_name[wname] = val

    if "widgets_values_named" in node:
        wvn = node["widgets_values_named"]
        if not isinstance(wvn, dict) or set(wvn) != set(by_name):
            raise ValueError(f"{name}: node {nid} ({ntype}) corrupted widgets_values_named keys")
        for k, v in wvn.items():
            ref = by_name[k]
            if isinstance(v, bool) != isinstance(ref, bool) or v != ref:
                raise ValueError(f"{name}: node {nid} ({ntype}).widgets_values_named[{k!r}]={v!r} != {ref!r}")

    if by_name.get("attention_backend") != "auto":
        raise ValueError(f"{name}: node {nid} ({ntype}).attention_backend={by_name.get('attention_backend')!r} != 'auto'")
    if by_name.get("sol_version") != 2:
        raise ValueError(f"{name}: node {nid} ({ntype}).sol_version={by_name.get('sol_version')!r} != 2")
    if "block_cache" in by_name and by_name["block_cache"] > 0.5:
        raise ValueError(f"{name}: node {nid} ({ntype}).block_cache={by_name['block_cache']!r} > 0.5")

    sol_kwargs = {k: by_name[k] for k in by_name if k.startswith("sol_")}
    qf_sol.from_loader(**sol_kwargs)

    hints = schema["hints"].get(by_name["model_config"], [])
    if hints and not name_matches_hints(by_name["transformer"], hints):
        raise ValueError(
            f"{name}: node {nid} ({ntype}).transformer={by_name['transformer']!r} "
            f"does not match file_hints for {by_name['model_config']!r}"
        )
    return by_name


QI21_BASE_ALPHA_WORKFLOWS = frozenset({
    "QuantFunc-QwenImage21-t2i-transparent.json",
})
QI21_FORBIDDEN_ALPHA_LORA_TYPES = frozenset({
    "QuantFuncNativeLoRA",
    "LoraLoader",
    "LoraLoaderModelOnly",
})


def _validate_qi21(name, doc, loaders):
    qi_loaders = [(n, vals) for n, vals in loaders if n.get("type") == "QuantFuncQwenImage21Loader"]
    if not qi_loaders:
        raise ValueError(f"{name}: missing QuantFuncQwenImage21Loader")
    for n, vals in qi_loaders:
        if vals.get("sol_min_tokens") != 0 or isinstance(vals.get("sol_min_tokens"), bool):
            raise ValueError(f"{name}: QI-2.1 loader sol_min_tokens={vals.get('sol_min_tokens')!r} != 0")
        if vals.get("sol_enabled") is not False:
            raise ValueError(f"{name}: QI-2.1 loader sol_enabled={vals.get('sol_enabled')!r} is not False")
        if vals.get("quality_enhance") is not True:
            raise ValueError(f"{name}: QI-2.1 loader quality_enhance={vals.get('quality_enhance')!r} is not True")
        if vals.get("pinned_memory") is not True:
            raise ValueError(f"{name}: QI-2.1 loader pinned_memory={vals.get('pinned_memory')!r} is not True")

    samplers = [n for n in doc["nodes"] if n.get("type") == "KSampler" and n.get("mode", 0) == 0]
    if not samplers:
        raise ValueError(f"{name}: QI-2.1 workflow missing active KSampler")
    loader_ids = {n["id"] for n, _ in qi_loaders if n.get("mode", 0) == 0}
    sampler_ids = {n["id"] for n in samplers}
    model_edges = {(l[1], l[3]) for l in doc["links"] if l[5] == "MODEL"}
    loras = [n for n in doc["nodes"] if n.get("type") == "QuantFuncNativeLoRA"]

    if name in QI21_BASE_ALPHA_WORKFLOWS:
        # Pure T2I transparent PNG requires base model (the 4-step RL Lightning LoRA suppresses RGBA alpha).
        forbidden_loras = [n for n in doc["nodes"] if n.get("type") in QI21_FORBIDDEN_ALPHA_LORA_TYPES]
        if forbidden_loras:
            bad_types = sorted({n.get("type") for n in forbidden_loras})
            raise ValueError(
                f"{name}: base RGBA transparent workflow must not attach LoRA nodes {bad_types!r} "
                f"(the 4-step Lightning LoRA breaks text-to-image alpha transparency)"
            )
        for ks in samplers:
            kwv = ks.get("widgets_values")
            if not isinstance(kwv, list) or len(kwv) < 7:
                raise ValueError(f"{name}: malformed KSampler widgets_values {kwv!r}")
            if not _is_strict_int(kwv[2]) or kwv[2] < 20:
                raise ValueError(f"{name}: QI-2.1 base transparent KSampler steps={kwv[2]!r} < 20")
            wvn = ks.get("widgets_values_named")
            if isinstance(wvn, dict) and "steps" in wvn:
                wvn_steps = wvn["steps"]
                if not _is_strict_int(wvn_steps) or wvn_steps < 20 or wvn_steps != kwv[2]:
                    raise ValueError(
                        f"{name}: QI-2.1 base transparent KSampler widgets_values_named['steps']="
                        f"{wvn_steps!r} invalid or disagrees with widgets_values[2]={kwv[2]!r}"
                    )
            sid = ks["id"]
            incoming_models = [src for src, dst in model_edges if dst == sid]
            if not incoming_models or not all(src in loader_ids for src in incoming_models):
                raise ValueError(
                    f"{name}: QI-2.1 base transparent workflow KSampler id={sid} must wire Loader -> KSampler directly"
                )
        return

    if not loras:
        raise ValueError(f"{name}: QI-2.1 workflow missing QuantFuncNativeLoRA node")
    active_loras = [n for n in loras if n.get("mode", 0) == 0]
    if not active_loras:
        raise ValueError(f"{name}: QI-2.1 QuantFuncNativeLoRA must be active (mode == 0)")
    for lora in active_loras:
        lwv = lora.get("widgets_values")
        if not isinstance(lwv, list) or len(lwv) < 2:
            raise ValueError(f"{name}: malformed QuantFuncNativeLoRA widgets_values {lwv!r}")
        if lwv[0] != QI21_LORA_FILE:
            raise ValueError(f"{name}: QI-2.1 LoRA file {lwv[0]!r} != {QI21_LORA_FILE!r}")
        if not _is_finite_number(lwv[1]) or lwv[1] != 1.0:
            raise ValueError(f"{name}: QI-2.1 LoRA strength {lwv[1]!r} != 1.0")

    for ks in samplers:
        kwv = ks.get("widgets_values")
        if not isinstance(kwv, list) or len(kwv) < 7:
            raise ValueError(f"{name}: malformed KSampler widgets_values {kwv!r}")
        if not _is_strict_int(kwv[2]) or kwv[2] != 6:
            raise ValueError(f"{name}: QI-2.1 KSampler steps={kwv[2]!r} != 6")
        if kwv[4] != "res_multistep":
            raise ValueError(f"{name}: QI-2.1 KSampler sampler_name={kwv[4]!r} != 'res_multistep'")

    lora_ids = {n["id"] for n in active_loras}
    if not any((lid, loid) in model_edges and (loid, sid) in model_edges
               for lid in loader_ids for loid in lora_ids for sid in sampler_ids):
        raise ValueError(f"{name}: QI-2.1 must wire QuantFuncQwenImage21Loader -> QuantFuncNativeLoRA -> KSampler")
    if any((lid, sid) in model_edges for lid in loader_ids for sid in sampler_ids):
        raise ValueError(f"{name}: QI-2.1 must not bypass QuantFuncNativeLoRA with direct Loader -> KSampler link")


def _validate_ltx25(name, loaders):
    ltx_loaders = [(n, vals) for n, vals in loaders if n.get("type") == "QuantFuncLTXLoader"]
    if not ltx_loaders:
        raise ValueError(f"{name}: missing QuantFuncLTXLoader")
    for n, vals in ltx_loaders:
        if vals.get("sol_min_tokens") != 0 or isinstance(vals.get("sol_min_tokens"), bool):
            raise ValueError(f"{name}: LTX-2.5 loader sol_min_tokens={vals.get('sol_min_tokens')!r} != 0")


def _validate_minimax_h3(name, raw_text, doc, loaders):
    h3_loaders = [(n, vals) for n, vals in loaders if n.get("type") == "QuantFuncH3Loader"]
    if not h3_loaders:
        raise ValueError(f"{name}: missing QuantFuncH3Loader")
    for n, vals in h3_loaders:
        if vals.get("sol_min_tokens") != 0 or isinstance(vals.get("sol_min_tokens"), bool):
            raise ValueError(f"{name}: MiniMax-H3 loader sol_min_tokens={vals.get('sol_min_tokens')!r} != 0")
    if "sol_vae" in raw_text.lower():
        raise ValueError(f"{name}: forbidden 'sol_vae' reference in MiniMax-H3 workflow")
    vaes = [n for n in doc["nodes"] if n.get("type") == "VAELoader"]
    vae_files = [
        (n.get("widgets_values") or [None])[0] for n in vaes
    ]
    if H3_VIDEO_VAE_FILE not in vae_files:
        raise ValueError(f"{name}: missing VAELoader for {H3_VIDEO_VAE_FILE!r} (got {vae_files!r})")
    if H3_AUDIO_VAE_FILE not in vae_files:
        raise ValueError(f"{name}: missing VAELoader for {H3_AUDIO_VAE_FILE!r} (got {vae_files!r})")
    for n in vaes:
        wvn = n.get("widgets_values_named")
        if isinstance(wvn, dict) and wvn.get("vae_name") != (n.get("widgets_values") or [None])[0]:
            raise ValueError(f"{name}: VAELoader node {n['id']} widgets_values_named mismatch")

    note = next((n for n in doc["nodes"] if n.get("id") == 1 and n.get("type") == "MarkdownNote"), None)
    if note is None:
        raise ValueError(f"{name}: missing MarkdownNote id=1")
    nwv = note.get("widgets_values")
    if not isinstance(nwv, list) or not nwv or not isinstance(nwv[0], str):
        raise ValueError(f"{name}: MarkdownNote id=1 missing widgets_values[0] text")
    text = nwv[0]
    nwvn = note.get("widgets_values_named")
    if isinstance(nwvn, dict) and nwvn.get("text") != text:
        raise ValueError(f"{name}: MarkdownNote id=1 widgets_values_named['text'] out of sync")
    if H3_VIDEO_VAE_FILE not in text:
        raise ValueError(f"{name}: MarkdownNote id=1 must mention {H3_VIDEO_VAE_FILE!r}")
    if "attention auto." not in text:
        raise ValueError(f"{name}: MarkdownNote id=1 must state 'attention auto.'")
    if re.search(r"\battention\s+(?:sage|int8)\b", text, re.I):
        raise ValueError(f"{name}: MarkdownNote id=1 contains stale attention backend")
    if re.search(r"Saved seed:.*\(fixed\)", text, re.I):
        raise ValueError(f"{name}: MarkdownNote id=1 contains stale '(fixed)' seed description")
    if "Saved seed:" in text and "(randomize)" not in text:
        raise ValueError(f"{name}: MarkdownNote id=1 Saved seed line must document '(randomize)'")

    for node in doc["nodes"]:
        if node.get("type") == "PrimitiveInt":
            pwv = node.get("widgets_values")
            if not isinstance(pwv, list) or len(pwv) < 2 or pwv[1] != "fixed":
                raise ValueError(
                    f"{name}: PrimitiveInt node {node.get('id')} control_after_generate must stay 'fixed' (got {pwv!r})"
                )
            pwvn = node.get("widgets_values_named")
            if isinstance(pwvn, dict) and "fixed" in pwvn and pwvn["fixed"] != "fixed":
                raise ValueError(
                    f"{name}: PrimitiveInt node {node.get('id')} widgets_values_named['fixed'] must stay 'fixed'"
                )


def _validate_all_samplers_res_multistep(name, doc):
    sampler_hits = 0
    for node in doc["nodes"]:
        ntype = node.get("type") or ""
        nid = node.get("id")
        wv = node.get("widgets_values")
        wvn = node.get("widgets_values_named")
        if ntype == "KSampler":
            sampler_hits += 1
            if node.get("mode", 0) != 0:
                raise ValueError(f"{name}: KSampler node {nid} must be active (mode=0, got {node.get('mode')!r})")
            if not isinstance(wv, list) or len(wv) < 5:
                raise ValueError(f"{name}: KSampler node {nid} malformed/truncated widgets_values {wv!r}")
            if wv[4] != "res_multistep":
                raise ValueError(f"{name}: KSampler node {nid} sampler_name={wv[4]!r} != 'res_multistep'")
            if isinstance(wvn, dict) and "sampler_name" in wvn and wvn["sampler_name"] != "res_multistep":
                raise ValueError(
                    f"{name}: KSampler node {nid} widgets_values_named['sampler_name']="
                    f"{wvn['sampler_name']!r} != 'res_multistep'"
                )
        elif ntype == "KSamplerSelect":
            sampler_hits += 1
            if node.get("mode", 0) != 0:
                raise ValueError(f"{name}: KSamplerSelect node {nid} must be active (mode=0, got {node.get('mode')!r})")
            if not isinstance(wv, list) or len(wv) < 1:
                raise ValueError(f"{name}: KSamplerSelect node {nid} malformed/truncated widgets_values {wv!r}")
            if wv[0] != "res_multistep":
                raise ValueError(f"{name}: KSamplerSelect node {nid} sampler_name={wv[0]!r} != 'res_multistep'")
            if isinstance(wvn, dict) and "sampler_name" in wvn and wvn["sampler_name"] != "res_multistep":
                raise ValueError(
                    f"{name}: KSamplerSelect node {nid} widgets_values_named['sampler_name']="
                    f"{wvn['sampler_name']!r} != 'res_multistep'"
                )
        elif ntype == "MarkdownNote":
            note_text = json.dumps(wv) + json.dumps(wvn)
            if "euler" in note_text.lower():
                raise ValueError(f"{name}: MarkdownNote node {nid} contains stale 'euler' reference")
    if sampler_hits < 1:
        raise ValueError(f"{name}: workflow contains zero KSampler/KSamplerSelect nodes")


def _validate_all_seeds_randomize(name, doc):
    seed_hits = 0
    for node in doc["nodes"]:
        ntype = node.get("type") or ""
        nid = node.get("id")
        wv = node.get("widgets_values")
        wvn = node.get("widgets_values_named")
        if ntype in ("KSampler", "RandomNoise"):
            seed_hits += 1
            if not isinstance(wv, list) or len(wv) < 2:
                raise ValueError(f"{name}: {ntype} node {nid} malformed widgets_values {wv!r}")
            if wv[1] != "randomize":
                raise ValueError(
                    f"{name}: {ntype} node {nid} control_after_generate={wv[1]!r} != 'randomize'"
                )
            if isinstance(wvn, dict) and "control_after_generate" in wvn and wvn["control_after_generate"] != "randomize":
                raise ValueError(
                    f"{name}: {ntype} node {nid} widgets_values_named['control_after_generate']="
                    f"{wvn['control_after_generate']!r} != 'randomize'"
                )
    if seed_hits < 1:
        raise ValueError(f"{name}: workflow contains zero KSampler/RandomNoise seed nodes")


def validate_workflow(name, raw_text, loader_schemas):
    """Validate one shipped example workflow JSON against all plugin contracts."""
    if not isinstance(raw_text, str):
        raise ValueError(f"{name}: raw_text must be str")
    doc = json.loads(raw_text, object_pairs_hook=_no_duplicate_keys_hook)
    if not isinstance(doc, dict):
        raise ValueError(f"{name}: top-level JSON must be an object")

    _check_topology(name, doc)
    _check_hygiene(name, raw_text, doc)
    _validate_all_samplers_res_multistep(name, doc)
    _validate_all_seeds_randomize(name, doc)

    loaders = []
    for node in doc["nodes"]:
        ntype = node.get("type") or ""
        if ntype.startswith("QuantFunc") and ntype.endswith("Loader"):
            loaders.append((node, _validate_loader_node(name, node, loader_schemas)))
    if not loaders:
        raise ValueError(f"{name}: workflow contains zero QuantFunc*Loader nodes")

    if name.startswith("QuantFunc-QwenImage21-"):
        _validate_qi21(name, doc, loaders)
    elif name.startswith("QuantFunc-LTX25-"):
        _validate_ltx25(name, loaders)
    elif name.startswith("QuantFunc-MiniMaxH3-"):
        _validate_minimax_h3(name, raw_text, doc, loaders)
    return doc


class ExampleWorkflowsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = load_loader_schemas()
        cls.raw_workflows = {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(WORKFLOW_DIR.glob("*.json"))
        }

    def _widget_index(self, cls_name, wname):
        return [k for k, _ in self.schemas[cls_name]["widgets"]].index(wname)

    def _mutate_doc(self, wf_name, mutator):
        doc = copy.deepcopy(json.loads(self.raw_workflows[wf_name]))
        mutator(doc)
        return json.dumps(doc, indent=2, ensure_ascii=False)

    def test_exact_12_example_workflows_exist_and_schemas_loaded(self):
        self.assertEqual(set(self.raw_workflows), EXPECTED_WORKFLOWS)
        self.assertEqual(
            set(self.schemas),
            {"QuantFuncLTXLoader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader", "QuantFuncH3Loader"},
        )

    def test_all_12_shipped_workflows_pass_contract(self):
        for wf_name, raw_text in sorted(self.raw_workflows.items()):
            with self.subTest(workflow=wf_name):
                doc = validate_workflow(wf_name, raw_text, self.schemas)
                self.assertIsInstance(doc, dict)

    # ── Negative / Mutation Arms ───────────────────────────────────────────────

    def test_mutation_qf_layout_2_rejected(self):
        def m(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncH3Loader")
            loader["properties"]["qf_layout"] = 2

        raw = self._mutate_doc("QuantFunc-MiniMaxH3-fl2va.json", m)
        with self.assertRaisesRegex(ValueError, "qf_layout"):
            validate_workflow("QuantFunc-MiniMaxH3-fl2va.json", raw, self.schemas)

    def test_mutation_stale_attention_backend_rejected(self):
        attn_idx = self._widget_index("QuantFuncH3Loader", "attention_backend")
        for wf, bad in (("QuantFunc-MiniMaxH3-fl2va.json", "sage"),
                        ("QuantFunc-MiniMaxH3-ref2va.json", "int8")):
            with self.subTest(wf=wf, bad=bad):
                def m(doc, b=bad):
                    loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncH3Loader")
                    loader["widgets_values"][attn_idx] = b

                raw = self._mutate_doc(wf, m)
                with self.assertRaisesRegex(ValueError, "attention_backend"):
                    validate_workflow(wf, raw, self.schemas)

    def test_mutation_qi21_loader_dials_rejected(self):
        qi_wf = "QuantFunc-QwenImage21-t2i.json"
        cases = [
            ("sol_enabled", True),
            ("quality_enhance", False),
            ("pinned_memory", False),
            ("sol_min_tokens", 12288),
        ]
        for field, bad_val in cases:
            with self.subTest(field=field):
                idx = self._widget_index("QuantFuncQwenImage21Loader", field)

                def m(doc, i=idx, v=bad_val):
                    loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncQwenImage21Loader")
                    loader["widgets_values"][i] = v

                raw = self._mutate_doc(qi_wf, m)
                with self.assertRaisesRegex(ValueError, field):
                    validate_workflow(qi_wf, raw, self.schemas)

    def test_mutation_qi21_lora_delete_bypass_wrong_file_strength_and_direct_link(self):
        qi_wf = "QuantFunc-QwenImage21-t2i.json"

        def delete_lora_rewire(doc):
            lora = next(n for n in doc["nodes"] if n["type"] == "QuantFuncNativeLoRA")
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncQwenImage21Loader")
            ks = next(n for n in doc["nodes"] if n["type"] == "KSampler")
            doc["nodes"] = [n for n in doc["nodes"] if n["id"] != lora["id"]]
            doc["links"] = [l for l in doc["links"] if l[1] != lora["id"] and l[3] != lora["id"]]
            doc["links"] = [l for l in doc["links"] if l[1] != loader["id"]]
            doc["links"].append([99, loader["id"], 0, ks["id"], 0, "MODEL"])
            doc["last_link_id"] = 99
            loader["outputs"][0]["links"] = [99]
            ks["inputs"][0]["link"] = 99

        with self.assertRaisesRegex(ValueError, "missing QuantFuncNativeLoRA"):
            validate_workflow(qi_wf, self._mutate_doc(qi_wf, delete_lora_rewire), self.schemas)

        def bypass_lora(doc):
            next(n for n in doc["nodes"] if n["type"] == "QuantFuncNativeLoRA")["mode"] = 4

        with self.assertRaisesRegex(ValueError, "active"):
            validate_workflow(qi_wf, self._mutate_doc(qi_wf, bypass_lora), self.schemas)

        def wrong_lora_file(doc):
            next(n for n in doc["nodes"] if n["type"] == "QuantFuncNativeLoRA")["widgets_values"][0] = "wrong.safetensors"

        with self.assertRaisesRegex(ValueError, "LoRA file"):
            validate_workflow(qi_wf, self._mutate_doc(qi_wf, wrong_lora_file), self.schemas)

        def wrong_lora_strength(doc):
            next(n for n in doc["nodes"] if n["type"] == "QuantFuncNativeLoRA")["widgets_values"][1] = 0.5

        with self.assertRaisesRegex(ValueError, "LoRA strength"):
            validate_workflow(qi_wf, self._mutate_doc(qi_wf, wrong_lora_strength), self.schemas)

        def bypass_link_direct(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncQwenImage21Loader")
            ks = next(n for n in doc["nodes"] if n["type"] == "KSampler")
            old_link = ks["inputs"][0]["link"]
            lora = next(n for n in doc["nodes"] if n["type"] == "QuantFuncNativeLoRA")
            lora["outputs"][0]["links"] = []
            doc["links"] = [l for l in doc["links"] if l[0] != old_link]
            doc["links"].append([99, loader["id"], 0, ks["id"], 0, "MODEL"])
            doc["last_link_id"] = 99
            loader["outputs"][0]["links"].append(99)
            ks["inputs"][0]["link"] = 99

        with self.assertRaisesRegex(ValueError, "wire|bypass"):
            validate_workflow(qi_wf, self._mutate_doc(qi_wf, bypass_link_direct), self.schemas)

    def test_mutation_qi21_ksampler_steps_and_sampler_rejected(self):
        qi_wf = "QuantFunc-QwenImage21-edit.json"

        def bad_steps(doc):
            next(n for n in doc["nodes"] if n["type"] == "KSampler")["widgets_values"][2] = 25

        with self.assertRaisesRegex(ValueError, "steps"):
            validate_workflow(qi_wf, self._mutate_doc(qi_wf, bad_steps), self.schemas)

        def bad_sampler(doc):
            next(n for n in doc["nodes"] if n["type"] == "KSampler")["widgets_values"][4] = "euler"

        with self.assertRaisesRegex(ValueError, "sampler_name"):
            validate_workflow(qi_wf, self._mutate_doc(qi_wf, bad_sampler), self.schemas)

    def test_mutation_qi21_transparent_rejects_lightning_lora_and_low_steps(self):
        wf = "QuantFunc-QwenImage21-t2i-transparent.json"

        # (a) Reject QuantFuncNativeLoRA and stock ComfyUI LoraLoader / LoraLoaderModelOnly
        for lora_type in ("QuantFuncNativeLoRA", "LoraLoader", "LoraLoaderModelOnly"):
            with self.subTest(case="forbidden_lora_type", lora_type=lora_type):
                def add_lora(doc, lt=lora_type):
                    loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncQwenImage21Loader")
                    loader["outputs"][0]["links"] = [9]
                    for link in doc["links"]:
                        if link[0] == 2:
                            link[1] = 10
                    doc["links"].append([9, loader["id"], 0, 10, 0, "MODEL"])
                    doc["nodes"].append({
                        "id": 10,
                        "type": lt,
                        "mode": 0,
                        "inputs": [{"name": "model", "type": "MODEL", "link": 9}],
                        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [2]}],
                        "widgets_values": [QI21_LORA_FILE, 1.0],
                    })
                    doc["last_node_id"] = 10
                    doc["last_link_id"] = 9

                with self.assertRaisesRegex(ValueError, "must not attach LoRA"):
                    validate_workflow(wf, self._mutate_doc(wf, add_lora), self.schemas)

        # (b) Exact boundary steps=20 passes; steps=19, 6, float 25.0, bool True all rejected
        def exact_20_steps(doc):
            next(n for n in doc["nodes"] if n["type"] == "KSampler")["widgets_values"][2] = 20

        self.assertIsInstance(
            validate_workflow(wf, self._mutate_doc(wf, exact_20_steps), self.schemas),
            dict,
        )

        for bad_steps in (6, 19, 25.0, True):
            with self.subTest(case="bad_steps", bad_steps=bad_steps):
                def mut_steps(doc, s=bad_steps):
                    next(n for n in doc["nodes"] if n["type"] == "KSampler")["widgets_values"][2] = s

                with self.assertRaisesRegex(ValueError, "steps"):
                    validate_workflow(wf, self._mutate_doc(wf, mut_steps), self.schemas)

        # (c) Truncated KSampler widgets_values len in (5, 6) rejected cleanly with ValueError
        for trunc_len in (5, 6):
            with self.subTest(case="truncated_kwv", trunc_len=trunc_len):
                def trunc_kwv(doc, n=trunc_len):
                    ks = next(x for x in doc["nodes"] if x["type"] == "KSampler")
                    ks["widgets_values"] = ks["widgets_values"][:n]

                with self.assertRaisesRegex(ValueError, "malformed KSampler widgets_values"):
                    validate_workflow(wf, self._mutate_doc(wf, trunc_kwv), self.schemas)

        # (d) Dual-write drift: widgets_values_named['steps'] disagrees or < 20 -> rejected
        for wvn_steps in (6, 19, 24, 25.0, True):
            with self.subTest(case="wvn_steps_drift", wvn_steps=wvn_steps):
                def mut_wvn_steps(doc, s=wvn_steps):
                    ks = next(x for x in doc["nodes"] if x["type"] == "KSampler")
                    ks["widgets_values_named"] = {"steps": s, "sampler_name": "res_multistep"}

                with self.assertRaisesRegex(ValueError, "widgets_values_named"):
                    validate_workflow(wf, self._mutate_doc(wf, mut_wvn_steps), self.schemas)

        def valid_wvn_steps(doc):
            ks = next(x for x in doc["nodes"] if x["type"] == "KSampler")
            ks["widgets_values_named"] = {"steps": 25, "sampler_name": "res_multistep"}

        self.assertIsInstance(
            validate_workflow(wf, self._mutate_doc(wf, valid_wvn_steps), self.schemas),
            dict,
        )

        # (e) Indirect Loader -> intermediate node -> KSampler bypass rejected
        def indirect_model_node(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncQwenImage21Loader")
            loader["outputs"][0]["links"] = [9]
            for link in doc["links"]:
                if link[0] == 2:
                    link[1] = 10
            doc["links"].append([9, loader["id"], 0, 10, 0, "MODEL"])
            doc["nodes"].append({
                "id": 10,
                "type": "ModelSamplingAuraFlow",
                "mode": 0,
                "inputs": [{"name": "model", "type": "MODEL", "link": 9}],
                "outputs": [{"name": "MODEL", "type": "MODEL", "links": [2]}],
                "widgets_values": [3.0],
            })
            doc["last_node_id"] = 10
            doc["last_link_id"] = 9

        with self.assertRaisesRegex(ValueError, "wire Loader -> KSampler directly"):
            validate_workflow(wf, self._mutate_doc(wf, indirect_model_node), self.schemas)

        # (f) Two-sampler graph where sampler 1 is direct but sampler 2 goes through intermediate node -> rejected
        def second_sampler_indirect(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncQwenImage21Loader")
            ks1 = next(n for n in doc["nodes"] if n["type"] == "KSampler")
            loader["outputs"][0]["links"].append(9)
            doc["links"].append([9, loader["id"], 0, 10, 0, "MODEL"])
            doc["links"].append([10, 10, 0, 11, 0, "MODEL"])
            doc["nodes"].append({
                "id": 10,
                "type": "ModelSamplingAuraFlow",
                "mode": 0,
                "inputs": [{"name": "model", "type": "MODEL", "link": 9}],
                "outputs": [{"name": "MODEL", "type": "MODEL", "links": [10]}],
                "widgets_values": [3.0],
            })
            ks2 = copy.deepcopy(ks1)
            ks2["id"] = 11
            ks2["inputs"] = [
                {"name": "model", "type": "MODEL", "link": 10},
                {"name": "positive", "type": "CONDITIONING", "link": None},
                {"name": "negative", "type": "CONDITIONING", "link": None},
                {"name": "latent_image", "type": "LATENT", "link": None},
            ]
            ks2["outputs"] = [{"name": "LATENT", "type": "LATENT", "links": []}]
            doc["nodes"].append(ks2)
            doc["last_node_id"] = 11
            doc["last_link_id"] = 10

        with self.assertRaisesRegex(ValueError, "KSampler id=11.*wire Loader -> KSampler directly"):
            validate_workflow(wf, self._mutate_doc(wf, second_sampler_indirect), self.schemas)

    def test_mutation_all_families_res_multistep_and_sampler_boundaries_rejected(self):
        # (a) KSamplerSelect mutated back to "euler" on LTX25 and MiniMaxH3 (single & double).
        for wf in (
            "QuantFunc-LTX25-t2v.json",
            "QuantFunc-MiniMaxH3-ref2va.json",
            "QuantFunc-MiniMaxH3-ref2va-double-sampling.json",
        ):
            with self.subTest(case="ksampler_select_euler", wf=wf):
                def mut_select_euler(doc):
                    n = next(x for x in doc["nodes"] if x["type"] == "KSamplerSelect")
                    n["widgets_values"][0] = "euler"
                with self.assertRaisesRegex(ValueError, "KSamplerSelect.*sampler_name"):
                    validate_workflow(wf, self._mutate_doc(wf, mut_select_euler), self.schemas)

        # (b) Krea2 KSampler mutated back to "euler" (single + both stage 1 and stage 2 of double).
        for wf, occ in (
            ("QuantFunc-Krea2-t2i.json", 0),
            ("QuantFunc-Krea2-t2i-double-sampling.json", 0),
            ("QuantFunc-Krea2-t2i-double-sampling.json", 1),
        ):
            with self.subTest(case="krea2_ksampler_euler", wf=wf, occ=occ):
                def mut_krea_euler(doc, idx=occ):
                    nodes = [x for x in doc["nodes"] if x["type"] == "KSampler"]
                    nodes[idx]["widgets_values"][4] = "euler"
                with self.assertRaisesRegex(ValueError, "KSampler.*sampler_name"):
                    validate_workflow(wf, self._mutate_doc(wf, mut_krea_euler), self.schemas)

        # (c) Dual-write drift: widgets_values stays "res_multistep" while widgets_values_named drifts to "euler".
        for wf, ntype in (
            ("QuantFunc-Krea2-t2i.json", "KSampler"),
            ("QuantFunc-LTX25-t2v.json", "KSamplerSelect"),
        ):
            with self.subTest(case="wvn_sampler_drift", wf=wf):
                def mut_wvn_drift(doc, t=ntype):
                    n = next(x for x in doc["nodes"] if x["type"] == t)
                    n["widgets_values_named"] = {"sampler_name": "euler"}
                with self.assertRaisesRegex(ValueError, "widgets_values_named"):
                    validate_workflow(wf, self._mutate_doc(wf, mut_wvn_drift), self.schemas)

        # (d) Non-QI-2.1 MarkdownNote referencing stale "euler" (in wv or wvn).
        for wf in (
            "QuantFunc-Krea2-t2i.json",
            "QuantFunc-LTX25-t2v.json",
            "QuantFunc-MiniMaxH3-ref2va.json",
        ):
            with self.subTest(case="note_stale_euler", wf=wf):
                def mut_note_euler(doc):
                    note = next(x for x in doc["nodes"] if x["type"] == "MarkdownNote")
                    note["widgets_values"][0] = note["widgets_values"][0] + "\nUse euler sampler."
                with self.assertRaisesRegex(ValueError, "MarkdownNote.*euler"):
                    validate_workflow(wf, self._mutate_doc(wf, mut_note_euler), self.schemas)

        # (e) Bypassed sampler (mode=4) and truncated widgets_values cleanly raise ValueError (never IndexError).
        for wf, ntype in (
            ("QuantFunc-Krea2-t2i.json", "KSampler"),
            ("QuantFunc-LTX25-t2v.json", "KSamplerSelect"),
        ):
            with self.subTest(case="bypassed_sampler", wf=wf):
                def mut_bypass(doc, t=ntype):
                    next(x for x in doc["nodes"] if x["type"] == t)["mode"] = 4
                with self.assertRaisesRegex(ValueError, "active.*mode=0"):
                    validate_workflow(wf, self._mutate_doc(wf, mut_bypass), self.schemas)

            with self.subTest(case="truncated_sampler_wv", wf=wf):
                def mut_trunc(doc, t=ntype):
                    next(x for x in doc["nodes"] if x["type"] == t)["widgets_values"] = []
                with self.assertRaisesRegex(ValueError, "truncated widgets_values"):
                    validate_workflow(wf, self._mutate_doc(wf, mut_trunc), self.schemas)

    def test_mutation_minimax_h3_sol_vae_and_stale_note_rejected(self):
        h3_wf = "QuantFunc-MiniMaxH3-ref2va.json"

        def restore_sol_vae(doc):
            vae = next(n for n in doc["nodes"] if n["type"] == "VAELoader" and "video" in n["widgets_values"][0])
            vae["widgets_values"][0] = "minimax_h3_sol_vae_int8_convrot.safetensors"
            vae["widgets_values_named"]["vae_name"] = "minimax_h3_sol_vae_int8_convrot.safetensors"

        with self.assertRaisesRegex(ValueError, "sol_vae"):
            validate_workflow(h3_wf, self._mutate_doc(h3_wf, restore_sol_vae), self.schemas)

        for stale in ("attention sage.", "attention int8."):
            with self.subTest(stale=stale):
                def restore_note(doc, s=stale):
                    note = next(n for n in doc["nodes"] if n["id"] == 1)
                    txt = note["widgets_values"][0].replace("attention auto.", s)
                    note["widgets_values"][0] = txt
                    note["widgets_values_named"]["text"] = txt

                with self.assertRaisesRegex(ValueError, "MarkdownNote"):
                    validate_workflow(h3_wf, self._mutate_doc(h3_wf, restore_note), self.schemas)

    def test_mutation_corrupted_block_cache_and_switches_rejected(self):
        def high_bc(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncLTXLoader")
            loader["widgets_values"][self._widget_index("QuantFuncLTXLoader", "block_cache")] = 0.9

        with self.assertRaisesRegex(ValueError, "block_cache"):
            validate_workflow("QuantFunc-LTX25-t2v.json",
                              self._mutate_doc("QuantFunc-LTX25-t2v.json", high_bc), self.schemas)

        def shifted_qe(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncH3Loader")
            loader["widgets_values"][self._widget_index("QuantFuncH3Loader", "quality_enhance")] = 12288

        with self.assertRaisesRegex(ValueError, "quality_enhance.*strict bool"):
            validate_workflow("QuantFunc-MiniMaxH3-ref2va.json",
                              self._mutate_doc("QuantFunc-MiniMaxH3-ref2va.json", shifted_qe), self.schemas)

        def shifted_ae_pm(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncH3Loader")
            loader["widgets_values"][self._widget_index("QuantFuncH3Loader", "audio_enhance")] = "exact_kv_and_rows"
            loader["widgets_values"][self._widget_index("QuantFuncH3Loader", "pinned_memory")] = 2

        with self.assertRaisesRegex(ValueError, "audio_enhance.*strict bool"):
            validate_workflow("QuantFunc-MiniMaxH3-ref2va.json",
                              self._mutate_doc("QuantFunc-MiniMaxH3-ref2va.json", shifted_ae_pm), self.schemas)

        def shifted_pm(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncH3Loader")
            loader["widgets_values"][self._widget_index("QuantFuncH3Loader", "pinned_memory")] = 2

        with self.assertRaisesRegex(ValueError, "pinned_memory.*strict bool"):
            validate_workflow("QuantFunc-MiniMaxH3-ref2va.json",
                              self._mutate_doc("QuantFunc-MiniMaxH3-ref2va.json", shifted_pm), self.schemas)

        def mismatched_named(doc):
            loader = next(n for n in doc["nodes"] if n["type"] == "QuantFuncKrea2Loader")
            schema = self.schemas["QuantFuncKrea2Loader"]
            named = dict(zip((k for k, _ in schema["widgets"]), loader["widgets_values"]))
            named["quality_enhance"] = 12288
            loader["widgets_values_named"] = named

        with self.assertRaisesRegex(ValueError, "widgets_values_named"):
            validate_workflow("QuantFunc-Krea2-t2i.json",
                              self._mutate_doc("QuantFunc-Krea2-t2i.json", mismatched_named), self.schemas)

    def test_mutation_topology_dangling_mismatch_and_orphan_rejected(self):
        wf = "QuantFunc-Krea2-t2i.json"

        def dangling(doc):
            doc["links"][0][3] = 9999

        with self.assertRaisesRegex(ValueError, "dangling"):
            validate_workflow(wf, self._mutate_doc(wf, dangling), self.schemas)

        def slot_mismatch(doc):
            doc["links"][0][5] = "LATENT"

        with self.assertRaisesRegex(ValueError, "type mismatch"):
            validate_workflow(wf, self._mutate_doc(wf, slot_mismatch), self.schemas)

        def orphan(doc):
            new_id = doc["last_node_id"] + 1
            doc["last_node_id"] = new_id
            doc["nodes"].append({
                "id": new_id,
                "type": "EmptyLatentImage",
                "pos": [0, 0],
                "size": [100, 100],
                "flags": {},
                "order": 99,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [512, 512, 1],
            })

        with self.assertRaisesRegex(ValueError, "orphan"):
            validate_workflow(wf, self._mutate_doc(wf, orphan), self.schemas)

        def stale_last_node(doc):
            doc["last_node_id"] = 1

        with self.assertRaisesRegex(ValueError, "last_node_id"):
            validate_workflow(wf, self._mutate_doc(wf, stale_last_node), self.schemas)

        def stale_last_link(doc):
            doc["last_link_id"] = 1

        with self.assertRaisesRegex(ValueError, "last_link_id"):
            validate_workflow(wf, self._mutate_doc(wf, stale_last_link), self.schemas)

    def test_mutation_hygiene_api_key_duplicate_key_and_path_traversal_rejected(self):
        wf = "QuantFunc-Krea2-t2i.json"

        def inject_api_key_field(doc):
            doc["nodes"][1]["properties"]["api_key"] = "placeholder"

        with self.assertRaisesRegex(ValueError, "forbidden 'api_key' field"):
            validate_workflow(wf, self._mutate_doc(wf, inject_api_key_field), self.schemas)

        def inject_literal_api_key(doc):
            doc["nodes"][1]["properties"]["note"] = "qf_" + "0" * 64

        with self.assertRaisesRegex(ValueError, "literal QuantFunc API key"):
            validate_workflow(wf, self._mutate_doc(wf, inject_literal_api_key), self.schemas)

        def inject_api_key_node(doc):
            save_node = next(n for n in doc["nodes"] if n["type"] == "SaveImage")
            save_node["type"] = "QuantFuncApiKey"

        with self.assertRaisesRegex(ValueError, "API-key node"):
            validate_workflow(wf, self._mutate_doc(wf, inject_api_key_node), self.schemas)

        raw_dup = '{"a": 1, "a": 2}'
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            validate_workflow(wf, raw_dup, self.schemas)

        def path_traversal(doc):
            vae = next(n for n in doc["nodes"] if n["type"] == "VAELoader")
            vae["widgets_values"][0] = "../evil.safetensors"

        with self.assertRaisesRegex(ValueError, "traversal"):
            validate_workflow(wf, self._mutate_doc(wf, path_traversal), self.schemas)

        def abs_path(doc):
            clip = next(n for n in doc["nodes"] if n["type"] == "CLIPLoader")
            clip["widgets_values"][0] = "/etc/passwd.safetensors"

        with self.assertRaisesRegex(ValueError, "absolute path"):
            validate_workflow(wf, self._mutate_doc(wf, abs_path), self.schemas)

    def test_mutation_ltx_and_h3_sol_min_tokens_nonzero_rejected(self):
        for wf, cls_name in (
            ("QuantFunc-LTX25-t2v.json", "QuantFuncLTXLoader"),
            ("QuantFunc-MiniMaxH3-fl2va.json", "QuantFuncH3Loader"),
            ("QuantFunc-MiniMaxH3-fl2va-double-sampling.json", "QuantFuncH3Loader"),
            ("QuantFunc-MiniMaxH3-ref2va.json", "QuantFuncH3Loader"),
            ("QuantFunc-MiniMaxH3-ref2va-double-sampling.json", "QuantFuncH3Loader"),
        ):
            idx = self._widget_index(cls_name, "sol_min_tokens")
            with self.subTest(wf=wf):
                def m(doc, c=cls_name, i=idx):
                    loader = next(n for n in doc["nodes"] if n["type"] == c)
                    loader["widgets_values"][i] = 12288

                with self.assertRaisesRegex(ValueError, "sol_min_tokens"):
                    validate_workflow(wf, self._mutate_doc(wf, m), self.schemas)

        # Missing target loader branch coverage (_validate_ltx25 & _validate_minimax_h3)
        with self.assertRaisesRegex(ValueError, "missing QuantFuncLTXLoader"):
            _validate_ltx25("QuantFunc-LTX25-t2v.json", [])
        h3_doc = json.loads(self.raw_workflows["QuantFunc-MiniMaxH3-fl2va.json"])
        with self.assertRaisesRegex(ValueError, "missing QuantFuncH3Loader"):
            _validate_minimax_h3("QuantFunc-MiniMaxH3-fl2va.json", self.raw_workflows["QuantFunc-MiniMaxH3-fl2va.json"], h3_doc, [])

        # H3 MarkdownNote stale '(fixed)' seed description & PrimitiveInt frame-count preservation
        for wf in (
            "QuantFunc-MiniMaxH3-fl2va.json",
            "QuantFunc-MiniMaxH3-ref2va.json",
            "QuantFunc-MiniMaxH3-ref2va-double-sampling.json",
        ):
            with self.subTest(note_wf=wf):
                def m_note(doc):
                    note = next(n for n in doc["nodes"] if n["id"] == 1 and n["type"] == "MarkdownNote")
                    note["widgets_values"][0] = note["widgets_values"][0].replace("(randomize).", "(fixed).")
                    if isinstance(note.get("widgets_values_named"), dict) and "text" in note["widgets_values_named"]:
                        note["widgets_values_named"]["text"] = note["widgets_values"][0]

                with self.assertRaisesRegex(ValueError, "stale '\\(fixed\\)'"):
                    validate_workflow(wf, self._mutate_doc(wf, m_note), self.schemas)

                def m_note_missing(doc):
                    note = next(n for n in doc["nodes"] if n["id"] == 1 and n["type"] == "MarkdownNote")
                    note["widgets_values"][0] = note["widgets_values"][0].replace("(randomize).", "(increment).")
                    if isinstance(note.get("widgets_values_named"), dict) and "text" in note["widgets_values_named"]:
                        note["widgets_values_named"]["text"] = note["widgets_values"][0]

                with self.assertRaisesRegex(ValueError, "must document '\\(randomize\\)'"):
                    validate_workflow(wf, self._mutate_doc(wf, m_note_missing), self.schemas)

        for wf in (
            "QuantFunc-MiniMaxH3-fl2va-double-sampling.json",
            "QuantFunc-MiniMaxH3-ref2va-double-sampling.json",
        ):
            with self.subTest(prim_wf=wf):
                def m_prim(doc):
                    prim = next(n for n in doc["nodes"] if n["type"] == "PrimitiveInt")
                    prim["widgets_values"][1] = "randomize"

                with self.assertRaisesRegex(ValueError, "PrimitiveInt.*must stay 'fixed'"):
                    validate_workflow(wf, self._mutate_doc(wf, m_prim), self.schemas)

                def m_prim_named(doc):
                    prim = next(n for n in doc["nodes"] if n["type"] == "PrimitiveInt")
                    prim["widgets_values_named"] = {"value": 124, "fixed": "randomize"}

                with self.assertRaisesRegex(ValueError, "widgets_values_named\\['fixed'\\] must stay 'fixed'"):
                    validate_workflow(wf, self._mutate_doc(wf, m_prim_named), self.schemas)

    def test_mutation_fixed_seed_rejected(self):
        # Every seed node across all 12 workflows (including Stage 2 KSampler[11] in Krea2 double-sampling)
        for wf in sorted(EXPECTED_WORKFLOWS):
            doc0 = json.loads(self.raw_workflows[wf])
            seed_ids = [n["id"] for n in doc0["nodes"] if n.get("type") in ("KSampler", "RandomNoise")]
            self.assertGreaterEqual(len(seed_ids), 1, f"{wf} missing seed nodes")
            for nid in seed_ids:
                for bad_ctrl in ("fixed", "increment", "decrement"):
                    with self.subTest(wf=wf, nid=nid, bad_ctrl=bad_ctrl):
                        def m(doc, target_id=nid, val=bad_ctrl):
                            node = next(n for n in doc["nodes"] if n["id"] == target_id)
                            node["widgets_values"][1] = val

                        with self.assertRaisesRegex(ValueError, "randomize"):
                            validate_workflow(wf, self._mutate_doc(wf, m), self.schemas)

        # widgets_values_named drift, truncated widgets_values, and zero seed nodes
        wf_krea = "QuantFunc-Krea2-t2i.json"
        def m_wvn_drift(doc):
            ks = next(n for n in doc["nodes"] if n["type"] == "KSampler")
            ks["widgets_values_named"] = {"control_after_generate": "fixed"}

        with self.assertRaisesRegex(ValueError, "widgets_values_named\\['control_after_generate'\\]"):
            validate_workflow(wf_krea, self._mutate_doc(wf_krea, m_wvn_drift), self.schemas)

        def m_trunc(doc):
            rn = next(n for n in doc["nodes"] if n["type"] == "RandomNoise")
            rn["widgets_values"] = [42]

        with self.assertRaisesRegex(ValueError, "malformed widgets_values"):
            validate_workflow("QuantFunc-LTX25-t2v.json", self._mutate_doc("QuantFunc-LTX25-t2v.json", m_trunc), self.schemas)

        with self.assertRaisesRegex(ValueError, "zero KSampler/RandomNoise seed nodes"):
            _validate_all_seeds_randomize("empty.json", {"nodes": []})


if __name__ == "__main__":
    result = unittest.main(argv=[__file__], exit=False, verbosity=2).result
    print(f"EXAMPLE_WORKFLOWS_CONTRACT: {'PASS' if result.wasSuccessful() else 'FAIL'} ({result.testsRun} arms)")
    raise SystemExit(0 if result.wasSuccessful() else 1)
