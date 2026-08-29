#!/usr/bin/env python3
"""qf_lora_convert.py — convert a LoRA into the ONE format the QuantFunc native
loader adapts to: diffusers / PEFT canonical (``<module>.lora_A.weight`` /
``<module>.lora_B.weight`` [+ ``<module>.alpha``]).

Why one format: the QuantFunc engine binds a diffusers-form key to ANY
1:1-named transformer module with zero per-family tables (its generic
module-probe). So a converted LoRA loads directly into the Krea-2 / LTX-2 /
MiniMax-H3 native loaders — one tool, every family, no engine change.

What it accepts (auto-detected):
  * diffusers / PEFT  — already canonical; normalized (down/up -> A/B) + copied
  * kohya sd-scripts / ai-toolkit — ``lora_unet_``/``lora_transformer_`` keys,
    underscored module path, ``.lora_down/up.weight`` + ``.alpha``
  * LyCORIS (LoHa/LoKr) — REFUSED loud (needs math reconstruction, not a rename)

Raw-safetensors byte-copy: tensor DTYPE is preserved exactly (bf16/fp16/fp32),
no torch, no numpy, no model load, no GPU. Text-encoder LoRA keys (``lora_te*``)
are dropped — the native loaders drive the transformer only.
"""
import json, struct, sys, argparse, os

# ---- multi-token diffusers MODULE names (union across Krea-2 / LTX-2 / H3) ----
# ONLY names whose INTERNAL underscore must survive kohya's dot->underscore
# flattening need listing. Single-token names (blocks, attn, ff, norm, proj,
# gate, up, down, …) naturally become correct dotted segments and need nothing.
_PROTECTED = sorted({
    # block groups
    "single_transformer_blocks", "transformer_blocks", "layerwise_blocks",
    "refiner_blocks", "token_refiner", "text_fusion",
    # attention modules (incl. the LTX-2 AV towers — from REAL LoRA corpora)
    "self_attn", "cross_attn", "audio_attn1", "audio_attn2",
    "audio_to_video_attn", "video_to_audio_attn", "audio_ff",
    "to_gate_logits", "to_gate", "to_out", "to_q", "to_k", "to_v",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "add_q", "add_k", "add_v",
    # ffn / embeds / singles
    "feed_forward", "time_mod_proj", "time_embed", "final_layer",
    "audio_patch_proj", "audio_proj_in", "audio_time_embed",
    "norm_out", "proj_out", "proj_in", "img_mlp", "txt_mlp", "img_mod",
    "txt_mod", "img_in", "txt_in", "linear_1", "linear_2",
    "norm_q", "norm_k", "norm1", "norm2",
    # LTX-2 model-level compounds (REAL 22b LoRA corpus)
    "av_ca_a2v_gate_adaln_single", "av_ca_v2a_gate_adaln_single",
    "av_ca_audio_scale_shift_adaln_single", "av_ca_video_scale_shift_adaln_single",
    "audio_prompt_adaln_single", "audio_adaln_single", "prompt_adaln_single",
    "adaln_single", "timestep_embedder", "audio_patchify_proj", "patchify_proj",
    "audio_proj_out", "caption_projection",
    # NOTE: "ff_net" was WRONG here (the diffusers name is ff.net — ff and net
    # are separate segments that split naturally; protecting the pair kept the
    # underscore and produced a nonexistent module). Removed after the REAL
    # LTX-2.5 corpus round-trip caught it.
}, key=len, reverse=True)  # longest-first so to_gate_logits beats to_gate

_ROOT_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet.", "unet.",
                  "base_model.model.", "model.")


_MAX_HEADER_BYTES = 512 * 1024 * 1024  # same plausibility cap as the plugin's node sniff


def _read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > _MAX_HEADER_BYTES:
            raise SystemExit(
                "qf_lora_convert: implausible safetensors header size %d in '%s' — "
                "corrupt or not a safetensors file" % (n, path))
        hdr = json.loads(f.read(n))
        blob = f.read()
    return hdr, blob


def _write_st(path, ordered_tensors, blob_src, metadata):
    """ordered_tensors: list of (name, info) where info has dtype/shape/data_offsets
    into blob_src. Rebuilds a contiguous blob (drops make holes) + header."""
    new_hdr = {}
    if metadata:
        new_hdr["__metadata__"] = metadata
    out = bytearray()
    off = 0
    for name, info in ordered_tensors:
        b0, b1 = info["data_offsets"]
        data = blob_src[b0:b1]
        new_hdr[name] = {"dtype": info["dtype"], "shape": info["shape"],
                         "data_offsets": [off, off + len(data)]}
        out += data
        off += len(data)
    hj = json.dumps(new_hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - (len(hj) % 8)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(bytes(out))


def kohya_join(underscored):
    """kohya underscored module path -> dotted diffusers path. Protect the known
    multi-token module names (their internal '_' stays), turn the rest into '.'."""
    s = underscored
    holds = {}
    for i, w in enumerate(_PROTECTED):
        tok = "\x00%d\x00" % i
        # underscore/dot/end-bounded whole-word match (kohya separators are '_')
        j = 0
        while True:
            k = s.find(w, j)
            if k < 0:
                break
            before_ok = (k == 0) or (s[k - 1] in "_.\x00")
            e = k + len(w)
            after_ok = (e == len(s)) or (s[e] in "_.\x00")
            if before_ok and after_ok:
                s = s[:k] + tok + s[e:]
                j = k + len(tok)
            else:
                j = k + 1
        holds[tok] = w
    s = s.replace("_", ".")
    for tok, w in holds.items():
        s = s.replace(tok, w)
    return s


def build_model_inverse(model_path):
    """Derive the EXACT underscore->dotted inverse map from a target checkpoint:
    every tensor key minus its leaf names a real module path; flatten each with
    '_' and map back. This makes kohya reconstruction unambiguous for ANY family
    with zero vocabulary — the checkpoint IS the dictionary. Colliding flats are
    refused (never guessed)."""
    # header-only read (no blob copy) — a checkpoint can be many GB
    with open(model_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > _MAX_HEADER_BYTES:
            raise SystemExit(
                "qf_lora_convert: implausible safetensors header size %d in '%s'"
                % (n, model_path))
        keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
    mods = set()
    for k in keys:
        if "." not in k:
            continue
        mods.add(k.rsplit(".", 1)[0])          # module = key minus tensor leaf
    # also add parent chains (to_out registered as to_out.0 -> its parent to_out)
    inv = {}
    clash = set()
    for m in mods:
        flat = m.replace(".", "_")
        if flat in inv and inv[flat] != m:
            clash.add(flat)
        inv[flat] = m
    for c in clash:
        del inv[c]                              # ambiguous — fall to vocab join
    return inv


def strip_root(path):
    changed = True
    while changed:
        changed = False
        for p in _ROOT_PREFIXES:
            if path.startswith(p):
                path = path[len(p):]
                changed = True
    return path


def detect_format(keys):
    ky = list(keys)
    if any(".hada_" in k or ".lokr_" in k for k in ky):
        return "lycoris"
    if any(k.startswith(("lora_unet_", "lora_transformer_", "lora_te_",
                         "lora_te1_", "lora_te2_")) for k in ky):
        return "kohya"
    if any(".lora_A." in k or ".lora_B." in k or ".lora_down." in k
           or ".lora_up." in k or ".lora_linear_layer." in k for k in ky):
        return "diffusers"
    return "unknown"


def _canon_leaf(role):
    # normalize the low-rank role to PEFT canonical A/B; carry alpha
    if role in ("lora_down", "lora_A", "lora_linear_layer.down"):
        return "lora_A"
    if role in ("lora_up", "lora_B", "lora_linear_layer.up"):
        return "lora_B"
    if role == "alpha":
        return "alpha"
    return None


def _split_role(key):
    """Return (module_body, role) or (None, None). role in
    {lora_down,lora_up,lora_A,lora_B,alpha,lora_linear_layer.down/up}."""
    for suf, role in ((".lora_down.weight", "lora_down"),
                      (".lora_up.weight", "lora_up"),
                      (".lora_A.weight", "lora_A"),
                      (".lora_B.weight", "lora_B"),
                      (".lora_linear_layer.down.weight", "lora_linear_layer.down"),
                      (".lora_linear_layer.up.weight", "lora_linear_layer.up"),
                      (".alpha", "alpha")):
        if key.endswith(suf):
            return key[:-len(suf)], role
    return None, None


def convert_key(orig, src_fmt, model_inv=None):
    """orig external key -> canonical diffusers key, or None to DROP.
    model_inv: exact flat->dotted map derived from the target checkpoint
    (--model); consulted BEFORE the vocabulary join — unambiguous for any
    family. Vocab join is the model-less fallback."""
    if orig == "__metadata__":
        return None
    if src_fmt == "kohya":
        if orig.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
            return None  # text-encoder LoRA — transformer-only loader
        body, role = _split_role(orig)
        if body is None:
            return None
        for pre in ("lora_unet_", "lora_transformer_"):
            if body.startswith(pre):
                body = body[len(pre):]
                break
        dotted = (model_inv or {}).get(body) or kohya_join(body)
    else:  # diffusers / peft
        if orig.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
            return None
        body, role = _split_role(orig)
        if body is None:
            return None
        dotted = body  # already dotted
    leaf = _canon_leaf(role)
    if leaf is None:
        return None
    if leaf == "alpha":
        return dotted + ".alpha"
    return dotted + "." + leaf + ".weight"


def convert_file(in_path, out_path, verbose=True, model_path=None):
    model_inv = build_model_inverse(model_path) if model_path else None
    hdr, blob = _read_st(in_path)
    meta = hdr.get("__metadata__", {})
    keys = [k for k in hdr if k != "__metadata__"]
    fmt = detect_format(keys)
    if fmt == "lycoris":
        raise SystemExit(
            "qf_lora_convert: this is a LyCORIS (LoHa/LoKr) file — it is a "
            "factored decomposition, not a plain (A,B) LoRA, so a key-rename "
            "cannot convert it. Re-export/merge it to a standard LoRA first, "
            "or load it through the ctypes engine path which reconstructs it.")
    if fmt == "unknown":
        raise SystemExit(
            "qf_lora_convert: could not detect the LoRA format (no kohya "
            "lora_unet_* / diffusers .lora_A/.lora_down keys found).")
    if fmt == "kohya" and model_inv is None:
        print("[qf_lora_convert] WARNING: converting kohya keys WITHOUT --model — the "
              "underscore->dot reconstruction falls back to a built-in vocabulary that is "
              "corpus-fitted and may miss a family's novel module names. STRONGLY "
              "recommended: pass --model <target-checkpoint.safetensors> for an exact, "
              "vocabulary-free inverse.", flush=True)
    out_tensors = []
    dropped = 0
    seen = {}
    for k in keys:
        nk = convert_key(k, fmt, model_inv)
        if nk is None:
            dropped += 1
            continue
        if nk in seen:
            raise SystemExit(
                "qf_lora_convert: two source keys map to the same target "
                "'%s' (%s and %s) — ambiguous, refusing." % (nk, seen[nk], k))
        seen[nk] = k
        out_tensors.append((nk, hdr[k]))
    if not out_tensors:
        raise SystemExit("qf_lora_convert: nothing to write (all keys dropped).")
    new_meta = {"qf_lora_convert": "from %s (%s)" % (os.path.basename(in_path), fmt)}
    _write_st(out_path, out_tensors, blob, new_meta)
    if verbose:
        print("[qf_lora_convert] %s: source=%s  wrote %d tensors, dropped %d  -> %s"
              % (os.path.basename(in_path), fmt, len(out_tensors), dropped, out_path))
    return fmt, len(out_tensors), dropped


def self_test(diffusers_path):
    """Synthesize a kohya twin of a REAL diffusers LoRA in-memory, convert it
    back, and assert the (module-path, role) set + tensor bytes round-trip
    losslessly (root prefix is cosmetic — the engine strips it, so compare
    stripped)."""
    hdr, blob = _read_st(diffusers_path)
    orig = {k: v for k, v in hdr.items() if k != "__metadata__"}
    # build a kohya twin header (A->lora_down, B->lora_up, dots->underscores)
    twin = {}
    for k, info in orig.items():
        body, role = _split_role(k)
        if body is None:
            continue
        body = strip_root(body)
        und = body.replace(".", "_")
        suf = {"lora_A": ".lora_down.weight", "lora_B": ".lora_up.weight",
               "lora_down": ".lora_down.weight", "lora_up": ".lora_up.weight"}.get(role)
        if suf is None:
            continue
        twin["lora_unet_" + und + suf] = info
    # convert the twin back
    refmt = detect_format(twin.keys())
    assert refmt == "kohya", "twin should read as kohya, got %s" % refmt
    recon = {}
    for k in twin:
        nk = convert_key(k, "kohya")
        assert nk is not None, "twin key dropped: %s" % k
        recon[nk] = twin[k]
    # compare stripped-module identity + byte-identity of the tensor data
    def norm(key):
        b, r = _split_role(key)
        return (strip_root(b), _canon_leaf(r))
    o_set = {norm(k) for k in orig}
    r_set = {norm(k) for k in recon}
    missing = o_set - r_set
    extra = r_set - o_set
    ok = True
    if missing or extra:
        ok = False
        print("  FAIL module-set: missing=%d extra=%d" % (len(missing), len(extra)))
        for m in list(missing)[:5]:
            print("    missing", m)
        for e in list(extra)[:5]:
            print("    extra  ", e)
    # byte identity per module/role
    o_by = {norm(k): hdr[k]["data_offsets"] for k in orig}
    r_by = {norm(k): recon[k]["data_offsets"] for k in recon}
    bytes_ok = 0
    bytes_bad = 0
    for key in o_set & r_set:
        (ob0, ob1) = o_by[key]
        (rb0, rb1) = r_by[key]
        if blob[ob0:ob1] == blob[rb0:rb1]:
            bytes_ok += 1
        else:
            bytes_bad += 1
    if bytes_bad:
        ok = False
    print("  module-set: %d orig / %d recon  match=%s | tensor-bytes: %d identical, %d differ"
          % (len(o_set), len(r_set), (not missing and not extra), bytes_ok, bytes_bad))
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Convert a LoRA to QuantFunc diffusers/PEFT canonical form.")
    ap.add_argument("--in", dest="inp", help="input LoRA .safetensors")
    ap.add_argument("--out", dest="out", help="output .safetensors (diffusers canonical)")
    ap.add_argument("--model", dest="model", help="target checkpoint .safetensors — derives the EXACT module-name inverse (any family, zero vocabulary)")
    ap.add_argument("--self-test", dest="selftest", help="round-trip a diffusers LoRA through a synthesized kohya twin")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if self_test(a.selftest) else 1)
    if not a.inp or not a.out:
        ap.error("need --in and --out (or --self-test)")
    convert_file(a.inp, a.out, model_path=a.model)


if __name__ == "__main__":
    main()
