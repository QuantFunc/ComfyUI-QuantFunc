#!/usr/bin/env python3
"""Provenance verifier for the LTX-2.5 connectors completion file.

The plugin's transformer-only support (build() connectors-source resolution, commit
43045b4) stages a shared *connector*.safetensors completion file next to a 15G
transformer-only export. That file is a PER-BOX asset (not committed — it is
model weights), extracted VERBATIM from the official ComfyUI dev int8 mirror's
embeddings-connector keys. This script is the committed EVIDENCE for that
provenance claim (验证契约): it recomputes, in full (not sampled),

  (A) the completion file's connector tensors are byte-identical (md5) to the
      official dev int8 mirror's SAME keys — i.e. a verbatim extraction, and
  (B) they are also byte-identical to the shipped distilled ALL-IN export's
      connector tensors — i.e. dev and distilled share one connector set, so the
      completion file legitimately completes the distilled 15G transformer, and
  (C) the key breakdown is exactly N video + N audio connector tensors.

Run it on a box that has the three files; it prints a verdict and exits non-zero
on any mismatch. The committed log next to it (ltx_connectors_provenance.log) is
the recorded run on the origin box. Paths are CLI-overridable for other boxes.

  python3 tests/ltx_connectors_provenance_verify.py \
      [--completion F] [--mirror F] [--allin F]
"""
import argparse
import hashlib
import json
import os
import struct
import sys

_VID = "model.diffusion_model.video_embeddings_connector."
_AUD = "model.diffusion_model.audio_embeddings_connector."

_DEF_COMPLETION = ("/media/jonathan/Data/ComfyUI/models/diffusion_models/"
                   "ltx-2.5-connectors-comfy-int8-convrot.safetensors")
_DEF_MIRROR = ("/media/jonathan/Data/ComfyUI/models/diffusers/"
               "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors")
_DEF_ALLIN = ("/media/jonathan/Data/ComfyUI/models/diffusion_models/"
              "ltx-2.5-22b-distilled-quantfunc-int4-r128-allin-int8.safetensors")


def _read_header(path):
    """(header_dict, data_base_offset). No tensor bytes read."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n


def _tensor_md5(path, hdr, base, key, chunk=1 << 24):
    a, b = hdr[key]["data_offsets"]
    h = hashlib.md5()
    with open(path, "rb") as fh:
        fh.seek(base + a)
        rem = b - a
        while rem > 0:
            buf = fh.read(min(rem, chunk))
            if not buf:
                break
            h.update(buf)
            rem -= len(buf)
    return h.hexdigest()


def _connector_keys(hdr):
    return sorted(k for k in hdr
                  if k != "__metadata__" and ("embeddings_connector" in k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completion", default=_DEF_COMPLETION)
    ap.add_argument("--mirror", default=_DEF_MIRROR)
    ap.add_argument("--allin", default=_DEF_ALLIN)
    args = ap.parse_args()

    missing = [p for p in (args.completion, args.mirror, args.allin)
               if not os.path.exists(p)]
    if missing:
        print("SKIP: missing input file(s) — provenance not verifiable on this box:")
        for p in missing:
            print("  " + p)
        return 77  # ctest SKIP convention

    comp_hdr, comp_base = _read_header(args.completion)
    mir_hdr, mir_base = _read_header(args.mirror)
    all_hdr, all_base = _read_header(args.allin)

    comp_keys = _connector_keys(comp_hdr)
    n_vid = sum(k.startswith(_VID) for k in comp_keys)
    n_aud = sum(k.startswith(_AUD) for k in comp_keys)
    print(f"completion: {os.path.basename(args.completion)}")
    print(f"  connector keys: {len(comp_keys)}  (video {n_vid} + audio {n_aud})")

    ok = True
    # (C) both modalities present and the count is fully accounted for
    if n_vid == 0 or n_aud == 0 or n_vid + n_aud != len(comp_keys):
        print("  FAIL: key breakdown not the expected all-connector video+audio set")
        ok = False

    # (A)+(B) FULL (not sampled) md5 compare of EVERY connector tensor
    mir_missing = [k for k in comp_keys if k not in mir_hdr]
    all_missing = [k for k in comp_keys if k not in all_hdr]
    if mir_missing:
        print(f"  FAIL: {len(mir_missing)} connector keys absent from the dev mirror "
              f"(e.g. {mir_missing[0]})")
        ok = False
    if all_missing:
        print(f"  FAIL: {len(all_missing)} connector keys absent from the allin "
              f"(e.g. {all_missing[0]})")
        ok = False

    same_mir = same_all = 0
    diff_examples = []
    for k in comp_keys:
        cm = _tensor_md5(args.completion, comp_hdr, comp_base, k)
        if k in mir_hdr:
            mm = _tensor_md5(args.mirror, mir_hdr, mir_base, k)
            if cm == mm:
                same_mir += 1
            elif len(diff_examples) < 3:
                diff_examples.append(("mirror", k))
        if k in all_hdr:
            am = _tensor_md5(args.allin, all_hdr, all_base, k)
            if cm == am:
                same_all += 1
            elif len(diff_examples) < 3:
                diff_examples.append(("allin", k))

    print(f"  (A) completion vs dev mirror : {same_mir}/{len(comp_keys)} md5-identical")
    print(f"  (B) completion vs distilled allin: {same_all}/{len(comp_keys)} md5-identical")
    for src, k in diff_examples:
        print(f"      DIFF vs {src}: {k}")
    if same_mir != len(comp_keys) or same_all != len(comp_keys):
        ok = False

    print("VERDICT:", "PASS — verbatim extraction, byte-identical to dev mirror AND "
          "distilled allin connector set" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
