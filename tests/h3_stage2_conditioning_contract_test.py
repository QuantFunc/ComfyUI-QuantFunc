#!/usr/bin/env python3
"""Stdlib-only check: Stage 2 reuses embeddings/tags without inheriting low-res spatial payloads."""

import ast
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / "qf_h3_conditioning.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "_ReuseH3Encoding")
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    embedding, tags, old_keyframes, old_refs = object(), object(), object(), object()
    metadata = {"minimax_token_tags": tags, "minimax_keyframes": old_keyframes,
                "minimax_refs": old_refs, "pooled_output": "preserved"}
    original = [[embedding, metadata]]
    reuse = namespace["_ReuseH3Encoding"](original)
    result = reuse.encode_from_tokens_scheduled(reuse.tokenize("same prompt"))
    assert result[0][0] is embedding
    assert result[0][1]["minimax_token_tags"] is tags
    assert result[0][1]["pooled_output"] == "preserved"
    assert "minimax_keyframes" not in result[0][1] and "minimax_refs" not in result[0][1]
    assert metadata["minimax_keyframes"] is old_keyframes and metadata["minimax_refs"] is old_refs
    assert result[0][1] is not metadata
    try:
        namespace["_ReuseH3Encoding"]([])
    except ValueError:
        pass
    else:
        raise AssertionError("missing Stage 1 conditioning must fail")
    print("PASS H3 Stage 2 encoding identity, spatial rebuild boundary, and input immutability")


if __name__ == "__main__":
    main()
