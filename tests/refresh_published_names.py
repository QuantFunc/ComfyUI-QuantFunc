#!/usr/bin/env python3
"""Refresh tests/published_names.json from ModelScope: every .safetensors QuantFunc publishes, per repo.

Run after EVERY ModelScope publish and before a plugin release (#713 — a published name nothing tested was refused
by our own loader). Known files keep their preset/arm assignment and repos keep their classification; a NEW repo is
written as "UNCLASSIFIED" and a new file in a native repo gets "preset": null. tests/published_names_test.py fails on
both until a person assigns them: the assignment says what the file IS, so it is never derived from the name matcher
the test checks.

  python3 tests/refresh_published_names.py           # rewrite the frozen list
  python3 tests/refresh_published_names.py --check   # exit 1 if ModelScope differs from the frozen list (no write)
"""
import datetime
import json
import os
import sys
import urllib.request

API = "https://www.modelscope.cn"
OWNER = "QuantFunc"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "published_names.json")


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _repos():
    out, page = [], 1
    while True:
        d = _get(f"{API}/openapi/v1/models?author={OWNER}&page_size=50&page_number={page}")["data"]
        out += [m["id"].split("/", 1)[1] for m in d["models"]]
        if not d["models"] or len(out) >= d["total_count"]:
            return sorted(out)
        page += 1


def _safetensors(repo):
    files = _get(f"{API}/api/v1/models/{OWNER}/{repo}/repo/files?Recursive=true")["Data"]["Files"]
    return sorted(f["Path"] for f in files if f.get("Type") == "blob" and f["Path"].endswith(".safetensors"))


def main():
    old = json.load(open(OUT)) if os.path.exists(OUT) else {"repos": {}}
    new = {"_about": old.get("_about", ""), "fetched": datetime.date.today().isoformat(), "repos": {}}
    for repo in _repos():
        prev = old["repos"].get(repo, {"loader": "UNCLASSIFIED", "files": {}})
        new["repos"][repo] = {"loader": prev["loader"],
                              "files": {p: prev["files"].get(p, {"preset": None, "arm": None}) for p in _safetensors(repo)}}
    changed = [f"{r}: {sorted(set(new['repos'].get(r, {}).get('files', {})) ^ set(old['repos'].get(r, {}).get('files', {})))}"
               for r in sorted(set(new["repos"]) | set(old["repos"]))
               if set(new["repos"].get(r, {}).get("files", {})) != set(old["repos"].get(r, {}).get("files", {}))]
    print("\n".join(changed) or "no change in published files")
    if "--check" in sys.argv:
        return 1 if changed else 0
    with open(OUT, "w") as f:
        json.dump(new, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
