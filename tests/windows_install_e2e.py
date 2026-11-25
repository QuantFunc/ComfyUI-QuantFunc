#!/usr/bin/env python3
"""Windows end-to-end check of the engine installer (tests-07 dispatch 2026-09-24). NOT part of the suite (the suite
runs *_test.py): it needs a Windows GPU box, the DLL builder's staged files and a ComfyUI with the QI-2.1 models.

Nothing is uploaded anywhere (user ruling: no Windows files on ModelScope until the user says so). The staged release is
served over LOCAL HTTPS on 127.0.0.1 with a throwaway CA made for this run (its private key never touches the disk when
the `cryptography` package is there; with the openssl CLI it is deleted right after signing):
  stage   <out>/stage/version.json (the live version.json + a win32 "<version>" entry with "kernel_so": true),
          <out>/stage/<version>/verify.json ("win32": the SHA-256 of every staged file, cross-checked against the DLL
          builder's own verify.json when given), <out>/stage/<version>/windows/{sets.json, <set>/<file>}
  A.      the plugin's OWN install_engine(). The ONLY deviation from production: _ENGINE_BASE_URL -> the local stage. The
          install process trusts the throwaway CA through SSL_CERT_FILE — the plugin fetches with urllib's default context,
          whose set_default_verify_paths() honours it; measured first, both ways (trusted with it, refused without).
  B.      a NORMAL ComfyUI start on this plugin copy (no overrides: its startup installer talks to the real repo, which
          publishes no Windows per-arch release yet, and leaves the staged install alone), the QI-2.1 t2i template's API
          prompt (the frontend's own graphToPrompt output), the engine DLL the ComfyUI process loaded, the image.

Run with that ComfyUI's python, from anywhere; the plugin is the folder this file sits in (a FRESH copy: `git archive`
of the plugin commit into ComfyUI\\custom_nodes\\ComfyUI-QuantFunc, no .dev_lib_lock, QF_NATIVE_SO_PATH unset):
  python tests\\windows_install_e2e.py --staged <dir with windows\\sets.json and windows\\<set>\\...> --comfy <ComfyUI>
         [--fixer-verify <the builder's verify.json>] [--xfm <QI-2.1 model file name>] [--out <dir>]
Pre-flight without a GPU (any OS): --dry-run stubs torch / the GPU / the Windows platform and runs the stage, the TLS
measurement and step A only."""
import argparse
import datetime
import functools
import hashlib
import http.server
import importlib.util
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = "https://www.modelscope.cn/models/QuantFunc/Plugin/resolve/master/version.json"
# Minimal standalone Windows installer QI-2.1 smoke prompt (without optional Lightning LoRA dependency).
PROMPT = {
    "2": {"inputs": {"transformer": "qwen-image-2.1-quantfunc-int4-r128-i8sidecar.qfc.safetensors",
                     "model_config": "qwen-image-2.1-int4", "attention_backend": "auto", "quality_enhance": False},
          "class_type": "QuantFuncQwenImage21Loader"},
    "3": {"inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image", "device": "default"},
          "class_type": "CLIPLoader"},
    "4": {"inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}, "class_type": "VAELoader"},
    "5": {"inputs": {"prompt": "A photorealistic fluffy orange cat sitting on a wooden windowsill, soft morning light, "
                               "detailed fur", "negative_prompt": "", "resolution": 1024, "clip": ["3", 0]},
          "class_type": "TextEncodeQwenImage21"},
    "6": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptyLatentImage"},
    "7": {"inputs": {"seed": 0, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
                     "model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["6", 0]},
          "class_type": "KSampler"},
    "8": {"inputs": {"samples": ["7", 0], "vae": ["4", 0]}, "class_type": "VAEDecode"},
    "9": {"inputs": {"filename_prefix": "QuantFunc_QwenImage21_t2i", "images": ["8", 0]}, "class_type": "SaveImage"},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_stage(staged, stage, version, plugin_req, fixer_verify):
    """The QuantFunc/Plugin layout of ONE Windows release, from the builder's staged windows\\ folder."""
    src = os.path.join(staged, "windows")
    if not os.path.isfile(os.path.join(src, "sets.json")):
        sys.exit(f"--staged must hold windows\\sets.json (got {src})")
    body = {}
    for root, _, files in os.walk(src):
        for n in files:
            rel = os.path.relpath(os.path.join(root, n), src).replace(os.sep, "/")
            dst = os.path.join(stage, version, "windows", *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(root, n), dst)
            body[rel] = sha256(dst)
    # PC Engine O1 Windows split layout:
    #   - Host DLLs (quantfunc.dll / quantfunc-12.dll) MUST have zero CUDA PE imports.
    #   - Per-SM kernel plugins (sm*/libquantfunc_kernels.dll / sm*/libquantfunc_kernels-12.dll)
    #     MUST carry their expected CUDA major in their PE imports.
    spec = importlib.util.spec_from_file_location("qf_engine_stage", os.path.join(PLUGIN, "qf_engine.py"))
    qfe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qfe)
    hosts_by_name = {dll: major for major, dll in qfe._ENGINE_PLATFORMS["windows"]["hosts"].items()}
    valid_majors = set(qfe._ENGINE_PLATFORMS["windows"]["hosts"])
    wrong_hosts, wrong_kernels, found_hosts, found_kernels = [], [], set(), set()
    for rel in sorted(body):
        parts = rel.split("/")
        name = parts[-1]
        fpath = os.path.join(stage, version, "windows", *parts)
        if name in hosts_by_name:
            leaked = qfe._host_forbidden_cuda_pe(fpath)
            print(f"   host {rel}: forbidden CUDA PE imports = {leaked} (expected [])")
            if leaked:
                wrong_hosts.append(f"{rel} ({', '.join(leaked)})")
            found_hosts.add(hosts_by_name[name])
        elif qfe._ENGINE_KERNEL_WIN_RE.fullmatch(name):
            want = 12 if "-12." in name.lower() else 13
            got = qfe._so_cuda_major(fpath)
            print(f"   kernel {rel}: CUDA {got} from its imports (expected {want or sorted(valid_majors)})")
            if (want is not None and got != want) or (want is None and got not in valid_majors):
                wrong_kernels.append(rel)
            if got in valid_majors:
                found_kernels.add(got)
    if wrong_hosts:
        sys.exit(f"staged host DLLs unexpectedly import CUDA (O1 zero-CUDA host required): {wrong_hosts}")
    if wrong_kernels:
        sys.exit(f"staged kernel DLLs whose imports do not carry their expected CUDA major: {wrong_kernels}")
    matched_majors = sorted(found_hosts & found_kernels)
    if not matched_majors:
        sys.exit(f"staged release incomplete: hosts={sorted(found_hosts)} kernels={sorted(found_kernels)}")
    if fixer_verify:
        with open(fixer_verify, encoding="utf-8") as fv:
            theirs = json.load(fv).get("win32", {})
        bad = sorted(k for k in body if theirs.get(k) != body[k])
        if bad:
            sys.exit(f"the staged files do not match the builder's verify.json: {bad}")
        print(f"   staged files match the builder's verify.json ({len(body)} of {len(theirs)} keys staged)")
    with open(os.path.join(stage, version, "verify.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "win32": body}, f)
    try:
        with urllib.request.urlopen(LIVE, timeout=30) as r:
            doc = json.loads(r.read())
        origin = "the live version.json"
    except Exception as e:  # noqa: BLE001 — offline: a minimal document still exercises the pick
        doc, origin = {"linux": {}, "win32": {}}, f"a minimal version.json (live one unreachable: {type(e).__name__})"
    win32 = {k: v for k, v in (doc.get("win32") or {}).items()
             if not (isinstance(v, dict) and v.get("kernel_so"))}
    win32[version] = {"comfy": plugin_req, "comfy-12": plugin_req, "lib": version,
                      "lib-12": version, "kernel_so": True}
    doc["win32"] = win32
    with open(os.path.join(stage, "version.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)
    with open(os.path.join(stage, version, "windows", "sets.json"), encoding="utf-8") as sf:
        sets_doc = json.load(sf).get("sets", {})
    sms = sorted({s for v in sets_doc.values() for s in v if s in qfe._CONSUMER_SMS})
    if not sms:
        sys.exit("staged sets.json has no consumer SM architecture")
    dry_major = 13 if 13 in matched_majors else matched_majors[-1]
    dry_sm = 89 if 89 in sms else sms[0]
    print(f"   stage: {origin} + win32 {version}; files: {sorted(body)}")
    return dry_major, dry_sm


def make_tls(d):
    """A throwaway CA and a 127.0.0.1 server certificate signed by it: (ca.pem, server.pem, server.key)."""
    ca_pem, crt, key = (os.path.join(d, n) for n in ("ca.pem", "server.pem", "server.key"))
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
        import ipaddress
        now = datetime.datetime.now(datetime.timezone.utc)
        span = dict(not_valid_before=now - datetime.timedelta(minutes=5), not_valid_after=now + datetime.timedelta(hours=3))
        ca_key, srv_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "qf-e2e throwaway CA")])
        ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
        ca = (x509.CertificateBuilder(issuer_name=ca_name, subject_name=ca_name, public_key=ca_key.public_key(),
                                      serial_number=x509.random_serial_number(), **span)
              .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
              .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                           content_commitment=False, key_encipherment=False, data_encipherment=False,
                                           key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
              .add_extension(ca_ski, critical=False)
              .sign(ca_key, hashes.SHA256()))
        srv = (x509.CertificateBuilder(issuer_name=ca_name, public_key=srv_key.public_key(),
                                       subject_name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]),
                                       serial_number=x509.random_serial_number(), **span)
               .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                              critical=False)
               .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
               .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
               .add_extension(x509.SubjectKeyIdentifier.from_public_key(srv_key.public_key()), critical=False)
               .add_extension(x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ca_ski), critical=False)
               .sign(ca_key, hashes.SHA256()))
        open(ca_pem, "wb").write(ca.public_bytes(serialization.Encoding.PEM))
        open(crt, "wb").write(srv.public_bytes(serialization.Encoding.PEM))
        open(key, "wb").write(srv_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                    serialization.NoEncryption()))
        return ca_pem, crt, key, "cryptography"
    except ImportError:
        pass
    exe = shutil.which("openssl") or next((p for p in (r"C:\Program Files\Git\usr\bin\openssl.exe",
                                                        r"C:\Program Files\Git\mingw64\bin\openssl.exe")
                                           if os.path.isfile(p)), None)
    if not exe:
        sys.exit("need the `cryptography` package or an openssl executable to make the throwaway certificate")
    ca_key, csr, ext = (os.path.join(d, n) for n in ("ca.key", "server.csr", "server.ext"))
    with open(ext, "w", encoding="utf-8") as f:
        f.write("subjectAltName=IP:127.0.0.1\nbasicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\n"
                "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n")
    run = functools.partial(subprocess.run, check=True, capture_output=True)
    run([exe, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", ca_key,
         "-out", ca_pem, "-days", "1", "-subj", "/CN=qf-e2e throwaway CA",
         "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0", "-addext", "keyUsage=critical,keyCertSign,cRLSign"])
    run([exe, "req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", key, "-out", csr,
         "-subj", "/CN=127.0.0.1"])
    try:
        run([exe, "x509", "-req", "-in", csr, "-CA", ca_pem, "-CAkey", ca_key, "-CAcreateserial", "-out", crt,
             "-days", "1", "-extfile", ext])
    finally:
        os.remove(ca_key)                                # the CA can sign nothing after this run
    return ca_pem, crt, key, "openssl"


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(stage, crt, key):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=stage))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"https://127.0.0.1:{srv.server_address[1]}"


FETCH = "import sys, urllib.request\ntry:\n    urllib.request.urlopen(sys.argv[1], timeout=20).read(); print('trusted')\n" \
        "except Exception as e:\n    print('refused:', type(e).__name__, str(e)[:120])\n"

STEP_A = r'''
import hashlib, importlib.util, json, os, shutil, sys, time, types
plugin, base, dry = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
dry_major, dry_sm, dry_bin = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
spec = importlib.util.spec_from_file_location("qf_engine_e2e", os.path.join(plugin, "qf_engine.py"))
qfe = importlib.util.module_from_spec(spec); spec.loader.exec_module(qfe)
if dry:   # no GPU / not Windows: stub what only the box can answer; the installer, HTTPS and FORK-2 stay real
    os.makedirs(dry_bin, exist_ok=True)
    shutil.copyfile(os.path.join(plugin, "bin", "windows", "version.json"), os.path.join(dry_bin, "version.json"))
    qfe._engine_bin_dir = lambda: dry_bin
    qfe._BIN_SUBDIR, qfe._LIB_BASENAME = "windows", "quantfunc.dll"
    qfe.platform.machine = lambda: "AMD64"
    qfe._torch_cuda_major, qfe._driver_cuda_major, qfe._gpu_sm = (
        (lambda: dry_major), (lambda: dry_major), (lambda idx=0: dry_sm))
    if "msvcrt" not in sys.modules:
        ms = types.ModuleType("msvcrt"); ms.LK_UNLCK, ms.LK_LOCK, ms.LK_NBLCK = 0, 1, 2
        ms.locking = lambda fd, mode, n: print(f"   lock call: {'lock' if mode else 'unlock'} {n} byte")
        sys.modules["msvcrt"] = ms
print("platform row:", qfe._BIN_SUBDIR, qfe._engine_platform()["key"], "| driver CUDA", qfe._driver_cuda_major())
major, sm = qfe.engine_choice(0)
print(f"engine_choice(0): torch CUDA {major}, GPU SM {sm}")
qfe._ENGINE_BASE_URL = base                      # the ONLY deviation from production: the local stage
t0 = time.time(); m = qfe.install_engine(0)
print(f"install_engine: {time.time() - t0:.1f} s, status {qfe.engine_install_status()}")
print("marker:", json.dumps(m, sort_keys=True))
man = json.loads(qfe._engine_http_get(f"{base}/{m['version']}/verify.json"))["win32"]
pair_dir = os.path.join(qfe._engine_bin_dir(), f"{m['version']}-{m['set']}-cu{m['cuda']}")
host_dll = os.path.join(pair_dir, m["host"])
kern_dll = os.path.join(pair_dir, m["kernel"])
for label, path, fname in (("host", host_dll, m["host"]), ("kernel", kern_dll, m["kernel"])):
    with open(path, "rb") as fh:
        b = fh.read()
    key = qfe._manifest_key(m["set"], fname, m["cuda"], man)
    ok = hashlib.sha256(b).hexdigest() == man.get(key)
    print(f"{label} {key}: {len(b)} B md5 {hashlib.md5(b).hexdigest()} sha256==verify.json {ok}")
    if not ok:
        sys.exit(f"{label} sha256 mismatch for {key}")
print("host PE imports:", qfe._pe_imports(host_dll), "forbidden CUDA:", qfe._host_forbidden_cuda_pe(host_dll))
print("kernel PE imports:", qfe._pe_imports(kern_dll), "CUDA major:", qfe._so_cuda_major(kern_dll))
qfe.assert_toolchain_compatible(host_dll)
print("FORK-2 guard: passed")
t0 = time.time(); print(f"re-run: same marker {qfe.install_engine(0) == m} in {time.time() - t0:.1f} s")
print("resolve_so_path:", qfe.resolve_so_path())
'''


def step_b(py, comfy, out, port, xfm, env):
    log = open(os.path.join(out, "comfy.log"), "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([py, "main.py", "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch",
                             "--output-directory", out], cwd=comfy, stdout=log, stderr=subprocess.STDOUT, env=env,
                            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    base, op = f"http://127.0.0.1:{port}", urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for _ in range(300):
            try:
                op.open(base + "/system_stats", timeout=5)
                break
            except OSError:
                if proc.poll() is not None:
                    sys.exit("ComfyUI exited; see " + log.name)
                time.sleep(2)
        prompt = json.loads(json.dumps(PROMPT))
        prompt["2"]["inputs"]["transformer"] = xfm
        req = urllib.request.Request(base + "/prompt", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode())
        try:
            pid = json.load(op.open(req, timeout=60))["prompt_id"]
        except urllib.error.HTTPError as e:
            sys.exit(f"PROMPT REFUSED {e.code}: {e.read().decode()[:1500]}")
        t0 = time.time()
        while time.time() - t0 < 1500:
            h = json.load(op.open(f"{base}/history/{pid}", timeout=30))
            if pid in h:
                st = h[pid].get("status", {})
                print(f"run: {st.get('status_str')} after {time.time() - t0:.0f} s")
                for kind, d in st.get("messages", []):
                    if kind in ("execution_error", "execution_interrupted"):
                        print("EXEC ERROR", d.get("node_type"), ":", str(d.get("exception_message"))[:600])
                for o in h[pid].get("outputs", {}).values():
                    for im in o.get("images", []):
                        f = os.path.join(out, im.get("subfolder", ""), im["filename"])
                        print("IMAGE", f, "md5", hashlib.md5(open(f, "rb").read()).hexdigest())
                break
            time.sleep(3)
        else:
            print("TIMEOUT")
        for ln in open(log.name, encoding="utf-8", errors="replace"):
            if "[qf_native]" in ln or "[QuantFunc]" in ln:
                print("   log:", ln.rstrip()[:220])
        # tasklist writes the console code page; only its ASCII module names matter here, so never fail on the rest
        mods = subprocess.run(["tasklist", "/m", "quantfunc*", "/fi", f"PID eq {proc.pid}", "/fo", "list"],
                              capture_output=True, encoding="utf-8", errors="replace").stdout
        print(f"engine modules loaded by ComfyUI (pid {proc.pid}):", " ".join(mods.split())[:500])
    finally:
        proc.terminate()
        log.close()


def main():
    sys.stdout.reconfigure(line_buffering=True)   # our lines interleave with the install process's in order
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True)
    ap.add_argument("--comfy")
    ap.add_argument("--version", default="0.0.14")
    ap.add_argument("--plugin-req", default="0.0.08")
    ap.add_argument("--fixer-verify")
    ap.add_argument("--xfm", default=PROMPT["2"]["inputs"]["transformer"])
    ap.add_argument("--port", type=int, default=18913)
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not a.comfy:
        sys.exit("--comfy is required (or --dry-run)")
    out = os.path.abspath(a.out or tempfile.mkdtemp(prefix="qf-e2e-"))
    stage, tls = os.path.join(out, "stage"), tempfile.mkdtemp(prefix="qf-e2e-tls-")
    env = {k: v for k, v in os.environ.items() if k not in ("QF_NATIVE_SO_PATH", "QF_NATIVE_KEYFILE", "SSL_CERT_FILE")}
    for k in ("NO_PROXY", "no_proxy"):     # the local stage and ComfyUI are loopback: never through a box proxy
        env[k] = ",".join(x for x in (env.get(k, ""), "127.0.0.1,localhost") if x)
    print(f"plugin {PLUGIN} | out {out} | bin\\windows holds {sorted(os.listdir(os.path.join(PLUGIN, 'bin', 'windows')))}")
    try:
        print("== stage (the QuantFunc/Plugin layout, local only)")
        dry_major, dry_sm = build_stage(a.staged, stage, a.version, a.plugin_req, a.fixer_verify)
        ca, crt, key, how = make_tls(tls)
        srv, base = serve(stage, crt, key)
        print(f"   serving {base} (throwaway CA via {how}); the CA is trusted only by the install process below")
        trusted = dict(env, SSL_CERT_FILE=ca)       # urllib's default context honours it (measured just below)
        for label, e in (("with SSL_CERT_FILE", trusted), ("without (control)", env)):
            r = subprocess.run([sys.executable, "-c", FETCH, base + "/version.json"], env=dict(e, PYTHONIOENCODING="utf-8"),
                               capture_output=True, encoding="utf-8", errors="replace")
            print(f"   urllib default context {label}: {r.stdout.strip()}")
        print("== A: the plugin's installer against the local stage")
        dry_bin = os.path.join(out, "dry-bin")
        r = subprocess.run([sys.executable, "-c", STEP_A, PLUGIN, base, "1" if a.dry_run else "0",
                            str(dry_major), str(dry_sm), dry_bin], env=trusted)
        srv.shutdown()
        if r.returncode:
            sys.exit(f"STEP A FAILED (rc {r.returncode})")
        if a.dry_run:
            print("== B skipped (--dry-run)")
            return
        print("== B: a normal ComfyUI start + the QI-2.1 template")
        step_b(sys.executable, a.comfy, out, a.port, a.xfm, env)
    finally:
        shutil.rmtree(tls, ignore_errors=True)


if __name__ == "__main__":
    main()
