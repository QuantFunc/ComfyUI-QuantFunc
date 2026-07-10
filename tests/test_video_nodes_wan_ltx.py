"""Tests for the Wan/LTX video node generalization (#329/#330 plugin wiring).

Covers, with a MOCKED C-API (no engine .so, no GPU):
  - _pipeline_video_family(): wan / ltx / None from model_index.json _class_name,
    the transformer/config.json fallback, and the _arch hint.
  - QuantFuncGenerateVideo: t2v vs i2v routing (first_frame present ⇒ image_to_video),
    num_frames + fps propagation, audio pass-through/None.
  - QuantFuncGenerate (image node): a video-family pipeline routes to a 1-frame
    text_to_video and returns a single IMAGE; an image pipeline is byte-unchanged
    (still calls text_to_image).
  - QuantFuncVideoPreview: registered with the right INPUT/RETURN types.

Run:  python3 tests/test_video_nodes_wan_ltx.py        (also pytest-compatible)
"""
import os
import sys
import json
import types
import importlib
import tempfile

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN)
_PKG = os.path.basename(_PLUGIN)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Minimal stubs so `nodes` imports without ComfyUI / torch / the engine .so.
for _n in ("comfy", "comfy.model_management", "comfy.utils", "folder_paths"):
    sys.modules.setdefault(_n, types.ModuleType(_n))

import numpy as np  # real

# Prefer REAL torch; install the pass-through stub ONLY when the ACTIVE torch is
# INCAPABLE — a CAPABILITY gate (hasattr from_numpy), NOT an import-success gate.
# Two failure modes this kills:
#  * the old unconditional `sys.modules["torch"] = _torch` leaked a bare stub
#    session-wide (alphabetical order silently SKIPPED / reversed order FAILED
#    every torch-gated test collected later);
#  * an import-success gate alone still bound a BARE foreign stub on torch-less
#    boxes (an earlier-collected file's `sys.modules.setdefault("torch", ...)`
#    satisfies `import torch` from the cache) — measured 9 AttributeError
#    failures in default order. `hasattr(sys.modules.get("torch"), ...)` is
#    deterministic in ANY collection order in BOTH environments.
try:
    import torch  # noqa: F401  (may resolve to a bare foreign stub from the cache)
except Exception:  # noqa: BLE001 — genuinely torch-less box
    pass
if not hasattr(sys.modules.get("torch"), "from_numpy"):
    _torch = types.ModuleType("torch")
    _torch.from_numpy = lambda a: a
    _torch.ones = lambda shape, dtype=None: np.ones(shape, dtype=np.float32)
    _torch.float32 = np.float32
    sys.modules["torch"] = _torch     # replaces an incapable bare stub too

# Bind the ACTIVE torch module (real or the stub above) — tests that temporarily
# swap `_torch.from_numpy` patch whichever module `nodes` actually resolves.
_torch = sys.modules["torch"]

nodes = importlib.import_module(f"{_PKG}.nodes")


# --------------------------- helpers ---------------------------
def _make_model_dir(class_name=None, transformer_class=None):
    d = tempfile.mkdtemp(prefix="qfvid_")
    if class_name is not None:
        with open(os.path.join(d, "model_index.json"), "w") as f:
            json.dump({"_class_name": class_name}, f)
    if transformer_class is not None:
        os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
        with open(os.path.join(d, "transformer", "config.json"), "w") as f:
            json.dump({"_class_name": transformer_class}, f)
    return d


class _FakeManager:
    """Records which generation entrypoint the node called + its args."""
    def __init__(self, n=3, h=64, w=64, audio=None):
        self.calls = []
        self._frames = np.zeros((n, h, w, 3), dtype=np.float32)
        self._audio = audio

    def ensure_pipeline(self, cfg, node_id=None, alive_node_ids=None):
        self.calls.append(("ensure_pipeline", dict(cfg)))
        return "cache-key-1"

    def text_to_video(self, cache_key, prompt, height, width, steps, seed,
                      guidance_scale, num_frames, options_json=None, pbar=None):
        self.calls.append(("text_to_video", {
            "num_frames": num_frames, "options_json": options_json,
            "height": height, "width": width, "guidance_scale": guidance_scale}))
        return self._frames[:num_frames], self._audio

    def image_to_video(self, cache_key, prompt, ref_paths, height, width, steps, seed,
                       guidance_scale, num_frames, negative_prompt="",
                       options_json=None, pbar=None):
        self.calls.append(("image_to_video", {
            "ref_paths": list(ref_paths), "num_frames": num_frames,
            "options_json": options_json, "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt}))
        return self._frames[:num_frames], self._audio

    def text_to_image(self, cache_key, prompt, height, width, steps, seed,
                      guidance_scale, options_json=None, pbar=None, latent_preview=None):
        self.calls.append(("text_to_image", {"height": height, "width": width}))
        return np.zeros((1, height, width, 3), np.float32), None

    # lock used by generate()'s _unload_modes bookkeeping (image path only)
    class _L:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    _lock = _L()
    _unload_modes = {}


def _patch_manager(mgr):
    old = nodes._manager
    nodes._manager = mgr
    return old


# --------------------------- family detection ---------------------------
def test_family_wan_from_model_index():
    for cls in ("WanPipeline", "WanImageToVideoPipeline"):
        d = _make_model_dir(class_name=cls)
        assert nodes._pipeline_video_family({"model_dir": d}) == "wan", cls


def test_family_ltx_from_model_index():
    d = _make_model_dir(class_name="LTX2Pipeline")
    assert nodes._pipeline_video_family({"model_dir": d}) == "ltx"


def test_family_from_transformer_config_fallback():
    d = _make_model_dir(class_name=None, transformer_class="WanTransformer3DModel")
    assert nodes._pipeline_video_family({"model_dir": d}) == "wan"
    d2 = _make_model_dir(class_name=None, transformer_class="LTX2VideoTransformer3DModel")
    assert nodes._pipeline_video_family({"model_dir": d2}) == "ltx"


def test_family_from_arch_hint():
    assert nodes._pipeline_video_family({"model_dir": "", "_arch": "Wan"}) == "wan"
    assert nodes._pipeline_video_family({"model_dir": "", "_arch": "LTX2"}) == "ltx"


def test_family_none_for_image_pipeline():
    d = _make_model_dir(class_name="QwenImagePipeline")
    assert nodes._pipeline_video_family({"model_dir": d}) is None
    assert nodes._pipeline_video_family({"model_dir": "", "_arch": "ZImage"}) is None
    assert nodes._pipeline_video_family({}) is None


# --------------------------- video node dispatch ---------------------------
def test_generate_video_t2v_no_first_frame():
    d = _make_model_dir(class_name="WanPipeline")
    mgr = _FakeManager(n=5)
    old = _patch_manager(mgr)
    try:
        node = nodes.QuantFuncGenerateVideo()
        cap = {}
        old_f2v = nodes._frames_to_video
        nodes._frames_to_video = lambda im, au, f: (cap.update(img=im, aud=au, fps=f)
                                                    or "VIDEO")
        try:
            out = node.generate_video(
                pipeline={"model_dir": d, "options": {}}, prompt="a cat",
                width=64, height=64, length=5, fps=24.0, steps=4,
                guidance_scale=4.0, seed=1)
        finally:
            nodes._frames_to_video = old_f2v
        assert out == ("VIDEO",)                        # SOLE video output
        kinds = [c[0] for c in mgr.calls]
        assert "text_to_video" in kinds and "image_to_video" not in kinds
        call = dict(mgr.calls[[c[0] for c in mgr.calls].index("text_to_video")][1])
        assert call["num_frames"] == 5
        assert json.loads(call["options_json"])["fps"] == 24.0
        # frames + audio are what get packed INTO the VIDEO
        assert cap["aud"] is None                        # Wan → silent VIDEO
        assert cap["img"].shape[0] == 5                  # 5 frames packed in
    finally:
        _patch_manager(old)


def test_generate_video_i2v_with_first_frame():
    d = _make_model_dir(class_name="WanImageToVideoPipeline")
    mgr = _FakeManager(n=5)
    old = _patch_manager(mgr)
    try:
        node = nodes.QuantFuncGenerateVideo()
        first = np.zeros((64, 64, 3), np.float32)   # [H,W,3] first-frame condition
        node.generate_video(
            pipeline={"model_dir": d, "options": {}}, prompt="a cat",
            width=64, height=64, length=5, fps=16.0, steps=4,
            guidance_scale=3.0, seed=1, start_image=first)
        kinds = [c[0] for c in mgr.calls]
        assert "image_to_video" in kinds and "text_to_video" not in kinds
        call = dict(mgr.calls[kinds.index("image_to_video")][1])
        assert call["num_frames"] == 5
        assert len(call["ref_paths"]) == 1            # first-frame condition path
        assert call["guidance_scale"] == 3.0
        assert json.loads(call["options_json"])["fps"] == 16.0
    finally:
        _patch_manager(old)


def _capture_warnings():
    import logging
    recs = []
    h = logging.Handler()
    h.emit = lambda r: recs.append(r.getMessage())
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.WARNING)
    return recs, h


def test_ltx_i2v_warns_not_silent():
    """LTX i2v is not engine-wired → the node must WARN (not silently drop the ref)
    and still forward first_frame (so a future LTX-i2v engine works)."""
    import logging
    d = _make_model_dir(class_name="LTX2Pipeline")
    mgr = _FakeManager(n=3)
    old = _patch_manager(mgr)
    recs, h = _capture_warnings()
    try:
        node = nodes.QuantFuncGenerateVideo()
        first = np.zeros((32, 32, 3), np.float32)
        node.generate_video(pipeline={"model_dir": d, "options": {}}, prompt="x",
                            width=32, height=32, length=3, fps=16.0, steps=4,
                            guidance_scale=3.0, seed=1, start_image=first)
        assert any("LTX i2v" in m for m in recs), recs
        assert "image_to_video" in [c[0] for c in mgr.calls]   # still forwarded
    finally:
        _patch_manager(old)
        logging.getLogger().removeHandler(h)


def test_image_node_video_family_warns_on_ref_images():
    """A video pipeline in the IMAGE node with ref_images wired → warn (not silent)."""
    import logging
    d = _make_model_dir(class_name="WanPipeline")
    mgr = _FakeManager(n=1, h=64, w=64)
    old = _patch_manager(mgr)
    recs, h = _capture_warnings()
    try:
        node = nodes.QuantFuncGenerate()
        node.generate(pipeline={"model_dir": d, "options": {}}, prompt="x",
                      width=64, height=64, steps=4, seed=1, guidance_scale=4.0,
                      ref_images=[np.zeros((1, 64, 64, 3), np.float32)])
        assert any("ref_images are IGNORED" in m for m in recs), recs
        # still produced a single frame via t2v
        assert "text_to_video" in [c[0] for c in mgr.calls]
    finally:
        _patch_manager(old)
        logging.getLogger().removeHandler(h)


def test_generate_video_ltx_audio_passthrough():
    d = _make_model_dir(class_name="LTX2Pipeline")
    audio = {"waveform": np.zeros((2, 100), np.float32), "sample_rate": 16000}
    mgr = _FakeManager(n=3, audio=audio)
    # LTX audio path uses .unsqueeze on the torch tensor — give the stub one.
    old_from = _torch.from_numpy
    _torch.from_numpy = lambda a: types.SimpleNamespace(
        unsqueeze=lambda _dim: a, shape=a.shape)
    old = _patch_manager(mgr)
    try:
        node = nodes.QuantFuncGenerateVideo()
        cap = {}
        old_f2v = nodes._frames_to_video
        nodes._frames_to_video = lambda im, au, f: cap.update(aud=au) or "VIDEO"
        try:
            out = node.generate_video(
                pipeline={"model_dir": d, "options": {}}, prompt="x",
                width=64, height=64, length=3, fps=24.0, steps=4,
                guidance_scale=4.0, seed=1)
        finally:
            nodes._frames_to_video = old_f2v
        assert out == ("VIDEO",)
        # the LTX audio track is what gets muxed into the VIDEO
        assert cap["aud"] is not None and cap["aud"]["sample_rate"] == 16000
    finally:
        _patch_manager(old)
        _torch.from_numpy = old_from


# --------------------------- image node → 1-frame video ---------------------------
def test_image_node_wan_single_frame():
    d = _make_model_dir(class_name="WanPipeline")
    mgr = _FakeManager(n=1, h=64, w=64)
    old = _patch_manager(mgr)
    try:
        node = nodes.QuantFuncGenerate()
        out = node.generate(pipeline={"model_dir": d, "options": {}},
                            prompt="a cat", width=64, height=64, steps=4,
                            seed=1, guidance_scale=4.0)
        kinds = [c[0] for c in mgr.calls]
        assert "text_to_video" in kinds and "text_to_image" not in kinds
        call = dict(mgr.calls[kinds.index("text_to_video")][1])
        assert call["num_frames"] == 1               # single frame = 1-frame t2v
        # image-node 3-tuple (IMAGE, MASK, latent_preview)
        assert len(out) == 3
        assert out[0].shape[0] == 1                   # exactly one frame → one IMAGE
    finally:
        _patch_manager(old)


def test_image_node_image_pipeline_unchanged():
    d = _make_model_dir(class_name="QwenImagePipeline")
    mgr = _FakeManager()
    old = _patch_manager(mgr)
    try:
        node = nodes.QuantFuncGenerate()
        node.generate(pipeline={"model_dir": d, "options": {}},
                      prompt="a cat", width=64, height=64, steps=4,
                      seed=1, guidance_scale=0.0)
        kinds = [c[0] for c in mgr.calls]
        # image pipeline must NOT be routed through the video path
        assert "text_to_image" in kinds
        assert "text_to_video" not in kinds and "image_to_video" not in kinds
    finally:
        _patch_manager(old)


# --------------------------- preview node registration ---------------------------
def test_preview_node_registered():
    assert "QuantFuncVideoPreview" in nodes.NODE_CLASS_MAPPINGS
    assert "QuantFuncVideoPreview" in nodes.NODE_DISPLAY_NAME_MAPPINGS
    cls = nodes.QuantFuncVideoPreview
    it = cls.INPUT_TYPES()
    assert "frames" in it["required"] and "fps" in it["required"]
    assert "audio" in it["optional"]
    assert cls.OUTPUT_NODE is True and cls.RETURN_TYPES == ()


def test_generate_video_input_types_have_i2v_and_fps():
    it = nodes.QuantFuncGenerateVideo.INPUT_TYPES()
    assert "start_image" in it["optional"]           # i2v input (ComfyUI Wan name)
    # fps is OPTIONAL (backward-compat: a pre-#344 saved prompt has no fps key and
    # ComfyUI validate_inputs would hard-fail a missing REQUIRED input).
    assert "fps" in it["optional"] and "fps" not in it["required"]
    assert nodes.QuantFuncGenerateVideo.RETURN_TYPES == ("VIDEO",)
    assert nodes.QuantFuncGenerateVideo.RETURN_NAMES == ("video",)


def test_generate_video_callable_without_fps():
    """Old workflow (no fps submitted) must still run — fps defaults to 24.0."""
    import inspect
    sig = inspect.signature(nodes.QuantFuncGenerateVideo.generate_video)
    assert sig.parameters["fps"].default == 24.0
    d = _make_model_dir(class_name="WanPipeline")
    mgr = _FakeManager(n=3)
    old = _patch_manager(mgr)
    try:
        node = nodes.QuantFuncGenerateVideo()
        # Call WITHOUT fps (as an old prompt would) — must not raise.
        node.generate_video(pipeline={"model_dir": d, "options": {}}, prompt="x",
                            width=64, height=64, length=3, steps=4,
                            guidance_scale=4.0, seed=1)
        call = dict(mgr.calls[[c[0] for c in mgr.calls].index("text_to_video")][1])
        # default fps=24.0 rides in options_json
        assert json.loads(call["options_json"])["fps"] == 24.0
    finally:
        _patch_manager(old)


def test_generate_video_widget_order_backcompat():
    """Graph-format (.json) workflows restore widgets_values by POSITION, so the
    frame-count widget must stay at its ORIGINAL slot-4 (renamed num_frames→length
    IN PLACE), steps/guidance_scale/seed keep their slots, negative_prompt stays the
    first optional widget (slot 8), and new widgets (fps) append AFTER it. Otherwise
    an old saved graph silently misaligns steps/cfg/seed."""
    it = nodes.QuantFuncGenerateVideo.INPUT_TYPES()
    # pipeline is a QUANTFUNC_PIPELINE socket (not a positional widget).
    req_widgets = [k for k in it["required"] if k != "pipeline"]
    assert req_widgets == ["prompt", "width", "height", "length",
                           "steps", "guidance_scale", "seed"], req_widgets
    # start_image is an IMAGE socket (not a widget); it must not shift widget slots.
    opt_widgets = [k for k in it["optional"] if k != "start_image"]
    assert opt_widgets[0] == "negative_prompt", opt_widgets   # slot 8 preserved
    assert "fps" in opt_widgets and \
        opt_widgets.index("fps") > opt_widgets.index("negative_prompt"), opt_widgets  # appended


def test_generate_video_old_widgets_values_positional_backcompat():
    """Simulate ComfyUI restoring an OLD (#344) saved graph's POSITIONAL
    widgets_values against the NEW widget order (ComfyUI applies widgets_values by
    array index for classic dict-INPUT_TYPES nodes). The old num_frames value (slot
    4) must land in `length`, and steps/guidance_scale/seed/negative_prompt must keep
    their slots — i.e. NO silent corruption. fps (a NEW slot-9 widget) is absent from
    the old 8-value array → falls to its default."""
    # An OLD saved graph's positional widgets_values (8 values, no fps), matching the
    # #344 widget order [prompt, width, height, num_frames, steps, guidance_scale,
    # seed, negative_prompt] (pipeline is a socket, not a widget).
    old_values = ["a scenic river", 640, 480, 81, 30, 4.0, 1234, "the negatives"]

    # NEW widget order derived from the LIVE INPUT_TYPES: ComfyUI builds widgets as
    # required-then-optional, skipping non-widget sockets (pipeline / IMAGE / AUDIO).
    it = nodes.QuantFuncGenerateVideo.INPUT_TYPES()
    def _is_socket(spec):
        return spec[0] in ("QUANTFUNC_PIPELINE", "IMAGE", "AUDIO")
    new_widget_order = [k for k, s in it["required"].items() if not _is_socket(s)] \
        + [k for k, s in it["optional"].items() if not _is_socket(s)]

    # ComfyUI applies old_values by index against new_widget_order.
    restored = dict(zip(new_widget_order, old_values))
    assert restored["length"] == 81, restored          # old num_frames → length (slot 4), not corrupted
    assert restored["steps"] == 30                      # NOT shifted
    assert restored["guidance_scale"] == 4.0
    assert restored["seed"] == 1234
    assert restored["negative_prompt"] == "the negatives"
    # fps is appended BEYOND the old 8-value array → never receives an old value.
    assert new_widget_order.index("fps") >= len(old_values), new_widget_order
    # The first 8 new slots line up name-for-name with the old order except slot 3
    # (num_frames→length renamed in place).
    assert new_widget_order[:8] == ["prompt", "width", "height", "length",
                                    "steps", "guidance_scale", "seed", "negative_prompt"], new_widget_order


def test_generate_video_length_is_required():
    it = nodes.QuantFuncGenerateVideo.INPUT_TYPES()
    assert "length" in it["required"] and "length" not in it.get("optional", {})


def test_single_frame_slices_when_engine_returns_multiframe():
    """The image-node single-frame path must yield exactly ONE image even when the
    engine returns >1 frame (a video VAE's minimum temporal decode)."""
    d = _make_model_dir(class_name="WanPipeline")

    class _MultiFrameMgr(_FakeManager):
        def text_to_video(self, *a, **k):   # engine returns 4 frames for num_frames=1
            self.calls.append(("text_to_video", {"num_frames": 1}))
            return np.zeros((4, 64, 64, 3), np.float32), None
    mgr = _MultiFrameMgr()
    old = _patch_manager(mgr)
    try:
        node = nodes.QuantFuncGenerate()
        out = node.generate(pipeline={"model_dir": d, "options": {}}, prompt="x",
                            width=64, height=64, steps=4, seed=1, guidance_scale=4.0)
        assert out[0].shape[0] == 1, out[0].shape   # sliced frames[:1]
        assert out[1].shape[0] == 1                  # mask matches
    finally:
        _patch_manager(old)


def test_i2v_first_frame_unlinked_on_error():
    """The staged first-frame temp file must be unlinked even if the gen raises."""
    d = _make_model_dir(class_name="WanImageToVideoPipeline")
    captured = {}
    real_stage = nodes._write_qfraw_image
    def _spy(img, staging):
        p = real_stage(img, staging); captured["path"] = p; return p
    nodes._write_qfraw_image = _spy

    class _RaisingMgr(_FakeManager):
        def image_to_video(self, *a, **k):
            raise RuntimeError("boom")
    mgr = _RaisingMgr()
    old = _patch_manager(mgr)
    try:
        node = nodes.QuantFuncGenerateVideo()
        first = np.zeros((32, 32, 3), np.float32)
        raised = False
        try:
            node.generate_video(pipeline={"model_dir": d, "options": {}}, prompt="x",
                                width=32, height=32, length=5, fps=16.0, steps=4,
                                guidance_scale=3.0, seed=1, start_image=first)
        except RuntimeError:
            raised = True
        assert raised, "the manager error must propagate"
        assert captured.get("path"), "first frame should have been staged"
        assert not os.path.exists(captured["path"]), "staged first frame must be unlinked"
    finally:
        _patch_manager(old)
        nodes._write_qfraw_image = real_stage


# --------------------------- encoder (real PyAV) ---------------------------
def _has_av():
    try:
        import av  # noqa: F401
        return True
    except Exception:
        return False


def test_encode_video_with_audio_muxed():
    if not _has_av():
        print("    (skipped: PyAV not installed)"); return
    import av
    frames = (np.random.rand(8, 48, 64, 3) * 255).astype(np.uint8)
    sr = 16000
    wav = (np.sin(np.linspace(0, 50, sr))[None, :] * 0.2).repeat(2, axis=0).astype(np.float32)
    out = os.path.join(tempfile.gettempdir(), "qf_test_av.mp4")
    try:
        has_audio = nodes._encode_video_preview(frames, 24.0, {"waveform": wav, "sample_rate": sr},
                                                out, "mp4")
        assert has_audio is True
        c = av.open(out); types_ = sorted(s.type for s in c.streams); c.close()
        assert types_ == ["audio", "video"], types_
    finally:
        try: os.remove(out)
        except OSError: pass


def test_encode_video_only_when_no_audio():
    if not _has_av():
        print("    (skipped: PyAV not installed)"); return
    import av
    frames = (np.random.rand(6, 48, 64, 3) * 255).astype(np.uint8)
    out = os.path.join(tempfile.gettempdir(), "qf_test_vo.mp4")
    try:
        has_audio = nodes._encode_video_preview(frames, 24.0, None, out, "mp4")
        assert has_audio is False
        c = av.open(out); types_ = sorted(s.type for s in c.streams); c.close()
        assert types_ == ["video"], types_
    finally:
        try: os.remove(out)
        except OSError: pass


def test_encode_webm_with_audio():
    if not _has_av():
        print("    (skipped: PyAV not installed)"); return
    import av
    frames = (np.random.rand(6, 48, 64, 3) * 255).astype(np.uint8)
    sr = 16000
    wav = (np.sin(np.linspace(0, 40, sr))[None, :] * 0.2).repeat(2, axis=0).astype(np.float32)
    out = os.path.join(tempfile.gettempdir(), "qf_test_av.webm")
    try:
        has_audio = nodes._encode_video_preview(frames, 24.0, {"waveform": wav, "sample_rate": sr},
                                                out, "webm")
        assert has_audio is True
        c = av.open(out); types_ = sorted(s.type for s in c.streams); c.close()
        assert types_ == ["audio", "video"], types_
    finally:
        try: os.remove(out)
        except OSError: pass


def test_encode_odd_dimensions_and_mono():
    """Odd width/height must be handled (yuv420p needs even); mono audio muxes."""
    if not _has_av():
        print("    (skipped: PyAV not installed)"); return
    import av
    frames = (np.random.rand(5, 49, 63, 3) * 255).astype(np.uint8)   # odd H=49, W=63
    sr = 16000
    wav = (np.sin(np.linspace(0, 40, sr))[None, :] * 0.2).astype(np.float32)  # mono [1,N]
    out = os.path.join(tempfile.gettempdir(), "qf_test_odd.mp4")
    try:
        has_audio = nodes._encode_video_preview(frames, 24.0, {"waveform": wav, "sample_rate": sr},
                                                out, "mp4")
        assert has_audio is True
        c = av.open(out)
        vs = [s for s in c.streams if s.type == "video"][0]
        w, h = vs.codec_context.width, vs.codec_context.height
        c.close()
        assert w % 2 == 0 and h % 2 == 0, (w, h)   # cropped to even
    finally:
        try: os.remove(out)
        except OSError: pass


def test_encode_multichannel_downmix():
    """>2 channel audio must be downmixed to stereo, not crash."""
    if not _has_av():
        print("    (skipped: PyAV not installed)"); return
    import av
    frames = (np.random.rand(5, 48, 64, 3) * 255).astype(np.uint8)
    sr = 16000
    wav = (np.random.rand(6, sr) * 0.1).astype(np.float32)   # 5.1 → downmix to stereo
    out = os.path.join(tempfile.gettempdir(), "qf_test_51.mp4")
    try:
        has_audio = nodes._encode_video_preview(frames, 24.0, {"waveform": wav, "sample_rate": sr},
                                                out, "mp4")
        assert has_audio is True
        c = av.open(out); types_ = sorted(s.type for s in c.streams); c.close()
        assert "audio" in types_
    finally:
        try: os.remove(out)
        except OSError: pass


def test_encode_pyav_failure_falls_back_to_cv2_clean():
    """Force the PyAV path to raise mid-encode → the container is closed + the
    partially-written file cleaned, and the cv2 fallback produces a VALID video-only
    clip (verifies the resource-leak/corruption fix, not just the happy path)."""
    if not _has_av():
        print("    (skipped: PyAV not installed)"); return
    try:
        import cv2  # noqa: F401
    except Exception:
        print("    (skipped: cv2 not installed)"); return
    import av
    orig = av.VideoFrame.from_ndarray
    def _boom(*a, **k):
        raise RuntimeError("forced mid-encode failure")   # container already open
    out = os.path.join(tempfile.gettempdir(), "qf_pyavfail.mp4")
    try:
        av.VideoFrame.from_ndarray = _boom
        frames = (np.random.rand(5, 32, 32, 3) * 255).astype(np.uint8)
        has_audio = nodes._encode_video_preview(frames, 24.0, None, out, "mp4")
    finally:
        av.VideoFrame.from_ndarray = orig
    assert has_audio is False                          # cv2 fallback = video-only
    assert os.path.exists(out) and os.path.getsize(out) > 0
    # the produced file is a VALID (cv2-written) video, not a corrupt PyAV partial
    c = av.open(out); types_ = [s.type for s in c.streams]; c.close()
    assert "video" in types_, types_
    try:
        os.remove(out)
    except OSError:
        pass


def test_preview_returns_native_comfy_video_payload():
    """QuantFuncVideoPreview must emit ComfyUI's NATIVE video-preview payload
    ({"ui": {"images": [SavedResult], "animated": (True,)}}, == ui.PreviewVideo) so
    the core <video> player renders it — NOT a bespoke custom-widget payload."""
    if not _has_av():
        print("    (skipped: PyAV not installed)"); return
    import folder_paths
    tmp = tempfile.mkdtemp(prefix="qfvp_")
    folder_paths.get_temp_directory = lambda: tmp

    class _T:   # minimal IMAGE tensor stub (.detach().cpu().numpy())
        def __init__(self, a): self._a = a
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self._a
    frames = _T(np.random.rand(4, 32, 32, 3).astype(np.float32))
    node = nodes.QuantFuncVideoPreview()
    out = node.preview(frames, 24.0, audio=None, container="mp4")
    assert "ui" in out and "images" in out["ui"], out
    assert out["ui"].get("animated") == (True,), out            # native video flag
    img = out["ui"]["images"][0]
    assert img["type"] == "temp" and img["filename"].endswith(".mp4")
    assert set(("filename", "subfolder", "type")).issubset(img)  # SavedResult shape
    assert os.path.exists(os.path.join(tmp, img["filename"]))
    try:
        os.remove(os.path.join(tmp, img["filename"]))
    except OSError:
        pass


# --------------------------- runner ---------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)


def test_frames_to_video_builds_real_video_when_api_present():
    # When the ComfyUI native VIDEO API is importable, _frames_to_video builds a real
    # VIDEO (not the None fallback) carrying the frames + audio + fps. Skips in a bare
    # test env without comfy_api (the fallback path is covered by the tests above).
    try:
        import comfy_api.input_impl  # noqa: F401
        from comfy_api.latest import VideoComponents  # noqa: F401
    except Exception:
        import pytest as _pt; _pt.skip("comfy_api (native VIDEO type) not available")
    import torch as _t
    img = _t.zeros(3, 8, 8, 3)
    aud = {"waveform": _t.zeros(1, 2, 100), "sample_rate": 16000}
    v = nodes._frames_to_video(img, aud, 24.0)
    assert v is not None
    comp = v.get_components()
    assert comp.images.shape[0] == 3                      # 3 frames packed in
    assert int(comp.frame_rate) == 24
    assert comp.audio is not None and comp.audio["sample_rate"] == 16000


def test_generate_video_sampler_rides_options_json():
    # a picked sampler is forwarded into options_json (the engine reads it there);
    # an unknown/default one is handled cleanly. Output is the sole VIDEO.
    d = _make_model_dir(class_name="WanPipeline")
    mgr = _FakeManager(n=5)
    old = _patch_manager(mgr)
    old_f2v = nodes._frames_to_video
    nodes._frames_to_video = lambda im, au, f: "VIDEO"
    try:
        node = nodes.QuantFuncGenerateVideo()
        out = node.generate_video(
            pipeline={"model_dir": d, "options": {}}, prompt="a cat",
            width=64, height=64, length=5, fps=24.0, steps=4,
            guidance_scale=1.0, seed=1, sampler_name="dpmpp_2m",
            scheduler="karras", true_cfg_scale=4.0, negative_prompt="ugly",
            sampler_eta=0.5)
        assert out == ("VIDEO",)
        call = dict(mgr.calls[[c[0] for c in mgr.calls].index("text_to_video")][1])
        opts = json.loads(call["options_json"])
        assert opts["sampler"] == "dpmpp_2m"           # rides in options_json
        assert opts["scheduler"] == "karras"           # non-default scheduler emitted
        assert opts["eta"] == 0.5                       # eta emitted (>0)
        assert opts["true_cfg_scale"] == 4.0            # classical CFG (with negative)
        assert opts["negative_prompt"] == "ugly"
        # defaults must NOT leak keys (byte-identical default path)
        mgr.calls.clear()
        out2 = node.generate_video(
            pipeline={"model_dir": d, "options": {}}, prompt="a cat",
            width=64, height=64, length=5, fps=24.0, steps=4,
            guidance_scale=1.0, seed=1)
        call2 = dict(mgr.calls[[c[0] for c in mgr.calls].index("text_to_video")][1])
        o2 = json.loads(call2["options_json"])
        assert o2["sampler"] == "euler" and "scheduler" not in o2 and "eta" not in o2
        assert "true_cfg_scale" not in o2
    finally:
        nodes._frames_to_video = old_f2v
        _patch_manager(old)
