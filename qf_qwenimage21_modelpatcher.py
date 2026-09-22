"""qf_qwenimage21_modelpatcher — the Qwen-Image-2.1 (image t2i) family seam (model + builder), self-contained.

The Krea2 shape (the first image family on the native seam): comfy owns CLIP (TextEncodeQwenImage21 →
Qwen3-VL-8B last-hidden [B, S, 4096]) / VAE (qwen_image_2.1_vae) / sampler / CFG; the engine owns ONLY
the 32-block denoise via the generic external session (QwenImage21Pipeline::prepareExternalDenoise /
quantfunc_denoise_begin + quantfunc_denoise_step). comfy's own QwenImage21Transformer2DModel receives
(x [B,64,H/16,W/16] normalized latent, timestep = sigma, context) per cond group — exactly what this
seam forwards, so cond and latent pass through unchanged (cast to bf16, the engine's activation dtype).
Reference-image EDIT conditioning (reference_latents / image_slots) is NOT on the seam (no field in the
generic step ABI) and is refused loud, never silently dropped.
"""
import ctypes
import json
import os

import torch

import comfy.model_base
import comfy.model_management
import comfy.supported_models

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import (_qf_dtype, _QFStub, _CtxKeyAssigner, QFModelPatcher,
                              QFSessionModelMixin, _interrupt_poll_end_session_on_raise)


FAMILY = "qwenimage21"


def matches(pipeline_class, transformer_class=""):
    """The engine's own detector (src/QwenImage21Pipeline.cpp qwenimage21_detect)."""
    return (str(pipeline_class) == "QwenImage21Pipeline"
            or str(transformer_class) == "QwenImage21Transformer2DModel")


class QFQwenImage21Model(QFSessionModelMixin, comfy.model_base.QwenImage21):
    _VAE_S = 16   # latent_formats.QwenImage21 spacial_downscale_ratio; session W/H = latent * 16

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)
        self.diffusion_model = _QFStub()
        self.diffusion_model.arm_concat_shape(64)   # in == latent channels → extra 0 → no concat build
        self._qf = engine
        self._num_steps = 0
        self._step_i = 0
        self._ctx_key_assigner = _CtxKeyAssigner()
        self._sess_denoise = 0
        self._max_batch = 0
        self._max_ctx_seq = 0
        self._out = None

    # comfy model_base.QwenImage21.current_patcher's setter drives the TORCH model's prefix K/V cache
    # (diffusion_model.reset_prefix_cache) — the engine owns its own prefix cache inside the session,
    # so the stub must not be asked for one.
    @property
    def current_patcher(self):
        return getattr(self, "_current_patcher", None)

    @current_patcher.setter
    def current_patcher(self, patcher):
        self._current_patcher = patcher

    def get_dynamic_vram__units(self):
        return [], []   # no torch blocks to page — the engine holds the weights

    # comfy model_base.QwenImage(21).extra_conds consumables this seam cannot honor — refuse LOUD,
    # never silently drop (the wan reject-list discipline). reference_latents/image_slots = the QI2.1
    # edit channel (engine generate_edit only); attention_mask = the engine forward is mask-blind;
    # concat_latent_image/concat_mask/denoise_mask = the BaseModel inpaint/concat channels (no i2i or
    # inpaint on the t2i seam). Audited by tests/reject_list_completeness.py (QwenImage21 row).
    _ENGINE_IGNORED_COND_KEYS = ("reference_latents", "reference_latents_method", "image_slots",
                                 "attention_mask", "cross_attn_controlnet", "noise_concat",
                                 "concat_latent_image", "concat_mask", "denoise_mask")

    def extra_conds(self, **kwargs):
        self._assert_wire_lora()
        was_open, ok = self._qf.end_session_if_open()   # run-start clean slate (#2 lifecycle)
        self._qf_needs_begin = True
        if was_open:
            print("[qf_native] qwenimage21: closed a pre-existing session at run start "
                  f"(prior run interrupted/uncleaned); end ok={ok}", flush=True)
        for _k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(_k) is not None:
                raise RuntimeError(
                    f"qf_native qwenimage21: '{_k}' conditioning is wired, but the native session "
                    f"cannot consume it — it would be silently ignored, so it is refused. "
                    f"Reference-image edit (images on TextEncodeQwenImage21) is not part of the "
                    f"qwenimage21 t2i seam — remove the reference images.")
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            self._max_ctx_seq = max(self._max_ctx_seq, int(cross_attn.shape[1]))
        return out

    def _begin(self, x_group, ctx_group):
        qfmp._qf_cancel_pending_detach(self._qf)
        lib = self._qf.lib                       # MATERIALIZE FIRST (deferred-wrapper no-op close)
        self._qf.end_session_if_open()
        bpx = qfe.DenoiseBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        bpx.width = int(x_group.shape[-1]) * self._VAE_S
        bpx.height = int(x_group.shape[-2]) * self._VAE_S
        bpx.num_steps = self._num_steps
        _max_seq = max(self._max_ctx_seq, int(ctx_group.shape[1]))
        bpx.max_context_dims = (ctypes.c_int * 3)(
            int(ctx_group.shape[0]), _max_seq, int(ctx_group.shape[2]))
        self._max_ctx_seq = 0
        bpx.cond_dtype = _qf_dtype(ctx_group.dtype)
        # IMAGE session: only the runtime attn-backend dial (applyAttnBackendDial); the video
        # residency_opts keys are engine-REFUSED here.
        _o = {}
        ab = str(getattr(self, "_attn_backend", "auto") or "auto")
        if ab != "auto":
            _o["attention_backend"] = ab
        bpx._opts = json.dumps(_o).encode()
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        st = lib.quantfunc_denoise_begin(self._qf.pipeline, ctypes.byref(bpx), ctypes.byref(session))
        self._begin_keep = bpx
        if st != qfe.QUANTFUNC_OK:
            raise RuntimeError(f"denoise_begin (qwenimage21 t2i) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        self._qf.unloaded = False
        self._step_i = 0
        self._sess_denoise = 0
        self._max_batch = 0
        self._ctx_key_assigner.reset()
        print(f"[qf_native] QWENIMAGE21 SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
              f"latent={tuple(x_group.shape)} cond={tuple(ctx_group.shape)}", flush=True)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        ctx = c_crossattn
        if ctx is None:
            raise RuntimeError("qf_native qwenimage21: no c_crossattn cond — wire TextEncodeQwenImage21")
        if control is not None:
            raise RuntimeError(
                "qf_native qwenimage21: a ControlNet is wired, but the native session consumes no "
                "control input — it would be silently ignored, so it is refused.")
        if c_concat is not None:
            raise RuntimeError(
                "qf_native qwenimage21: c_concat conditioning is not part of the t2i seam — remove "
                "the node feeding it.")
        # BF16-latent/BF16-cond family (the engine's activation dtype); comfy may hand FP32 and may
        # keep the cond host-side — cast to the LATENT's device + bf16 in one .to() (a no-op when
        # already there). The step ABI takes DEVICE pointers.
        _dev = x.device
        x_bf = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        if ctx.dtype != torch.bfloat16 or ctx.device != _dev:
            ctx = ctx.to(device=_dev, dtype=torch.bfloat16)
        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        sched = transformer_options.get("sample_sigmas", None)
        if sched is not None:
            self._num_steps = max(1, int(sched.numel()) - 1)
        elif self._num_steps <= 0:
            self._num_steps = 1
        xin = x_bf
        B = int(xin.shape[0])
        if getattr(self, "_qf_needs_begin", True) or self._qf.current_session is None:
            self._begin(xin[0:1], ctx[0:1])
            self._qf_needs_begin = False
        if self._out is None or self._out.shape != xin.shape or self._out.dtype != xin.dtype \
                or self._out.device != xin.device:
            self._out = torch.empty_like(xin)
        self._max_batch = max(self._max_batch, B)
        # comfy carries the per-conditioning uuids as transformer_options["uuids"] (samplers.py:324/511 —
        # the key the wan/ltx/h3 seams read); a wrong key silently yields ctx key 0 = caches OFF (measured:
        # the engine logged "cfg_context_key=0 — prefix K/V cache OFF" on the first QI2.1 native run).
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
        for i in range(B):
            _interrupt_poll_end_session_on_raise(self._qf)
            xi = xin[i:i + 1].contiguous()          # QI2.1 latent is IMAGE 4D [1,64,h,w] (no T axis)
            oi = self._out[i:i + 1]
            ci = ctx[i:i + 1].contiguous()
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            ctx_key = self._ctx_key_assigner.key(cuid)
            sig_i = float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= B) else \
                (float(sig_all[0].item()) if sig_all is not None else float(sigma))
            p = qfe.DenoiseStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = xi.data_ptr()
            p.velocity_out = oi.data_ptr()
            p.velocity_out_capacity = oi.numel() * oi.element_size()
            dims = list(xi.shape) + [0] * (5 - xi.dim())
            p.dims = (ctypes.c_int * 5)(*dims)
            p.dtype = _qf_dtype(xi.dtype)
            p.sigma = sig_i
            p.step_index = step_index
            p.total_steps = self._num_steps
            p.context = ci.data_ptr()
            p.context_dims = (ctypes.c_int * 3)(*ci.shape)
            p.context_dtype = _qf_dtype(ci.dtype)
            p.cfg_context_key = ctx_key
            self._call_denoise_step(p, f"qwenimage21 denoise_step[step={step_index},group={i},key={ctx_key}]")
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        return self.model_sampling.calculate_denoised(sigma, self._out.float(), x)

    def process_latent_out(self, latent):
        # NORMAL end-of-sampling: close the session (no finalize — the image seam has no
        # masked-blend); an INTERRUPT skips this and extra_conds closes at the next run start.
        if self._qf.current_session is not None:
            self._qf.end_session_if_open()
            print(f"[qf_native] QWENIMAGE21 SESSION CLOSED after {self._step_i} sampler steps, "
                  f"{self._sess_denoise} denoise_step calls", flush=True)
        return super().process_latent_out(latent)


def register(deps):
    get_engine = deps["get_engine"]
    bind_pipeline_model = deps["bind_pipeline_model"]
    retire_handle = deps["retire_handle"]
    estimate_footprint = deps["estimate_footprint"]

    def build(transformer1_path, transformer2_path, bundle_dir=None,
              lora_entries=(), sparse_opts=None):
        """File-based Qwen-Image-2.1 t2i — the Krea2 single-expert staging pattern: stage the shipped
        config bundle (configs/qwen-image-2.1-*/, model_index.json only — the engine synthesizes the
        per-component configs from the reference arch) + symlink the transformer file; engine create
        runs denoise_only=True (TE + VAE weights skipped — comfy's TextEncodeQwenImage21 owns
        conditioning, comfy's VAEDecode decodes)."""
        if transformer2_path:
            raise RuntimeError("qf_native qwenimage21: single-expert family — transformer2 must be empty")
        model_dir = qfmp.stage_denoise_only_package(bundle_dir, transformer1_path, None)
        create_extra = {"denoise_only": True}
        if sparse_opts:
            create_extra.update(sparse_opts)
        model_name = os.path.basename(transformer1_path)

        def _build(lora_entries):
            _lora_cfg = dict(create_extra or {})
            if lora_entries:
                _lora_cfg["lora"] = list(lora_entries)   # engine svdq load: sidecar apply post-load
            _factory, _register_model = qfmp.make_engine_factory(
                lambda: get_engine(model_dir, create_cfg=(_lora_cfg or None)),
                bind_pipeline_model)
            engine = qfmp.QFLazyEngine(_factory, estimate_footprint(model_dir),
                                       retire=retire_handle)
            device = comfy.model_management.get_torch_device()
            offload = comfy.model_management.unet_offload_device()
            unet_config = {"image_model": "qwen_image21", "disable_unet_model_creation": True}
            model_config = comfy.supported_models.QwenImage21(unet_config)
            qfmp.ensure_model_config_attrs(model_config)
            model = QFQwenImage21Model(model_config, engine, device=device)
            _register_model(model)
            patcher = QFModelPatcher(model, load_device=device, offload_device=offload)
            print(f"[qf_native] loaded QuantFuncNativeLoader (Qwen-Image-2.1 t2i svdq) package={model_name} "
                  f"footprint~{engine.footprint_bytes // (1024*1024)}MB (create deferred)", flush=True)
            return qfmp.tag_lora_rebuild(patcher, lora_entries, _build)

        return _build(list(lora_entries))

    return build
