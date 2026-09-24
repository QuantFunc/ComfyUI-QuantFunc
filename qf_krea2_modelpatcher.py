"""qf_krea2_modelpatcher — the Krea-2 (image t2i) family seam (model + builder), self-contained.

The FIRST image family on the native seam (user 2026-08-29): same denoise_only shape as the
video families — comfy owns CLIP(Qwen3-VL taps)/VAE/sampler; the engine owns ONLY the
28-block single-stream MMDiT denoise via the generic external session
(Krea2Pipeline::prepareExternalDenoise / quantfunc_denoise_begin + quantfunc_denoise_step).
comfy's krea2 cond arrives as (B, seq, txtlayers*txtdim) — comfy/ldm/krea2/model.py:304 —
which is byte-identical to the engine's expected [1, S, L*td] staging, so cond passes through.
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
import weakref


FAMILY = "krea2"


def matches(pipeline_class, transformer_class=""):
    """The engine's own detector: Krea2Pipeline / Krea2Transformer2DModel."""
    return (str(pipeline_class).startswith("Krea2")
            or str(transformer_class) == "Krea2Transformer2DModel")


class QFKrea2Model(QFSessionModelMixin, comfy.model_base.Krea2):
    _VAE_S = 8   # AutoencoderKLQwenImage spatial scale; session W/H = latent * 8

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)
        self.diffusion_model = _QFStub()
        self.diffusion_model.arm_concat_shape(16)   # in==latent channels → extra 0 → no concat build
        self._qf = engine
        self._num_steps = 0
        self._step_i = 0
        self._ctx_key_assigner = _CtxKeyAssigner()
        self._sess_denoise = 0
        self._max_batch = 0
        self._max_ctx_seq = 0
        self._out = None

    # comfy model_base.Krea2.extra_conds consumables this seam cannot honor — refuse LOUD,
    # never silently drop (the wan reject-list discipline; reference_latents = the krea2
    # ref2img channel, no field in the generic denoise ABI).
    # + the BaseModel inpaint/concat channels (concat_latent_image/concat_mask/denoise_mask): the
    # reject_list_completeness audit (Krea2 row, registered 2026-09-22) found them unguarded.
    _ENGINE_IGNORED_COND_KEYS = ("reference_latents", "reference_latents_method",
                                 "cross_attn_controlnet", "noise_concat",
                                 "concat_latent_image", "concat_mask", "denoise_mask")

    def extra_conds(self, **kwargs):
        was_open, ok = self._qf.end_session_if_open()   # run-start clean slate (#2 lifecycle)
        self._qf_needs_begin = True
        if was_open:
            qfe.info("[qf_native] krea2: closed a pre-existing session at run start "
                  f"(prior run interrupted/uncleaned); end ok={ok}", flush=True)
        for _k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(_k) is not None:
                raise RuntimeError(
                    f"qf_native krea2: '{_k}' conditioning is wired, but the native session "
                    f"cannot consume it — it would be silently ignored, so it is refused. "
                    f"Remove the node feeding '{_k}' (reference/edit conditioning is not part "
                    f"of the krea2 t2i seam).")
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            self._max_ctx_seq = max(self._max_ctx_seq, int(cross_attn.shape[1]))
        return out

    def _begin(self, x_group, ctx_group):
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
        # IMAGE session: only the dials every family takes (the video residency_opts keys are engine-refused here).
        bpx._opts = json.dumps(self.dial_opts()).encode()
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        st = lib.quantfunc_denoise_begin(self._qf.pipeline, ctypes.byref(bpx), ctypes.byref(session))
        self._begin_keep = bpx
        if st != qfe.QUANTFUNC_OK:
            raise RuntimeError(f"denoise_begin (krea2 t2i) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        self._qf.unloaded = False
        self._step_i = 0
        self._sess_denoise = 0
        self._max_batch = 0
        self._ctx_key_assigner.reset()
        qfe.info(f"[qf_native] KREA2 SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
              f"latent={tuple(x_group.shape)} cond={tuple(ctx_group.shape)}", flush=True)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        ctx = c_crossattn
        if ctx is None:
            raise RuntimeError("qf_native krea2: no c_crossattn cond — wire a krea2 CLIPTextEncode")
        if control is not None:
            raise RuntimeError(
                "qf_native krea2: a ControlNet is wired, but the native session consumes no "
                "control input — it would be silently ignored, so it is refused.")
        if c_concat is not None:
            raise RuntimeError(
                "qf_native krea2: c_concat conditioning is not part of the t2i seam — remove "
                "the node feeding it.")
        # krea2 engine is a BF16-latent/BF16-cond family (factory dtype=Tensor::BF16;
        # the video seams run FP32 latents — NOT this one). comfy hands FP32 through the
        # stubbed model config, so the seam casts HERE (x for the step ABI, ctx for begin).
        # DEVICE discipline (measured on cu12/3090, 2026-08-30): the engine step ABI
        # takes DEVICE pointers; comfy is free to hand the cond on CPU (its device
        # placement varies by version/vram mode) — a CPU ctx data_ptr reached the
        # engine's cond copy as cudaMemcpy 'invalid argument' (size=26x30720x2 exact).
        # Cast to the LATENT's device + bf16 in one .to() — a no-op when already there
        # (the 4090 boxes), the fix where comfy kept cond host-side.
        _dev = x.device
        if x.dtype != torch.bfloat16:
            x_bf = x.to(torch.bfloat16)
        else:
            x_bf = x
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
        # comfy's key is "uuids" (samplers.py:324/511 — what the wan/ltx/h3 seams read); "cond_uuids" never
        # existed, so every Krea2 ctx key was 0 = the engine's step caches OFF (found 2026-09-22 via the
        # QI2.1 seam cloned from this file: the engine logged cfg_context_key=0).
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
        for i in range(B):
            _interrupt_poll_end_session_on_raise(self._qf)
            xi = xin[i:i + 1].contiguous()
            oi = self._out[i:i + 1]
            # comfy's krea2 rides the Wan21 VIDEO latent format ([B,C,T=1,H,W]); the
            # engine session is IMAGE 4D — squeeze the singleton T for the ABI (both
            # views share storage, so the velocity lands back in _out correctly).
            if xi.dim() == 5 and xi.shape[2] == 1:
                xi = xi.squeeze(2).contiguous()
                oi = oi.squeeze(2)
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
            self._call_denoise_step(p, f"krea2 denoise_step[step={step_index},group={i},key={ctx_key}]")
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
            qfe.info(f"[qf_native] KREA2 SESSION CLOSED after {self._step_i} sampler steps, "
                  f"{self._sess_denoise} denoise_step calls", flush=True)
        return super().process_latent_out(latent)


def register(deps):
    get_engine = deps["get_engine"]
    bind_pipeline_model = deps["bind_pipeline_model"]

    def build(transformer1_path, bundle_dir=None, lora_entries=()):
        """File-based Krea-2 Turbo t2i — the H3 single-expert staging pattern: stage the
        shipped config bundle (configs/krea2-turbo-*/, minimal official skeleton) + symlink
        the transformer file; engine create runs denoise_only=True (TE + VAE weights
        skipped — comfy's krea2 CLIP owns conditioning, comfy's VAEDecode decodes; the
        engine reads the staged configs for session geometry only)."""
        model_dir = qfmp.stage_denoise_only_package(bundle_dir, transformer1_path)
        create_extra = {"denoise_only": True}
        model_name = os.path.basename(transformer1_path)

        def _build(lora_entries):
            _lora_cfg = dict(create_extra or {})
            # NO "lora" in the create: the cache key is the weights only (user rule 2026-09-24), so every LoRA set of
            # this model shares one pipeline; the lazy engine below applies THIS build's set in place (QFLazyEngine._apply_runtime_lora).
            device, device_idx = qfmp.current_torch_device()
            _factory, _register_model = qfmp.make_engine_factory(
                lambda: get_engine(model_dir, create_cfg=(_lora_cfg or None),
                                   device_idx=device_idx),
                bind_pipeline_model)
            engine = qfmp.QFLazyEngine(_factory)
            offload = comfy.model_management.unet_offload_device()
            unet_config = {"image_model": "krea2", "disable_unet_model_creation": True}
            model_config = comfy.supported_models.Krea2(unet_config)
            qfmp.ensure_model_config_attrs(model_config)
            model = QFKrea2Model(model_config, engine, device=device)
            _register_model(model)
            patcher = QFModelPatcher(model, load_device=device, offload_device=offload)
            qfe.info(f"[qf_native] loaded QuantFuncNativeLoader (Krea-2 t2i svdq) package={model_name} "
                  f"capacity=native Prepared query (create deferred)", flush=True)
            return qfmp.tag_lora_rebuild(patcher, lora_entries, _build)

        return _build(list(lora_entries))

    return build
