"""Owning extra_conds/begin tests; CPU only, no native library or model weights."""
import json
import unittest
import torch

try:
    harness
except NameError:
    import ltx_reference_audio_test as harness
    SOURCE = ENGINE_SOURCE = SHARED_SOURCE = None


class MaskPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.ns = harness.production_namespace(
            source=SOURCE, engine_source=ENGINE_SOURCE, shared_source=SHARED_SOURCE)

    def make(self, av=True, batch=1):
        m = harness.model(self.ns, av)
        m._num_frames, m._audio_rows = 17, 4
        m.latent_shapes = [[batch, 128, 3, 2, 2], [batch, 8, 4, 16]]
        return m

    def mask(self, av, value, audio_value, dtype=torch.float32, batch=1):
        video = torch.ones((batch, 128, 3, 2, 2), dtype=dtype)
        video[:, :, 0] = value
        audio = torch.ones((batch, 8, 4, 16), dtype=dtype)
        audio[:, :, 0] = audio_value
        if av:
            return torch.cat((video.reshape(batch, -1), audio.reshape(batch, -1)), dim=1).unsqueeze(1)
        return video

    def expected(self, value, dtype, length):
        # Official floating condition conversion precedes multiplication by FP32 t.
        v = float(torch.tensor(value, dtype=dtype).to(torch.bfloat16))
        return None if v >= 0.9999 else [v] + [1.0] * (length - 1)

    def conds(self, m, mask):
        return m.extra_conds(denoise_mask=mask, frame_rate=24, unprocessed_ltxav_embeds=True)

    def test_fractional_input_dtypes_and_both_modalities(self):
        for av in (False, True):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for value in (0.0, 0.3, 0.5, 0.999, 0.9999, 1.0):
                    with self.subTest(av=av, dtype=dtype, value=value):
                        m = self.make(av)
                        mask = self.mask(av, value, value, dtype)
                        original = mask.clone()
                        self.conds(m, mask)
                        self.assertEqual(m._pending_frame_t_scale, self.expected(value, dtype, 3))
                        self.assertEqual(m._pending_audio_frame_t_scale,
                                         self.expected(value, dtype, 4) if av else None)
                        self.assertTrue(torch.equal(mask, original))

    def test_av_begin_carries_rounded_scales_and_same_abi(self):
        for video_value, audio_value in ((0.3, 0.0), (0.0, 0.3), (0.3, 0.3), (1.0, 1.0)):
            with self.subTest(video=video_value, audio=audio_value):
                m = self.make()
                self.conds(m, self.mask(True, video_value, audio_value))
                m._begin(torch.ones(1, 128, 3, 2, 2), torch.ones(1, 2, 4096))
                opts = json.loads(m._qf.lib.calls[-1]["options_utf8"])
                self.assertEqual(opts.get("video_frame_t_scale"), self.expected(video_value, torch.float32, 3))
                self.assertEqual(opts.get("audio_frame_t_scale"), self.expected(audio_value, torch.float32, 4))
                self.assertEqual(opts["num_frames"], 17)
                self.assertEqual(opts["fps"], 24.0)
                self.assertEqual(opts["audio_dims"], [1, 4, 128])
                self.assertTrue(opts["av_unprocessed_ctx"])

    def test_successive_runs_clear_scales(self):
        m = self.make()
        self.conds(m, self.mask(True, 0.3, 0.3))
        self.assertIsNotNone(m._pending_frame_t_scale)
        self.assertIsNotNone(m._pending_audio_frame_t_scale)
        self.conds(m, self.mask(True, 1.0, 1.0))
        self.assertIsNone(m._pending_frame_t_scale)
        self.assertIsNone(m._pending_audio_frame_t_scale)
        self.conds(m, self.mask(True, 0.3, 0.3))
        m.extra_conds(frame_rate=24, unprocessed_ltxav_embeds=True)
        self.assertIsNone(m._pending_frame_t_scale)
        self.assertIsNone(m._pending_audio_frame_t_scale)

    def test_nonfinite_and_hidden_spatial_variation_reject_before_rounding(self):
        for av in (False, True):
            for fault in ("nan", "inf", "hidden_variation"):
                with self.subTest(av=av, fault=fault):
                    m = self.make(av)
                    mask = self.mask(av, 0.3, 0.3)
                    mask.flatten()[0] = 0.3002 if fault == "hidden_variation" else float(fault)
                    with self.assertRaises(RuntimeError):
                        self.conds(m, mask)
                    self.assertFalse(m._qf.lib.calls)

    def test_hidden_audio_row_variation_rejects(self):
        m = self.make()
        mask = self.mask(True, 0.3, 0.3)
        video_elements = 128 * 3 * 2 * 2
        mask.flatten()[video_elements] = 0.3002
        self.assertEqual(mask.flatten()[video_elements].to(torch.bfloat16),
                         mask.flatten()[video_elements + 1].to(torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, "WITHIN an audio row"):
            self.conds(m, mask)
        self.assertFalse(m._qf.lib.calls)

    def test_target_batch_difference_rejects_even_if_rounding_would_hide_it(self):
        for av in (False, True):
            with self.subTest(av=av):
                m = self.make(av, batch=2)
                mask = self.mask(av, 0.3, 0.3, batch=2)
                if av:
                    mask[1, 0, :128 * 3 * 2 * 2].reshape(128, 3, 2, 2)[:, 0] = 0.3002
                else:
                    mask[1, :, 0] = 0.3002
                with self.assertRaisesRegex(RuntimeError, "same per-frame/audio-row"):
                    self.conds(m, mask)
                self.assertFalse(m._qf.lib.calls)

    def test_existing_clipping(self):
        for av in (False, True):
            for value in (-0.25, 1.25, 0.0):
                with self.subTest(av=av, value=value):
                    m = self.make(av)
                    mask = self.mask(av, value, value)
                    self.conds(m, mask)
                    clipped = max(0.0, min(1.0, value))
                    self.assertEqual(m._pending_frame_t_scale, self.expected(clipped, torch.float32, 3))
                    if av:
                        self.assertEqual(m._pending_audio_frame_t_scale, self.expected(clipped, torch.float32, 4))

    def test_all_zero_masks_and_identical_batches(self):
        for av in (False, True):
            for batch in (1, 2):
                with self.subTest(av=av, batch=batch):
                    m = self.make(av, batch)
                    mask = torch.zeros_like(self.mask(av, 0.0, 0.0, batch=batch))
                    self.conds(m, mask)
                    self.assertEqual(m._pending_frame_t_scale, [0.0] * 3)
                    self.assertEqual(m._pending_audio_frame_t_scale, [0.0] * 4 if av else None)


if __name__ == "__main__":
    unittest.main()
