"""Reuse Stage 1 Qwen conditioning; rebuild Stage 2 spatial conditions with ComfyUI."""

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo


class _ReuseH3Encoding:
    def __init__(self, positive):
        if not positive:
            raise ValueError("MiniMax-H3 Stage 2 needs the Stage 1 positive conditioning")
        # Spatial payloads must be rebuilt at the target size; preserve embeddings and token tags.
        self.positive = [[embedding, {k: v for k, v in values.items()
                                     if k not in ("minimax_keyframes", "minimax_refs")}]
                         for embedding, values in positive]

    def tokenize(self, *args, **kwargs):
        return None

    def encode_from_tokens_scheduled(self, tokens):
        return self.positive


class QuantFuncH3ImageToVideoStage2(MiniMaxH3ImageToVideo):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "QuantFuncH3ImageToVideoStage2"
        schema.display_name = "QuantFunc MiniMax-H3 Stage 2 Image Conditioning"
        schema.description = ("Reuses Stage 1 Qwen text/vision embeddings and token tags. Connect the same first/last "
                              "images; their VAE keyframes are rebuilt at the Stage 2 size.")
        schema.inputs = [io.Conditioning.Input("positive")] + [
            entry for entry in schema.inputs if entry.id not in ("clip", "prompt")]
        return schema

    @classmethod
    def execute(cls, positive, vae, width, height, length, first_frame=None, last_frame=None):
        return MiniMaxH3ImageToVideo.execute(
            clip=_ReuseH3Encoding(positive), vae=vae, prompt="", width=width, height=height, length=length,
            first_frame=first_frame, last_frame=last_frame)


class QuantFuncH3ReferenceToVideoStage2(MiniMaxH3ReferenceToVideo):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "QuantFuncH3ReferenceToVideoStage2"
        schema.display_name = "QuantFunc MiniMax-H3 Stage 2 Reference Conditioning"
        schema.description = ("Reuses Stage 1 Qwen text/vision embeddings and token tags. Connect the same references; "
                              "their VAE conditions are rebuilt using the Stage 2 size and reference sizing mode.")
        schema.inputs = [io.Conditioning.Input("positive")] + [
            entry for entry in schema.inputs if entry.id not in ("clip", "prompt")]
        return schema

    @classmethod
    def execute(cls, positive, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        return MiniMaxH3ReferenceToVideo.execute(
            clip=_ReuseH3Encoding(positive), prompt="", width=width, height=height, length=length,
            ref_image_size=ref_image_size,
            vae=vae, audio_vae=audio_vae, ref_images=ref_images, ref_videos=ref_videos,
            ref_video_audios=ref_video_audios, ref_audios=ref_audios)
