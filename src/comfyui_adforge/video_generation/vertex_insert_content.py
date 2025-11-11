"""
Video Insert Content Node for ComfyUI

Insert new content into existing videos using Google GenAI SDK.
"""

from typing import Optional

from comfy_api.latest import IO, ImageInput, VideoInput, ui
from google.genai.types import (
    GenerateVideosConfig,
    GenerateVideosSource,
    Image,
    Video,
    VideoGenerationMask,
    VideoGenerationMaskMode,
)

from .. import settings, utils
from ..documentation import get_documentation, get_tooltip


class VertexVeoInsertContentNode(IO.ComfyNode):
    """
    Insert new content into an existing video.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        """Define input parameters for the node."""
        return IO.Schema(
            node_id="VertexVeoInsertContentNode",
            display_name="Vertex Veo Insert Content",
            category="AdForge/Video Generation",
            description=get_documentation("VertexVeoInsertContentNode"),
            inputs=[
                IO.String.Input("prompt", tooltip=get_tooltip("prompt"), force_input=True),
                IO.Video.Input("video", tooltip=get_tooltip("video"), optional=True),
                IO.String.Input(
                    "video_gcs_uri",
                    placeholder=settings.get_default_gcs_uri("input-video.mp4"),
                    tooltip=get_tooltip("video_gcs_uri"),
                    optional=True,
                ),
                IO.Image.Input("mask_image", tooltip=get_tooltip("mask_image"), optional=True),
                IO.String.Input(
                    "mask_image_gcs_uri",
                    placeholder=settings.get_default_gcs_uri("mask.png"),
                    tooltip=get_tooltip("mask_image_gcs_uri"),
                    optional=True,
                ),
                # --- Optional ---
                IO.String.Input(
                    "negative_prompt", tooltip=get_tooltip("negative_prompt"), optional=True, force_input=True
                ),
                IO.String.Input(
                    "output_gcs_uri",
                    default=settings.get_default_gcs_uri("video-insert-content"),
                    tooltip=get_tooltip("output_gcs_uri"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "model", options=settings.VeoModel.options(), default=settings.VeoModel.default(), optional=True
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=settings.AspectRatio.options(),
                    default=settings.AspectRatio.default(),
                    tooltip=get_tooltip("aspect_ratio"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "video_mime_type",
                    options=settings.VideoMimeType.options(),
                    default=settings.VideoMimeType.default(),
                    tooltip=get_tooltip("mime_type"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "mask_mime_type",
                    options=settings.ImageMimeType.options(),
                    default=settings.ImageMimeType.default(),
                    tooltip=get_tooltip("mime_type"),
                    optional=True,
                ),
                IO.Int.Input(
                    "duration_seconds", default=settings.DEFAULT_DURATION_SECONDS, min=1, max=10, optional=True
                ),
                IO.Combo.Input(
                    "resolution",
                    options=settings.Resolution.options(),
                    default=settings.Resolution.default(),
                    optional=True,
                ),
                IO.Int.Input("fps", default=24, min=1, max=60, optional=True),
                IO.Int.Input("seed", default=0, min=0, max=2147483647, control_after_generate=True, optional=True),
                IO.Int.Input("number_of_videos", default=1, min=1, max=4, optional=True),
                IO.Boolean.Input("enhance_prompt", default=True, optional=True),
                IO.Boolean.Input("generate_audio", default=False, optional=True),
                IO.Combo.Input(
                    "person_generation",
                    options=settings.PersonGeneration.options(),
                    default=settings.PersonGeneration.default(),
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(id="output_videos", display_name="videos", is_output_list=True),
                IO.String.Output(id="output_video_path_list", display_name="video_path_list", is_output_list=True),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        video: Optional[VideoInput],
        video_gcs_uri: str,
        mask_image: Optional[ImageInput],
        mask_image_gcs_uri: str,
        output_gcs_uri: str,
        model: str,
        aspect_ratio: str,
        video_mime_type: str,
        mask_mime_type: str,
        duration_seconds: int,
        resolution: str,
        fps: int,
        seed: int,
        number_of_videos: int,
        enhance_prompt: bool,
        generate_audio: bool,
        person_generation: str,
        negative_prompt: Optional[str] = None,
    ) -> IO.NodeOutput:
        """Insert content into video using mask-based editing."""
        client = utils.get_genai_client()

        if not output_gcs_uri:
            output_gcs_uri = settings.get_default_gcs_uri("video-insert-content")

        mask_bytes = utils.bytify_image(mask_image)
        if mask_bytes:
            mask = Image(image_bytes=mask_bytes, mime_type=mask_mime_type)
        elif mask_image_gcs_uri:
            mask = Image(gcs_uri=mask_image_gcs_uri, mime_type=mask_mime_type)
        else:
            raise ValueError("Either a mask image or a mask image GCS URI must be provided.")

        config_params = {
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_seconds,
            "resolution": resolution,
            "fps": fps,
            "enhance_prompt": enhance_prompt,
            "generate_audio": generate_audio,
            "number_of_videos": number_of_videos,
            "person_generation": person_generation,
            "compression_quality": "LOSSLESS",
            "output_gcs_uri": output_gcs_uri,
            "mask": VideoGenerationMask(image=mask, mask_mode=VideoGenerationMaskMode.INSERT),
        }

        if negative_prompt:
            config_params["negative_prompt"] = negative_prompt
        if seed >= 0:
            config_params["seed"] = seed

        video_bytes = utils.bytify_video(video or video_gcs_uri)
        if video_bytes:
            source_video = Video(video_bytes=video_bytes, mime_type=video_mime_type)
        else:
            raise ValueError("Either an input video or a video GCS URI must be provided.")

        operation = client.models.generate_videos(
            model=model,
            source=GenerateVideosSource(prompt=prompt, video=source_video),
            config=GenerateVideosConfig(**config_params),
        )

        result = utils.poll_operation(client, operation)
        videos, video_paths, previews = utils.process_genai_results(result, "vertex-insert")

        return IO.NodeOutput(videos, video_paths, ui=ui.PreviewVideo(previews))
