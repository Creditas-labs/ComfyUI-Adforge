"""
Video with Reference Node for ComfyUI

Generate videos using a reference image for style guidance with Google GenAI SDK.
"""

from typing import Optional

from comfy_api.latest import IO, ImageInput, ui
from google.genai.types import (
    GenerateVideosConfig,
    GenerateVideosSource,
    Image,
    VideoGenerationReferenceImage,
    VideoGenerationReferenceType,
)

from .. import settings, utils
from ..documentation import get_documentation, get_tooltip


class VertexVeoVideoWithReferenceNode(IO.ComfyNode):
    """
    Generate a video using a reference image for style or content guidance.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        """Define input parameters for the node."""
        return IO.Schema(
            node_id="VertexVeoVideoWithReferenceNode",
            display_name="Vertex Veo Video with Reference",
            category="AdForge/Video Generation",
            description=get_documentation("VertexVeoVideoWithReferenceNode"),
            inputs=[
                IO.String.Input("prompt", tooltip=get_tooltip("prompt"), force_input=True),
                IO.Image.Input("reference_image", tooltip=get_tooltip("reference_image"), optional=True),
                IO.String.Input(
                    "reference_image_gcs_uri",
                    placeholder=settings.get_default_gcs_uri("reference.png"),
                    tooltip=get_tooltip("reference_image_gcs_uri"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "reference_type",
                    options=settings.ReferenceType.options(),
                    default=settings.ReferenceType.default(),
                    tooltip=get_tooltip("reference_type"),
                    optional=True,
                ),
                IO.String.Input(
                    "negative_prompt", tooltip=get_tooltip("negative_prompt"), optional=True, force_input=True
                ),
                IO.String.Input(
                    "output_gcs_uri",
                    default=settings.get_default_gcs_uri("video-with-reference"),
                    tooltip=get_tooltip("output_gcs_uri"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=settings.VeoModel.options(),
                    default=settings.VeoModel.default(),
                    tooltip=get_tooltip("model"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=settings.AspectRatio.options(),
                    default=settings.AspectRatio.default(),
                    tooltip=get_tooltip("aspect_ratio"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "mime_type",
                    options=settings.ImageMimeType.options(),
                    default=settings.ImageMimeType.default(),
                    tooltip=get_tooltip("image_mime_type"),
                    optional=True,
                ),
                IO.Int.Input(
                    "duration_seconds",
                    default=settings.DEFAULT_DURATION_SECONDS,
                    min=1,
                    max=10,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=get_tooltip("duration_seconds"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=settings.Resolution.options(),
                    default=settings.Resolution.default(),
                    tooltip=get_tooltip("resolution"),
                    optional=True,
                ),
                IO.Int.Input(
                    "fps",
                    default=24,
                    min=1,
                    max=60,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=get_tooltip("fps"),
                    optional=True,
                ),
                IO.Int.Input(
                    "number_of_videos",
                    default=1,
                    min=1,
                    max=4,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=get_tooltip("number_of_videos"),
                    optional=True,
                ),
                IO.Boolean.Input("enhance_prompt", default=True, tooltip=get_tooltip("enhance_prompt"), optional=True),
                IO.Boolean.Input("generate_audio", default=False, tooltip=get_tooltip("generate_audio"), optional=True),
                IO.Combo.Input(
                    "person_generation",
                    options=settings.PersonGeneration.options(),
                    default=settings.PersonGeneration.default(),
                    tooltip=get_tooltip("person_generation"),
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=get_tooltip("seed"),
                    optional=True,
                ),
            ],
            outputs=[
                IO.Video.Output(
                    id="output_videos", display_name="videos", tooltip=get_tooltip("output_videos"), is_output_list=True
                ),
                IO.String.Output(
                    id="output_video_path_list",
                    display_name="video_path_list",
                    tooltip=get_tooltip("output_video_path_list"),
                    is_output_list=True,
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        reference_type: str,
        output_gcs_uri: str,
        model: settings.VeoModel,
        aspect_ratio: settings.AspectRatio,
        mime_type: settings.ImageMimeType,
        duration_seconds: int,
        resolution: settings.Resolution,
        fps: int,
        seed: int,
        number_of_videos: int,
        enhance_prompt: bool,
        generate_audio: bool,
        person_generation: settings.PersonGeneration,
        negative_prompt: Optional[str] = None,
        reference_image: Optional[ImageInput] = None,
        reference_image_gcs_uri: Optional[str] = None,
    ) -> IO.NodeOutput:
        """Generate video with reference image."""
        client = utils.get_genai_client()

        if not output_gcs_uri:
            output_gcs_uri = settings.get_default_gcs_uri("video-with-reference")

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
        }

        if negative_prompt:
            config_params["negative_prompt"] = negative_prompt
        if seed >= 0:
            config_params["seed"] = seed

        image_bytes = utils.bytify_image(reference_image)
        ref_image = None

        if image_bytes:
            ref_image = Image(image_bytes=image_bytes, mime_type=mime_type)
        elif reference_image_gcs_uri:
            ref_image = Image(gcs_uri=reference_image_gcs_uri, mime_type=mime_type)

        if not ref_image:
            raise ValueError("A reference image (local upload or GCS URI) must be provided.")

        ref_type_enum = VideoGenerationReferenceType[reference_type]
        config_params["reference_images"] = [
            VideoGenerationReferenceImage(image=ref_image, reference_type=ref_type_enum)
        ]

        try:
            operation = client.models.generate_videos(
                model=model,
                source=GenerateVideosSource(prompt=prompt),
                config=GenerateVideosConfig(**config_params),
            )

            result = utils.poll_operation(client, operation)

            videos, video_paths, previews = utils.process_genai_results(result, "vertex-ref")

            return IO.NodeOutput(videos, video_paths, ui=ui.PreviewVideo(previews))

        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}")
