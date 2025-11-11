"""
Video First-Last Frame Node for ComfyUI

Generate videos guided by first and last frames using Google GenAI SDK.
"""

from typing import Optional

from comfy_api.latest import IO, ImageInput, ui
from google.genai.types import GenerateVideosConfig, GenerateVideosSource, Image

from .. import settings, utils
from ..documentation import get_documentation, get_tooltip


class VertexVeoFirstLastFrameToVideoNode(IO.ComfyNode):
    """Generate a video guided by first and last frames."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        """Define input parameters for the node."""
        return IO.Schema(
            node_id="VertexVeoFirstLastFrameToVideoNode",
            display_name="Vertex Veo First-Last Frame to Video",
            category="AdForge/Video Generation",
            description=get_documentation("FirstLastFrameToVideoNode"),
            inputs=[
                IO.String.Input(
                    "prompt",
                    tooltip=get_tooltip("prompt"),
                    force_input=True,
                ),
                IO.Image.Input(
                    "first_frame_image",
                    tooltip=get_tooltip("first_frame_image"),
                    optional=True,
                ),
                IO.Image.Input(
                    "last_frame_image",
                    tooltip=get_tooltip("last_frame_image"),
                    optional=True,
                ),
                IO.String.Input(
                    "first_frame_gcs_uri",
                    placeholder=settings.get_default_gcs_uri("first-frame.png"),
                    tooltip=get_tooltip("first_frame_gcs_uri"),
                    optional=True,
                ),
                IO.String.Input(
                    "last_frame_gcs_uri",
                    placeholder=settings.get_default_gcs_uri("last-frame.png"),
                    tooltip=get_tooltip("last_frame_gcs_uri"),
                    optional=True,
                ),
                # --- Optional ---
                IO.String.Input(
                    "negative_prompt",
                    tooltip=get_tooltip("negative_prompt"),
                    optional=True,
                    force_input=True,
                ),
                IO.String.Input(
                    "output_gcs_uri",
                    default=settings.get_default_gcs_uri("video-first-last-frame"),
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
                    "first_frame_mime_type",
                    options=settings.ImageMimeType.options(),
                    default=settings.ImageMimeType.default(),
                    tooltip=get_tooltip("mime_type"),
                    optional=True,
                ),
                IO.Combo.Input(
                    "last_frame_mime_type",
                    options=settings.ImageMimeType.options(),
                    default=settings.ImageMimeType.default(),
                    tooltip=get_tooltip("mime_type"),
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
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=get_tooltip("seed"),
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
            ],
            outputs=[
                IO.Video.Output(
                    id="output_videos",
                    display_name="videos",
                    tooltip=get_tooltip("output_videos"),
                    is_output_list=True,
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
        first_frame_gcs_uri: str,
        last_frame_gcs_uri: str,
        output_gcs_uri: str,
        model: settings.VeoModel,
        aspect_ratio: settings.AspectRatio,
        first_frame_mime_type: settings.ImageMimeType,
        last_frame_mime_type: settings.ImageMimeType,
        duration_seconds: int,
        resolution: settings.Resolution,
        fps: int,
        seed: int,
        number_of_videos: int,
        enhance_prompt: bool,
        generate_audio: bool,
        person_generation: settings.PersonGeneration,
        negative_prompt: Optional[str] = None,
        first_frame_image: Optional[ImageInput] = None,
        last_frame_image: Optional[ImageInput] = None,
    ) -> IO.NodeOutput:
        """Generate video guided by first and last frames."""
        client = utils.get_genai_client()

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

        try:
            first_frame_bytes = utils.bytify_image(first_frame_image)
            last_frame_bytes = utils.bytify_image(last_frame_image)

            if first_frame_bytes and last_frame_bytes:
                config_params["last_frame"] = Image(image_bytes=last_frame_bytes, mime_type=last_frame_mime_type)
                source_image = Image(image_bytes=first_frame_bytes, mime_type=first_frame_mime_type)
            elif first_frame_gcs_uri and last_frame_gcs_uri:
                config_params["last_frame"] = Image(gcs_uri=last_frame_gcs_uri, mime_type=last_frame_mime_type)
                source_image = Image(gcs_uri=first_frame_gcs_uri, mime_type=first_frame_mime_type)
            else:
                raise ValueError("Either both images or both GCS URIs must be provided.")

            operation = client.models.generate_videos(
                model=model,
                source=GenerateVideosSource(prompt=prompt, image=source_image),
                config=GenerateVideosConfig(**config_params),
            )

            result = utils.poll_operation(client, operation)

            videos, video_paths, previews = utils.process_genai_results(result, "vertex-fl-frame")

            return IO.NodeOutput(videos, video_paths, ui=ui.PreviewVideo(previews))
        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}")
