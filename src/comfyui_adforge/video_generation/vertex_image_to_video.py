"""
Image to Video Node for ComfyUI

Animate static images into videos using Google GenAI SDK.
"""

from typing import Optional

from comfy_api.latest import IO, ImageInput, ui
from google.genai.types import GenerateVideosConfig, GenerateVideosSource, Image

from comfyui_adforge import settings, utils
from comfyui_adforge.documentation import get_documentation, get_tooltip


class VertexVeoImageToVideoNode(IO.ComfyNode):
    """
    Generate a video from an input image using Google GenAI Veo model.
    """

    @classmethod
    def define_schema(cls):
        """Define input parameters for the node."""
        return IO.Schema(
            node_id="VertexVeoImageToVideoNode",
            display_name="Vertex Veo Image to Video",
            category="AdForge/Video Generation",
            description=get_documentation("VertexVeoImageToVideoNode"),
            inputs=[
                IO.String.Input(
                    "prompt",
                    tooltip=get_tooltip("prompt"),
                    force_input=True,
                ),
                IO.String.Input(
                    "negative_prompt",
                    tooltip=get_tooltip("negative_prompt"),
                    optional=True,
                    force_input=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip=get_tooltip("input_image"),
                ),
                IO.String.Input(
                    "input_image_gcs_uri",
                    placeholder="gs://bucket_name/path/to/image.jpg",
                    tooltip=get_tooltip("input_image_gcs_uri"),
                    optional=True,
                ),
                IO.String.Input(
                    "output_gcs_uri",
                    default=settings.get_default_gcs_uri("image-to-video"),
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
                    "image_mime_type",
                    options=settings.ImageMimeType.options(),
                    default=settings.ImageMimeType.default(),
                    tooltip=get_tooltip("mime_type"),
                    optional=True,
                ),
                IO.Int.Input(
                    "duration_seconds",
                    default=settings.DEFAULT_DURATION_SECONDS,
                    min=4,
                    max=8,
                    step=1,
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
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=get_tooltip("fps"),
                    optional=True,
                ),
                IO.Int.Input(
                    "number_of_videos",
                    default=1,
                    min=1,
                    max=4,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=get_tooltip("number_of_videos"),
                    optional=True,
                ),
                IO.Boolean.Input(
                    "enhance_prompt",
                    default=True,
                    tooltip=get_tooltip("enhance_prompt"),
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=True,
                    tooltip=get_tooltip("generate_audio"),
                    optional=True,
                ),
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
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip=get_tooltip("seed"),
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
        input_image_gcs_uri: Optional[str],
        negative_prompt: Optional[str],
        output_gcs_uri: Optional[str],
        model: settings.VeoModel,
        aspect_ratio: settings.AspectRatio,
        image_mime_type: settings.ImageMimeType,
        duration_seconds: int,
        resolution: settings.Resolution,
        fps: int,
        seed: int,
        number_of_videos: int,
        enhance_prompt: bool,
        generate_audio: bool,
        person_generation: settings.PersonGeneration,
        image: Optional[ImageInput] = None,
    ) -> IO.NodeOutput:
        """Generate video from image and prompt."""

        if not output_gcs_uri:
            output_gcs_uri = settings.get_default_gcs_uri("image-to-video")

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
            image_bytes = utils.bytify_image(image)

            if image_bytes:
                source_image = Image(image_bytes=image_bytes, mime_type=image_mime_type)
            elif input_image_gcs_uri:
                source_image = Image(gcs_uri=input_image_gcs_uri, mime_type=image_mime_type)
            else:
                raise ValueError("Either an image or an image GCS URI must be provided.")

            operation = client.models.generate_videos(
                model=model,
                source=GenerateVideosSource(prompt=prompt, image=source_image),
                config=GenerateVideosConfig(**config_params),
            )

            result = utils.poll_operation(client, operation)

            filename_prefix = "vertex-i2v"
            videos, video_paths, previews = utils.process_genai_results(result, filename_prefix)

            return IO.NodeOutput(videos, video_paths, ui=ui.PreviewVideo(previews))

        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}")
