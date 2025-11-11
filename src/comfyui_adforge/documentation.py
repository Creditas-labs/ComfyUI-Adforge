"""
Documentation infrastructure for AdForge nodes.
"""


def format_dict_as_text(dictionary, depth=0):
    """Format a dictionary as a text string."""
    text = ""
    for key, value in dictionary.items():
        indent = "  " * depth
        if isinstance(value, dict):
            text += f"{indent}{key}:\n{format_dict_as_text(value, depth + 1)}"
        elif isinstance(value, list):
            text += f"{indent}{key}:\n" + "\n".join(f"{indent}  - {item}" for item in value) + "\n"
        else:
            text += f"{indent}{key}: {value}\n"
    return text


def create_documentation(title, short_description, sections):
    """
    Create formatted documentation for a node.

    Args:
        title: Node title
        short_description: Brief description
        sections: Dict of inputs and outputs documentation sections.

    Returns:
        Formatted documentation string
    """
    text = f"{title}\n{short_description}\n\n"
    text += format_dict_as_text(sections)
    return text


CONFIGURATION = {
    "aspect_ratio": "Optional. Specifies the aspect ratio of generated videos.",
    "duration_seconds": "Required. The length in seconds of video files that you want to generate.",
    "enhance_prompt": "Optional. Use Gemini to enhance your prompts.",
    "generate_audio": "Generate audio for the video.",
    "person_generation": (
        "Optional. The safety setting that controls whether people or face " "generation is allowed."
    ),
    "resolution": "Optional. Veo 3 models only. The resolution of the generated video.",
    "seed": (
        "Optional. A number to request to make generated videos deterministic. Adding a seed "
        "number with your request without changing other parameters will cause the model to "
        "produce the same videos."
    ),
    "output_gcs_uri": (
        "GCS URI where the generated videos will be stored, in the format " "'gs://BUCKET_NAME/SUBDIRECTORY'."
    ),
    "model": "The Veo model to use for video generation.",
    "fps": "Optional. Frames per second for the generated video.",
    "number_of_videos": "Optional. Number of video variations to generate.",
    "mime_type": "Mime type of the input image or video, e.g., 'image/png' or 'video/mp4'.",
}
# "resizeMode": (
#     "Optional. Veo 3 models only, used with image for image-to-video. The resize "
#     "mode that the model uses to resize the video. Accepted values are "pad" "
#     "(default) or "crop"."
# ),

OUTPUTS = {
    "output_videos": "A list of generated videos (VIDEO type).",
    "output_video_path_list": "A list of local paths to the generated videos (when output_format=local_file)",
}

INPUTS = {
    "prompt": "The text prompt used to guide video generation.",
    "negative_prompt": (
        "Optional. A text string that describes anything you want to discourage " "the model from generating."
    ),
    "input_image": "Input image to animate (IMAGE type).",
    "input_image_gcs_uri": ("GCS URI of the input image in the format 'gs://BUCKET_NAME/SUBDIRECTORY.'."),
    "mask": "The mask to use for generating videos.",
    "reference_images": (
        "The images to use as the references to generate the videos. If this "
        "field is provided, the text prompt field must also be provided. The image, "
        "video, or last_frame field are not supported. Each image must be associated "
        "with a type. Veo 2 supports up to 3 asset images *or* 1 style image."
    ),
    "last_frame": (
        "Image to use as the last frame of generated videos. Only supported for image " "to video use cases."
    ),
    "first_frame": (
        "Image to use as the first frame of generated videos. Only supported for image " "to video use cases."
    ),
    "reference_type": "Asset image: You provide up to three images of a single person, character, or product. "
    "Veo preserves the subject's appearance in the output video. Style image: You provide a single style image.  "
    "Veo applies the style from your uploaded image in the output video. This feature "
    "is only supported by veo-2.0-generate-exp in Preview.",
}

TOOLTIPS = {
    **INPUTS,
    **OUTPUTS,
    **CONFIGURATION,
}


def get_tooltip(io_name: str):
    """Get the tooltip"""
    return TOOLTIPS.get(io_name, "No documentation available")


# Documentation for all AdForge nodes
DOCUMENTATION = {
    "VertexVeoImageToVideoNode": create_documentation(
        "Image to Video",
        "Animate static images with motion prompts using Google's Veo",
        {
            "Inputs": {
                "prompt": get_tooltip("prompt"),
                "negative_prompt": get_tooltip("negative_prompt"),
                "input_image": get_tooltip("input_image"),
                "input_image_gcs_uri": get_tooltip("input_image_gcs_uri"),
            },
            "Outputs": OUTPUTS,
            "Configuration": CONFIGURATION,
        },
    ),
    "VertexVeoTextToVideoNode": create_documentation(
        "Text to Video",
        "Generate videos from text prompts using Google's Veo",
        {
            "Inputs": {
                "prompt": get_tooltip("prompt"),
                "negative_prompt": get_tooltip("negative_prompt"),
            },
            "Outputs": OUTPUTS,
            "Configuration": CONFIGURATION,
        },
    ),
    "VertexVeoVideoWithReferenceNode": create_documentation(
        "Video with Reference",
        "Generate videos using a reference image for style guidance with Google's Veo.",
        {
            "Inputs": {
                "prompt": get_tooltip("prompt"),
                "reference_image": "The image to use as a reference for content or style.",
            },
            "Outputs": OUTPUTS,
            "Configuration": CONFIGURATION,
        },
    ),
}


def get_documentation(node_class_name):
    """Get documentation for a node class."""
    return DOCUMENTATION.get(node_class_name, "No documentation available")
