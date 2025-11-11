# """
# Text to Video Node for ComfyUI
#
# Generate videos from text prompts using Google GenAI SDK.
# """
# import os
# from typing import Optional
#
# from comfy_api.latest import IO, VideoFromFile, ui, VideoInput
# import folder_paths
#
# from comfyui_adforge.documentation import get_documentation, get_tooltip
#
#
# class PreviewVideo(IO.ComfyNode):
#     """
#     Previews a Video
#     """
#
#     @classmethod
#     def define_schema(cls):
#         """Define input parameters for the node."""
#         return IO.Schema(
#             node_id="PreviewVideo",
#             display_name="Preview Video",
#             category="AdForge/Tools",
#             description=get_documentation("PreviewVideo"),
#             inputs=[
#                 IO.Video.Input(
#                     "video",
#                     tooltip=get_tooltip("video_path"),
#                     optional=True,
#                 ),
#                 IO.String.Input(
#                     "video_path",
#                     tooltip=get_tooltip("video_path"),
#                     optional=True,
#                 )
#             ],
#             outputs=[],
#             is_output_node=True,
#         )
#
#     @classmethod
#     def execute(
#         cls,
#         video: Optional[VideoInput | VideoFromFile | str] = None,
#         video_path: Optional[str] = None,
#     ) -> IO.NodeOutput:
#         """Preview a video from a path or a video input."""
#
#         if hasattr(video, "get_file_path"): # VideoInput
#             video_path = video.get_file_path()
#         elif hasattr(video, "file_path"): # VideoFromFile
#             video_path = video.file_path
#
#         if not video_path or not os.path.exists(video_path):
#             return IO.NodeOutput(ui=ui.PreviewVideo([]))
#
#         filename = os.path.basename(video_path)
#         subfolder = os.path.relpath(os.path.dirname(video_path), folder_paths.get_output_directory())
#
#         return IO.NodeOutput(
#             ui=ui.PreviewVideo([ui.SavedResult(filename, subfolder, IO.FolderType.output)])
#         )
