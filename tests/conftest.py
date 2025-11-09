import os
import sys


# conftest.py
def pytest_configure(config):
    import sys

    sys._called_from_test = True


def pytest_unconfigure(config):
    import sys

    del sys._called_from_test


# Now your relative imports might work, but it's often better to use absolute imports
# Add ComfyUI path to Python path so comfy_api can be imported
COMFYUI_CODE_PATH = os.environ.get("COMFYUI_CODE_PATH")
if os.path.exists(COMFYUI_CODE_PATH):
    sys.path.insert(0, COMFYUI_CODE_PATH)
