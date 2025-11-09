"""Top-level package for comfyui_adforge."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """Creditas-labs"""
__email__ = "martech1@creditas.com"
__version__ = "0.0.1"

# mymodule/__init__.py
import sys

if not hasattr(sys, "_called_from_test"):
    from .src.comfyui_adforge.nodes import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
