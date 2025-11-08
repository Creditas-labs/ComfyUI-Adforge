# Adforge
Creditas' Ad Creation Toolkit for ComfyUI

<img src="docs/Creditas Logo.png" alt="Creditas" style="width:50%;height:auto;">

- [About](https://www.creditas.com/quem-somos)
- [Careers](https://creditas.gupy.io/)

## Quickstart
1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager as `adforge`. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

# Features
- Vertex AI API nodes implemented with [Google's Gen AI SDK](https://github.com/googleapis/python-genai)
- TBA

# Develop
1. Make sure you have [uv](https://docs.astral.sh/uv/reference/installer/#__tabbed_1_1) installed
2. Install dev dependencies:
```shell
make init
```

## Writing custom nodes
An example custom node is located in [node.py](src/adforge/nodes.py).

## Import python types
You need to add ComfyUI folders to your path.

### JetBrains IDEs
Follow [this doc](https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-reloading-interpreter-paths.html)
### vscode
Check the [.vscode](.vscode) folder



# Contributing
If you intend to add features, please follow Comfy's [docs](https://docs.comfy.org/development/overview) and use [cookiecutter-comfy-extension](https://github.com/Comfy-Org/cookiecutter-comfy-extension) folder/file structure