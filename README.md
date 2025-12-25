# Prompt2Edit

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A zero-shot semantic image editing pipeline that enables intelligent object replacement in images using natural language prompts.

## Overview

**Prompt2Edit** is an image editing tool that allows you to replace specific objects in images using text descriptions. by providing:
1. An input image 
2. A text description of what to find in the image
3. A text description of what to replace it with

The system automatically detects the object, segments it, and uses stable diffusion inpainting models to replace it.

![Project pipeline](https://raw.githubusercontent.com/sameddallaa/prompt2edit/refs/heads/main/reports/figures/pipeline.svg)

## Features

- **Text-Guided Object Detection**: Uses GROUNDED-SAM for zero-shot object detection with text prompts
- **Semantic Segmentation**: Automatically segments detected objects
- **Intelligent Inpainting**: Uses diffusion models for object replacement

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU (slower)
- PyTorch with appropriate CUDA support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sameddallaa/prompt2edit
cd prompt2edit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python main.py <image_path> "<detect_prompt>" "<replace_prompt>"
```

**Arguments:**
- `image_path`: Path to your input image
- `detect_prompt`: Text description of the object to find (e.g., "a cat")
- `replace_prompt`: Text description of what to replace it with (e.g., "a dog wearing sunglasses")

**Example:**
```bash
python main.py photo.jpg "a red car" "a blue bicycle"
```

## Example results

![Results](https://raw.githubusercontent.com/sameddallaa/prompt2edit/refs/heads/main/reports/figures/output.png)