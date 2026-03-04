# Molmo2 Local

Run [Molmo2](https://github.com/allenai/molmo2) (AI2's vision-language model) locally on a consumer GPU using 4-bit quantization. Supports image and video understanding.

## Requirements

- NVIDIA GPU with ~7GB free VRAM (tested on RTX 3060 12GB; the 4B model uses ~6.5GB with 4-bit quantization)
- [Pixi](https://pixi.sh) package manager
- ~8GB disk space for the 4B model weights

## Setup

```bash
cd ~/CodingProjects/molmo2-local
pixi install
```

## Usage

### 1. Download the model (one-time)

```bash
pixi run download
```

This downloads `allenai/Molmo2-4B` from HuggingFace (~8GB).

### 2. Chat

```bash
# Text-only
pixi run chat

# With an image
pixi run chat --image path/to/image.jpg

# With a video
pixi run chat --video path/to/video.mp4
```

Type prompts interactively. Ctrl+C to quit.

### In-chat commands

You can load images and videos during a chat session without restarting:

- `/image path/to/image.jpg` — load a new image (replaces any current video)
- `/video path/to/video.mp4` — load a new video (replaces any current image)
- `/clear` — clear current image/video for text-only mode

## VRAM usage

Measured on an RTX 3060 (12GB) with 4-bit NF4 quantization (vision backbone kept in full precision):

| Model | Params | Measured VRAM |
|-------|--------|---------------|
| `allenai/Molmo2-4B` | 4B | ~6.5 GB |

To try the larger models, edit `MODEL_ID` in `chat.py` and `download_model.py`:

- `allenai/Molmo2-O-7B` (7B params)
- `allenai/Molmo2-8B` (8B params)
