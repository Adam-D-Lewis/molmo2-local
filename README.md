# Molmo2 Local

Run [Molmo2](https://github.com/allenai/molmo2) (AI2's vision-language model) locally using 4-bit quantization on NVIDIA GPUs, or in float16/float32 on macOS (Apple Silicon) and Windows.

## Requirements

- [Pixi](https://pixi.sh) package manager
- ~19GB disk space for the 4B model weights
- One of:
  - **Linux** — NVIDIA GPU with ~7GB free VRAM (4-bit quantized, tested on RTX 3060 12GB)
  - **macOS** (Apple Silicon) — ~8GB free RAM (runs in float16 via MPS)
  - **Windows** — NVIDIA GPU recommended; falls back to CPU if unavailable

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

This downloads `allenai/Molmo2-4B` from HuggingFace (~19GB — 4 shards in bf16, including a large vocab embedding and SigLIP vision backbone).

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

## Platform details

| Platform | Device | Quantization | Approx memory |
|----------|--------|-------------|---------------|
| Linux (NVIDIA) | CUDA | 4-bit NF4 | ~6.5 GB VRAM |
| macOS (Apple Silicon) | MPS | float16 | ~8 GB RAM |
| Windows (NVIDIA) | CUDA | 4-bit NF4 | ~6.5 GB VRAM |
| Windows / CPU | CPU | float32 | ~16 GB RAM |

Video support (`/video`, `--video`) requires `torchcodec` and `decord2`, which are currently only installed on Linux. Image and text chat work on all platforms.

To try the larger models, edit `MODEL_ID` in `chat.py` and `download_model.py`:

- `allenai/Molmo2-O-7B` (7B params)
- `allenai/Molmo2-8B` (8B params)
