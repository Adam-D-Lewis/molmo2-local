"""
Molmo2 interactive chat — runs the 4B model locally.

On Linux with an NVIDIA GPU the model is 4-bit quantized (~6.5 GB VRAM).
On macOS (Apple Silicon) or CPU-only machines it runs in float16/float32.

Usage:
  pixi run chat                          # text-only chat
  pixi run chat --image path/to/img.jpg  # start with an image loaded
  pixi run chat --video path/to/vid.mp4  # start with a video loaded

In-chat commands:
  /image path/to/img.jpg   Load a new image
  /video path/to/vid.mp4   Load a new video
  /clear                   Clear current image/video
"""

import argparse
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_ID = "allenai/Molmo2-4B"


def _get_device():
    """Return the best available device string and compute dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model():
    device, dtype = _get_device()
    use_quant = device == "cuda"

    if use_quant:
        from transformers import BitsAndBytesConfig

        print(f"Loading {MODEL_ID} (4-bit quantized on {device})...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["vision_backbone"],
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        print(f"Loading {MODEL_ID} (float16 on {device})...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        padding_side="left",
    )
    return model, processor, device, dtype


def generate(model, processor, device, dtype, prompt: str, image_path: str | None = None, video_path: str | None = None):
    content = [{"type": "text", "text": prompt}]
    if image_path:
        img = Image.open(image_path).convert("RGB")
        content.append({"type": "image", "image": img})
    if video_path:
        content.append({"type": "video", "video": video_path})

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.autocast(device, dtype=dtype):
        output = model.generate(**inputs, max_new_tokens=512)

    return processor.decode(output[0, inputs["input_ids"].size(1):], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Chat with Molmo2")
    parser.add_argument("--image", type=str, default=None, help="Path to an image file")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file")
    args = parser.parse_args()

    model, processor, device, dtype = load_model()

    image_path = args.image
    video_path = args.video

    print("\nMolmo2 ready! Type your prompt (Ctrl+C to quit).")
    print("Commands: /image <path>, /video <path>, /clear\n")
    if image_path:
        print(f"  Image loaded: {image_path}")
    if video_path:
        print(f"  Video loaded: {video_path}")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue

            if prompt.startswith("/image "):
                path = prompt[7:].strip()
                if os.path.isfile(path):
                    image_path = path
                    video_path = None
                    print(f"  Image loaded: {image_path}\n")
                else:
                    print(f"  File not found: {path}\n")
                continue

            if prompt.startswith("/video "):
                path = prompt[7:].strip()
                if os.path.isfile(path):
                    video_path = path
                    image_path = None
                    print(f"  Video loaded: {video_path}\n")
                else:
                    print(f"  File not found: {path}\n")
                continue

            if prompt == "/clear":
                image_path = None
                video_path = None
                print("  Cleared image/video.\n")
                continue

            response = generate(model, processor, device, dtype, prompt, image_path, video_path)
            print(f"\nMolmo2: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
