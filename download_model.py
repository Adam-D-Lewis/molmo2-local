"""Download the Molmo2 model ahead of time so chat.py starts fast."""
from huggingface_hub import snapshot_download

MODEL_ID = "allenai/Molmo2-4B"

print(f"Downloading {MODEL_ID}...")
path = snapshot_download(MODEL_ID)
print(f"Model cached at: {path}")
print("You can now run: pixi run chat")
