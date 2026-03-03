"""
PoisonPill Model Downloader
Downloads the CLIP ViT-B/32 ONNX model on first launch.
Stores it in the engine/models/ directory.
"""
import os
import sys
import urllib.request

MODEL_URL = "https://clip-as-service.jina.ai/api/models/onnx/ViT-B-32/visual.onnx"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "clip_visual.onnx")

def progress_hook(count, block_size, total_size):
    pct = int(count * block_size * 100 / total_size)
    pct = min(pct, 100)
    sys.stdout.write(f"\rDOWNLOAD_PROGRESS:{pct}")
    sys.stdout.flush()

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"MODEL_READY:{MODEL_PATH}")
        return MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("DOWNLOAD_START:Downloading CLIP model (~85MB)...")
    
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress_hook)
        print(f"\nMODEL_READY:{MODEL_PATH}")
        return MODEL_PATH
    except Exception as e:
        print(f"\nDOWNLOAD_FAILED:{str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_model()
