# SANFA Cloud Engine (Brick 4)
# Deploying to Modal.com for Serverless GPU Processing

import modal
import os
import io
from PIL import Image

# 1. Define the Container Image
# We need PyTorch, CUDA, and OpenCLIP for Engine V5
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "open_clip_torch",
        "Pillow",
        "numpy",
        "python-dotenv"
    )
)

app = modal.App("sanfa-engine-v5", image=image)

# 2. Deployment Constants
# We'll use a standard T4 or A10G on Modal for production speed
@app.function(gpu="T4", timeout=300)
def protect_image_cloud(image_bytes: bytes, settings: dict = None):
    """
    The Cloud Gateway for SANFA Engine V5.
    Takes raw image bytes, runs the adversarial poison, and returns protected bytes.
    """
    import sys
    # Add engine directory to path (we'll mount the engine folder)
    sys.path.append("/root/engine")
    from engine import poison_image_raw # We'll need to expose a raw bytes version of the engine logic
    
    # Process
    protected_bytes = poison_image_raw(image_bytes, settings)
    
    return protected_bytes

# 3. Web API Wrapper
@app.function()
@modal.web_endpoint(method="POST")
def api_v1(image_data: bytes):
    """
    Public Endpoint for sanfa.dev frontend or desktop apps.
    """
    protected = protect_image_cloud.remote(image_data)
    return protected
