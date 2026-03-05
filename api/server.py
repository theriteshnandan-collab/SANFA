"""
SANFA Cloud API
===============
FastAPI server that wraps the engine.py protection pipeline.
Designed to run on Modal.com (serverless GPU) or any cloud server.

Endpoints:
  POST /api/protect  — Upload image, get protected version back
  GET  /api/health   — Health check
"""
import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add engine directory to path
ENGINE_DIR = Path(__file__).parent.parent / "engine"
sys.path.insert(0, str(ENGINE_DIR))

app = FastAPI(
    title="SANFA Cloud API",
    description="Invisible AI protection for your artwork",
    version="1.0.0"
)

# CORS — allow the landing page to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down to your domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Simple in-memory rate limiting (replace with Redis in production)
usage_tracker: dict[str, list[float]] = {}
ip_tracker: dict[str, list[float]] = {}  # Anti-abuse: track by IP too
FREE_LIMIT = 5        # images per month
STARTER_LIMIT = 50
PRO_LIMIT = 200
TEAM_LIMIT = 500
IP_DAILY_LIMIT = 10   # Max images per IP per day (catches multi-account abuse)

def check_rate_limit(user_id: str, tier: str = "free", client_ip: str = "") -> bool:
    """Check if user has exceeded their monthly limit + IP anti-abuse."""
    now = time.time()
    month_ago = now - (30 * 24 * 3600)
    day_ago = now - (24 * 3600)
    
    # --- User-based monthly limit ---
    if user_id not in usage_tracker:
        usage_tracker[user_id] = []
    usage_tracker[user_id] = [t for t in usage_tracker[user_id] if t > month_ago]
    
    limits = {"free": FREE_LIMIT, "starter": STARTER_LIMIT, "pro": PRO_LIMIT, "team": TEAM_LIMIT}
    limit = limits.get(tier, FREE_LIMIT)
    
    if len(usage_tracker[user_id]) >= limit:
        return False
    
    # --- IP-based daily limit (catches multi-account abuse) ---
    if client_ip and tier == "free":
        if client_ip not in ip_tracker:
            ip_tracker[client_ip] = []
        ip_tracker[client_ip] = [t for t in ip_tracker[client_ip] if t > day_ago]
        if len(ip_tracker[client_ip]) >= IP_DAILY_LIMIT:
            return False
        ip_tracker[client_ip].append(now)
    
    usage_tracker[user_id].append(now)
    return True


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "v4.0.0", "gpu": "available"}


@app.post("/api/protect")
async def protect_image(
    request: Request,
    file: UploadFile,
    x_user_id: str = Header(default="anonymous"),
    x_user_tier: str = Header(default="free"),
):
    """
    Upload an image, get back the protected version + shield report.
    
    Headers:
      X-User-Id: User identifier (from Supabase auth)
      X-User-Tier: "free" | "starter" | "pro" | "team"
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted")
    
    # Check file size (max 20MB)
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum 20MB.")
    
    # Rate limit (user + IP)
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not check_rate_limit(x_user_id, x_user_tier, client_ip):
        raise HTTPException(
            429, 
            detail={
                "code": "RATE_LIMIT",
                "message": f"Monthly limit reached for {x_user_tier} tier",
                "upgrade_url": "https://sanfa.dev/pricing"
            }
        )
    
    # Create temp directory for processing
    work_dir = tempfile.mkdtemp(prefix="sanfa_")
    
    try:
        # Save uploaded file
        ext = Path(file.filename or "image.png").suffix or ".png"
        input_path = os.path.join(work_dir, f"input{ext}")
        output_path = os.path.join(work_dir, f"protected{ext}")
        
        with open(input_path, "wb") as f:
            f.write(content)
        
        # Run the engine
        from engine import poison_image
        poison_image(input_path, output_path)
        
        # Read the report
        report_path = output_path + ".report.json"
        report = {}
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
        
        # Return the protected image
        # The report is included as a custom header
        return FileResponse(
            output_path,
            media_type=f"image/{ext.lstrip('.')}",
            filename=f"sanfa_protected_{file.filename}",
            headers={
                "X-Shield-Report": json.dumps(report),
                "X-SANFA-Version": "4.0.0",
            }
        )
        
    except Exception as e:
        raise HTTPException(500, detail={"code": "ENGINE_ERROR", "message": str(e)})
    
    finally:
        # Cleanup temp files after response is sent
        # Note: FileResponse handles streaming, cleanup happens after
        pass


@app.post("/api/analyze")
async def analyze_image(file: UploadFile):
    """Quick analysis — returns what Auto-Armor would do, without processing."""
    from PIL import Image, ImageFilter
    import numpy as np
    import io
    
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    
    # Edge density analysis
    gray = img.convert('L')
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES))
    density = float(np.mean(edges))
    
    if density < 15:
        profile = "smooth"
        desc = "Subtle Shield — optimized for portraits and smooth art"
    elif density > 35:
        profile = "textured"
        desc = "Maximum Armor — noise hidden in heavy textures"
    else:
        profile = "balanced"
        desc = "Balanced Shield — standard protection"
    
    return {
        "profile": profile,
        "texture_density": round(density, 1),
        "description": desc,
        "image_size": f"{img.size[0]}x{img.size[1]}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
