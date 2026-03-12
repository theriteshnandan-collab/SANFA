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


import uuid
from pydantic import BaseModel

try:
    from redis import Redis
    from rq import Queue
    
    redis_conn = Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", None),
        ssl=os.getenv("REDIS_SSL", "False").lower() == "true",
    )
    # The 'sanfa_q' queue for GPU workers
    q = Queue('sanfa_q', connection=redis_conn)
except ImportError:
    q = None
    print("WARNING: Redis/RQ not installed. Install with `pip install redis rq`")


class JobStartRequest(BaseModel):
    image_url: str
    user_id: str = "anonymous"
    user_tier: str = "free"


@app.post("/api/job/start")
async def start_protection_job(
    request: JobStartRequest,
    client_request: Request
):
    """
    Decoupled Upload Architecture:
    The frontend has ALREADY uploaded the image directly to S3 / Supabase Storage using a Pre-Signed URL.
    It passes the public `image_url` here. We just queue the heavy GPU job and return a Job ID.
    """
    if not q:
        raise HTTPException(500, detail="Redis queue not configured on server")
        
    client_ip = client_request.client.host if client_request.client else "unknown"
    if not check_rate_limit(request.user_id, request.user_tier, client_ip):
        raise HTTPException(
            429, 
            detail={
                "code": "RATE_LIMIT",
                "message": f"Monthly limit reached for {request.user_tier} tier",
                "upgrade_url": "https://sanfa.dev/pricing"
            }
        )
    
    # Enqueue the background GPU process
    from worker import process_image_job
    job = q.enqueue(
        process_image_job,
        args=(str(uuid.uuid4()), request.image_url, request.user_tier),
        job_timeout=600  # 10 minute timeout
    )
    
    return {
        "status": "queued",
        "job_id": job.id,
        "message": "Image sent to GPU queue"
    }


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Poll this endpoint to get the status of the GPU processing."""
    if not q:
        raise HTTPException(500, detail="Redis queue not configured")
        
    job = q.fetch_job(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
        
    if job.is_finished:
        # The worker returns a dict dict on success: {"status": "success", "result_url": "...", "report": {...}}
        return {
            "job_id": job_id,
            "status": "completed",
            "result": job.result
        }
    elif job.is_failed:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": "The GPU worker failed to process this image."
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing",
            "position_in_queue": q.get_job_position(job_id)
        }


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
