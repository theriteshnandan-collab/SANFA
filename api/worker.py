"""
SANFA GPU Worker (Redis Queue)
==============================
This worker pulls jobs from the Redis queue, downloads the input image from S3,
runs the heavy GPU engine (CLIP/DCT/Nightshade), stamps C2PA metadata,
uploads the result back to S3, and updates the job status.

Run this via: rq worker sanfa_q
"""
import os
import time
import requests
import tempfile
from pathlib import Path

# In production this will be your Supabase/S3 credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "mock_url")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "mock_key")

# Import the engine
import sys
ENGINE_DIR = Path(__file__).parent.parent / "engine"
sys.path.insert(0, str(ENGINE_DIR))
from engine import poison_image

def process_image_job(job_id: str, image_url: str, user_tier: str):
    """
    Background job that runs on the GPU.
    1. Downloads image from `image_url`
    2. Runs PoisonPill engine
    3. Uploads protected image back to Cloud Storage
    4. Returns the result URL and execution report
    """
    print(f"[{job_id}] Started processing: {image_url}")
    
    work_dir = tempfile.mkdtemp(prefix=f"sanfa_{job_id}_")
    input_path = os.path.join(work_dir, "input.png")
    output_path = os.path.join(work_dir, "protected.png")
    
    try:
        # 1. Download the pre-signed uploaded image
        # In a real scenario, this would use requests.get(image_url) -> save to input_path
        # For mock/local testing, we assume it's a URL we can fetch
        print(f"[{job_id}] Downloading image...")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        with open(input_path, "wb") as f:
            f.write(response.content)
            
        # 2. Run the heavy GPU Protection
        print(f"[{job_id}] Running PoisonPill Engine...")
        poison_image(input_path, output_path)
        
        # 3. Read the report
        report_path = output_path + ".report.json"
        import json
        report = {}
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
                
        # 4. Upload the protected image back to S3 / Supabase Storage
        print(f"[{job_id}] Uploading protected image to cloud storage...")
        # (Mock upload logic)
        # In production: supabase.storage.from_("protected-images").upload(...)
        output_url = f"https://mock-s3-bucket.sanfa.dev/protected/{job_id}.png"
        
        print(f"[{job_id}] Finished successfully.")
        return {
            "status": "success",
            "result_url": output_url,
            "report": report
        }
        
    except Exception as e:
        print(f"[{job_id}] FAILED: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # Cleanup temp directory
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
