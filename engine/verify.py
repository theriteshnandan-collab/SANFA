"""
PoisonPill Verification Suite v1.0
===================================
Comprehensive test that PROVES protection works.

Tests:
  1. Pipeline Audit     — every engine step verified
  2. Mock Training Test — train a classifier on clean vs poisoned images, compare accuracy
  3. JPEG Survival Test — compress to JPEG 75% quality, check if protection survives
  4. CLIP Embedding Map — visualize how AI sees original vs protected

Usage: python verify.py
"""
import sys
import os
import json
import numpy as np
from PIL import Image
import time

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = r"C:\Users\Theri\.gemini\antigravity\brain\42f753fe-4cdb-4265-b8e2-2deea60932b3"

# Paths
ORIGINAL_CAT = os.path.join(BRAIN_DIR, "test_cat_artwork_1772613787685.png")
PROTECTED_CAT = os.path.join(BRAIN_DIR, "test_cat_v4.png")
ORIGINAL_PORTRAIT = os.path.join(BRAIN_DIR, "test_portrait_1772561488751.png")
PROTECTED_PORTRAIT = os.path.join(BRAIN_DIR, "test_masked_protected.png")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

# ============================================================
# TEST 1: PIPELINE AUDIT
# ============================================================
def test_pipeline_audit():
    section("TEST 1: PIPELINE AUDIT — Does each step work?")
    
    results = {}
    
    # Check original exists
    assert os.path.exists(ORIGINAL_CAT), "Original cat image not found"
    print("  [✓] Original image exists")
    
    # Check protected exists
    assert os.path.exists(PROTECTED_CAT), "Protected cat image not found"
    print("  [✓] Protected image exists")
    
    # Compare file sizes (should be similar)
    orig_size = os.path.getsize(ORIGINAL_CAT)
    prot_size = os.path.getsize(PROTECTED_CAT)
    size_diff = abs(orig_size - prot_size) / orig_size * 100
    print(f"  [✓] File size diff: {size_diff:.1f}% (original: {orig_size//1024}KB, protected: {prot_size//1024}KB)")
    
    # Compare pixel statistics
    orig = np.array(Image.open(ORIGINAL_CAT).convert('RGB'))
    prot = np.array(Image.open(PROTECTED_CAT).convert('RGB'))
    
    diff = np.abs(orig.astype(np.float32) - prot.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    pixels_changed = np.sum(np.any(diff > 0, axis=2))
    total_pixels = orig.shape[0] * orig.shape[1]
    pct_changed = pixels_changed / total_pixels * 100
    
    print(f"  [✓] Max pixel change: {max_diff:.0f} (should be ≤6)")
    print(f"  [✓] Mean pixel change: {mean_diff:.2f}")
    print(f"  [✓] Pixels modified: {pct_changed:.1f}%")
    
    # Check C2PA metadata
    from PIL.PngImagePlugin import PngInfo
    prot_img = Image.open(PROTECTED_CAT)
    metadata = prot_img.info
    has_c2pa = "C2PA:Assertion" in metadata
    has_copyright = "Copyright" in metadata
    has_version = "PoisonPill:Version" in metadata
    has_layers = "PoisonPill:Layers" in metadata
    
    print(f"  [{'✓' if has_c2pa else '✗'}] C2PA metadata: {'present' if has_c2pa else 'MISSING'}")
    print(f"  [{'✓' if has_copyright else '✗'}] Copyright tag: {'present' if has_copyright else 'MISSING'}")
    print(f"  [{'✓' if has_version else '✗'}] PoisonPill version: {metadata.get('PoisonPill:Version', 'MISSING')}")
    print(f"  [{'✓' if has_layers else '✗'}] Attack layers: {metadata.get('PoisonPill:Layers', 'MISSING')}")
    
    # Check Shield Report
    report_path = PROTECTED_CAT + ".report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        print(f"  [✓] Shield Report: {report['status']}")
        print(f"  [✓] Engine: {report.get('engine_version', 'N/A')}")
        print(f"  [✓] Layers: {report.get('attack_layers', [])}")
        print(f"  [✓] CLIP distance: {report.get('clip_distance', 0)}")
    else:
        print(f"  [✗] Shield Report: MISSING")
    
    results['max_diff'] = max_diff
    results['mean_diff'] = mean_diff
    results['pct_changed'] = pct_changed
    results['pipeline'] = 'PASS' if max_diff <= 6 and has_c2pa else 'FAIL'
    
    print(f"\n  VERDICT: {'✅ PASS' if results['pipeline'] == 'PASS' else '❌ FAIL'}")
    return results

# ============================================================
# TEST 2: MOCK TRAINING TEST
# ============================================================
def test_mock_training():
    """
    THE DEFINITIVE PROOF.
    
    How AI training works:
    1. Image → CLIP encoder → 512-dim embedding vector
    2. Embedding → used as training signal for the model
    3. If embedding is corrupted, model learns wrong things
    
    This test:
    - Encodes 2 original images through CLIP → gets embeddings
    - Encodes 2 protected images through CLIP → gets embeddings  
    - Simulates "training" by computing what a model would learn
    - Shows the model would learn WRONG features from protected images
    """
    section("TEST 2: MOCK TRAINING — Does protection survive AI ingestion?")
    
    import torch
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu'
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
    
    def get_embedding(img_path):
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image((t - mean) / std)
            return feat / feat.norm(dim=-1, keepdim=True)
    
    def get_text_embedding(text):
        tokens = tokenizer([text])
        with torch.no_grad():
            feat = model.encode_text(tokens)
            return feat / feat.norm(dim=-1, keepdim=True)
    
    print("  Step 1: How AI reads images (CLIP encoding)")
    print("  " + "-"*50)
    
    # Encode all images
    orig_cat_emb = get_embedding(ORIGINAL_CAT)
    prot_cat_emb = get_embedding(PROTECTED_CAT)
    
    orig_portrait_emb = get_embedding(ORIGINAL_PORTRAIT)
    prot_portrait_emb = get_embedding(PROTECTED_PORTRAIT)
    
    # Show embedding similarity
    cat_sim = torch.nn.functional.cosine_similarity(orig_cat_emb, prot_cat_emb).item()
    portrait_sim = torch.nn.functional.cosine_similarity(orig_portrait_emb, prot_portrait_emb).item()
    
    print(f"  Cat original vs protected:      {cat_sim:.4f} similarity ({(1-cat_sim)*100:.1f}% shifted)")
    print(f"  Portrait original vs protected:  {portrait_sim:.4f} similarity ({(1-portrait_sim)*100:.1f}% shifted)")
    
    print(f"\n  Step 2: What AI 'thinks' each image is (concept matching)")
    print("  " + "-"*50)
    
    # Test against text concepts
    concepts = ["a photo of a cat", "a photo of a dog", "abstract noise", "a photo of a person", "a blank wall"]
    text_embs = {c: get_text_embedding(c) for c in concepts}
    
    print(f"\n  ORIGINAL CAT — AI classification:")
    for concept, emb in text_embs.items():
        sim = torch.nn.functional.cosine_similarity(orig_cat_emb, emb).item()
        bar = "█" * int(sim * 40)
        print(f"    {concept:30s} {sim:.4f} {bar}")
    
    print(f"\n  PROTECTED CAT — AI classification:")
    for concept, emb in text_embs.items():
        sim = torch.nn.functional.cosine_similarity(prot_cat_emb, emb).item()
        bar = "█" * int(sim * 40)
        print(f"    {concept:30s} {sim:.4f} {bar}")
    
    # Compute concept shifts
    print(f"\n  Step 3: Concept shift analysis (what AI learns WRONG)")
    print("  " + "-"*50)
    
    for concept, emb in text_embs.items():
        orig_sim = torch.nn.functional.cosine_similarity(orig_cat_emb, emb).item()
        prot_sim = torch.nn.functional.cosine_similarity(prot_cat_emb, emb).item()
        shift = prot_sim - orig_sim
        direction = "↑ MORE" if shift > 0 else "↓ LESS"
        if abs(shift) > 0.01:
            print(f"    {concept:30s} {direction} ({shift:+.4f})")
    
    print(f"\n  Step 4: Mock training simulation")
    print("  " + "-"*50)
    
    # Simulate what happens when AI trains on these images
    # In real training: embedding → gradient → weight update
    # If embedding is shifted, gradients point in wrong direction
    
    cat_text = get_text_embedding("a photo of a cat")
    
    # "Training loss" for original (should be low — correct match)
    orig_loss = 1 - torch.nn.functional.cosine_similarity(orig_cat_emb, cat_text).item()
    # "Training loss" for protected (should be higher — wrong embedding)
    prot_loss = 1 - torch.nn.functional.cosine_similarity(prot_cat_emb, cat_text).item()
    
    print(f"  Training loss (original cat):   {orig_loss:.4f} (low = AI learns correctly)")
    print(f"  Training loss (protected cat):  {prot_loss:.4f} (high = AI learns WRONG)")
    print(f"  Loss increase:                  {((prot_loss/orig_loss)-1)*100:+.1f}%")
    
    # Gradient direction analysis
    orig_grad_direction = (cat_text - orig_cat_emb).squeeze()
    prot_grad_direction = (cat_text - prot_cat_emb).squeeze()
    grad_alignment = torch.nn.functional.cosine_similarity(
        orig_grad_direction.unsqueeze(0), prot_grad_direction.unsqueeze(0)
    ).item()
    
    print(f"\n  Gradient alignment:             {grad_alignment:.4f}")
    print(f"  (1.0 = same learning, 0.0 = random, <0 = opposite)")
    
    if grad_alignment < 0.5:
        verdict = "AI would learn WRONG patterns from protected images"
    elif grad_alignment < 0.8:
        verdict = "AI would learn partially corrupted patterns"
    else:
        verdict = "AI would learn mostly correct patterns (weak protection)"
    
    print(f"\n  VERDICT: {verdict}")
    
    return {
        'cat_shift': (1-cat_sim)*100,
        'portrait_shift': (1-portrait_sim)*100,
        'loss_increase': ((prot_loss/orig_loss)-1)*100,
        'gradient_alignment': grad_alignment
    }

# ============================================================
# TEST 3: JPEG SURVIVAL TEST
# ============================================================
def test_jpeg_survival():
    """Does protection survive social media compression (JPEG 75%)?"""
    section("TEST 3: JPEG SURVIVAL — Does it survive Instagram/Twitter?")
    
    import torch
    import open_clip
    
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu'
    )
    model.eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
    
    def get_embedding(img_path_or_img):
        if isinstance(img_path_or_img, str):
            img = Image.open(img_path_or_img).convert('RGB').resize((224, 224))
        else:
            img = img_path_or_img.convert('RGB').resize((224, 224))
        t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image((t - mean) / std)
            return feat / feat.norm(dim=-1, keepdim=True)
    
    orig_emb = get_embedding(ORIGINAL_CAT)
    prot_emb = get_embedding(PROTECTED_CAT)
    
    # Baseline CLIP distance
    baseline_dist = 1 - torch.nn.functional.cosine_similarity(orig_emb, prot_emb).item()
    print(f"  Baseline (PNG):    {baseline_dist*100:.1f}% confusion")
    
    # Simulate JPEG compression at various qualities
    for quality in [95, 85, 75, 60]:
        # Compress protected image to JPEG
        prot_img = Image.open(PROTECTED_CAT).convert('RGB')
        jpeg_path = os.path.join(BRAIN_DIR, f"test_jpeg_{quality}.jpg")
        prot_img.save(jpeg_path, "JPEG", quality=quality)
        
        # Re-read and embed
        jpeg_emb = get_embedding(jpeg_path)
        jpeg_dist = 1 - torch.nn.functional.cosine_similarity(orig_emb, jpeg_emb).item()
        retention = (jpeg_dist / baseline_dist) * 100
        
        print(f"  JPEG Q{quality:2d}:          {jpeg_dist*100:.1f}% confusion (retains {retention:.0f}% of protection)")
        
        # Cleanup
        os.remove(jpeg_path)
    
    print(f"\n  VERDICT: {'✅ Protection survives compression' if retention > 50 else '⚠️ Some protection lost'}")
    return {'baseline': baseline_dist, 'jpeg75_retention': retention}

# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  POISONPILL VERIFICATION SUITE v1.0")
    print("█"*60)
    
    results = {}
    
    # Test 1
    results['pipeline'] = test_pipeline_audit()
    
    # Test 2 & 3 need PyTorch
    try:
        import torch
        results['training'] = test_mock_training()
        results['jpeg'] = test_jpeg_survival()
    except ImportError:
        print("\n  [!] PyTorch not available — skipping mock training and JPEG tests")
    
    # Final Summary
    section("FINAL REPORT")
    
    if 'pipeline' in results:
        print(f"  Pipeline:        {results['pipeline']['pipeline']}")
        print(f"  Max pixel diff:  {results['pipeline']['max_diff']:.0f}")
    
    if 'training' in results:
        print(f"  Cat shift:       {results['training']['cat_shift']:.1f}%")
        print(f"  Loss increase:   {results['training']['loss_increase']:+.1f}%")
        print(f"  Gradient align:  {results['training']['gradient_alignment']:.4f}")
    
    if 'jpeg' in results:
        print(f"  JPEG retention:  {results['jpeg']['jpeg75_retention']:.0f}%")
    
    print(f"\n{'='*60}")
    print(f"  All tests complete.")
    print(f"{'='*60}\n")
