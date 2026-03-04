"""
PoisonPill Engine v4.0 — Multi-Layer Adversarial Protection
Three simultaneous attack vectors:
  1. PGD against CLIP (embedding confusion)
  2. DCT frequency poisoning (survives JPEG/social media)
  3. Nightshade data poisoning (teaches AI wrong concepts)

Usage: python engine.py <input_image> <output_image>
"""
import sys
import os
import json
import hashlib
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------- Config ----------
CLIP_EPSILON = 4.0 / 255.0    # CLIP PGD budget
CLIP_ITERATIONS = 20          # PGD steps
CLIP_STEP = 1.0 / 255.0       # Per-step size
DCT_STRENGTH = 0.08           # Frequency domain noise strength
CLIP_SIZE = 224

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ============================================================
# LAYER 1: CLIP PGD Attack (embedding confusion)
# ============================================================
def clip_pgd_attack(img, model, mean, std):
    """PGD attack against CLIP — shifts embedding away from original."""
    import torch
    
    img_resized = img.convert("RGB").resize((CLIP_SIZE, CLIP_SIZE), Image.BICUBIC)
    img_tensor = torch.from_numpy(
        np.array(img_resized, dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    
    img_normalized = (img_tensor - mean) / std
    with torch.no_grad():
        original_features = model.encode_image(img_normalized)
        original_features = original_features / original_features.norm(dim=-1, keepdim=True)
    
    delta = torch.zeros_like(img_tensor, requires_grad=True)
    
    for i in range(CLIP_ITERATIONS):
        perturbed = torch.clamp(img_tensor + delta, 0, 1)
        perturbed_normalized = (perturbed - mean) / std
        perturbed_features = model.encode_image(perturbed_normalized)
        perturbed_features = perturbed_features / perturbed_features.norm(dim=-1, keepdim=True)
        
        loss = torch.nn.functional.cosine_similarity(perturbed_features, original_features).mean()
        loss.backward()
        
        with torch.no_grad():
            delta.data = delta.data - CLIP_STEP * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -CLIP_EPSILON, CLIP_EPSILON)
            delta.data = torch.clamp(img_tensor + delta.data, 0, 1) - img_tensor
            delta.grad.zero_()
        
        print(f"PROGRESS:{int((i+1)/CLIP_ITERATIONS*33)}")
        sys.stdout.flush()
    
    # Get CLIP distance
    with torch.no_grad():
        final = torch.clamp(img_tensor + delta, 0, 1)
        final_norm = (final - mean) / std
        final_feat = model.encode_image(final_norm)
        final_feat = final_feat / final_feat.norm(dim=-1, keepdim=True)
        clip_dist = 1.0 - torch.nn.functional.cosine_similarity(final_feat, original_features).item()
    
    delta_pixels = delta.squeeze(0).permute(1, 2, 0).detach().numpy() * 255.0
    return delta_pixels, clip_dist

# ============================================================
# LAYER 2: DCT Frequency Poisoning (survives JPEG compression)
# ============================================================
def dct_frequency_poison(image_array):
    """
    Inject noise in mid-frequency DCT bands.
    This survives JPEG compression because JPEG preserves these frequencies.
    Humans can't see mid-frequency changes but AI models use them heavily.
    """
    from scipy.fft import dct, idct
    
    h, w, c = image_array.shape
    poisoned = image_array.copy().astype(np.float32)
    
    rng = np.random.RandomState(1337)
    
    for ch in range(c):
        channel = poisoned[:, :, ch]
        
        # Process in 8x8 blocks (same as JPEG)
        for y in range(0, h - 7, 8):
            for x in range(0, w - 7, 8):
                block = channel[y:y+8, x:x+8]
                
                # Forward DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Inject noise in mid-frequency coefficients (positions 2-5)
                # These survive JPEG quantization but are invisible to humans
                for i in range(2, 6):
                    for j in range(2, 6):
                        dct_block[i, j] += rng.uniform(-DCT_STRENGTH, DCT_STRENGTH) * abs(dct_block[i, j] + 1)
                
                # Inverse DCT
                channel[y:y+8, x:x+8] = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        
        poisoned[:, :, ch] = channel
    
    return np.clip(poisoned, 0, 255).astype(np.float32)

# ============================================================
# LAYER 3: Nightshade Data Poisoning (wrong concept injection)
# ============================================================
def nightshade_poison(img, model, mean, std):
    """
    Push image embedding TOWARD a completely wrong concept.
    If the image is a person, push toward 'abstract painting'.
    If it's a landscape, push toward 'circuit board'.
    This makes AI learn WRONG associations.
    """
    import torch
    import open_clip
    
    # Get text embeddings for wrong concepts
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    wrong_concepts = [
        "a photo of static noise and glitch artifacts",
        "an abstract pattern of random colored squares",
        "a blank concrete wall with no features",
    ]
    
    with torch.no_grad():
        text_tokens = tokenizer(wrong_concepts)
        target_features = model.encode_text(text_tokens)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)
        # Average the wrong concept embeddings
        target_embedding = target_features.mean(dim=0, keepdim=True)
        target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    
    # PGD attack pushing TOWARD the wrong concept
    img_resized = img.convert("RGB").resize((CLIP_SIZE, CLIP_SIZE), Image.BICUBIC)
    img_tensor = torch.from_numpy(
        np.array(img_resized, dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    
    delta = torch.zeros_like(img_tensor, requires_grad=True)
    ns_epsilon = 3.0 / 255.0
    ns_step = 0.5 / 255.0
    
    for i in range(15):
        perturbed = torch.clamp(img_tensor + delta, 0, 1)
        perturbed_normalized = (perturbed - mean) / std
        perturbed_features = model.encode_image(perturbed_normalized)
        perturbed_features = perturbed_features / perturbed_features.norm(dim=-1, keepdim=True)
        
        # MAXIMIZE similarity to wrong concept (minimize negative)
        loss = -torch.nn.functional.cosine_similarity(perturbed_features, target_embedding).mean()
        loss.backward()
        
        with torch.no_grad():
            delta.data = delta.data - ns_step * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -ns_epsilon, ns_epsilon)
            delta.data = torch.clamp(img_tensor + delta.data, 0, 1) - img_tensor
            delta.grad.zero_()
        
        print(f"PROGRESS:{33 + int((i+1)/15*33)}")
        sys.stdout.flush()
    
    delta_pixels = delta.squeeze(0).permute(1, 2, 0).detach().numpy() * 255.0
    return delta_pixels

# ============================================================
# PERCEPTUAL MASK: Hide noise in textured areas
# ============================================================
def compute_perceptual_mask(img):
    """Sobel edge detection mask — full noise on textures, minimal on smooth areas."""
    gray = img.convert('L')
    edges_x = np.array(gray.filter(ImageFilter.Kernel(
        (3,3), [-1,0,1,-2,0,2,-1,0,1], scale=1, offset=128
    )), dtype=np.float32) - 128
    edges_y = np.array(gray.filter(ImageFilter.Kernel(
        (3,3), [-1,-2,-1,0,0,0,1,2,1], scale=1, offset=128
    )), dtype=np.float32) - 128
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    mask = edge_mag / max(edge_mag.max(), 1)
    mask = np.clip(mask * 3.0, 0.15, 1.0)
    return np.stack([mask, mask, mask], axis=2)

# ============================================================
# MAIN PIPELINE
# ============================================================
def poison_image(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            print(f"ERROR:File not found - {input_path}", file=sys.stderr)
            sys.exit(1)

        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        original_hash = sha256_file(input_path)
        clip_distance = 0.0
        attack_layers = []
        
        # Check for PyTorch
        use_pytorch = False
        try:
            import torch
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu'
            )
            model.eval()
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
            use_pytorch = True
            print("ENGINE:PyTorch + CLIP loaded. Multi-layer attack mode.")
        except ImportError:
            print("ENGINE:PyTorch not available. Using fallback.")
        
        sys.stdout.flush()
        orig_array = np.array(img, dtype=np.float32)
        combined_noise = np.zeros_like(orig_array)
        
        if use_pytorch:
            # === LAYER 1: CLIP PGD ===
            print("ENGINE:Layer 1/3 — CLIP PGD adversarial attack...")
            sys.stdout.flush()
            clip_delta_224, clip_distance = clip_pgd_attack(img, model, mean, std)
            
            # Upscale CLIP delta to full resolution
            clip_noise = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                ch = Image.fromarray(
                    np.clip(clip_delta_224[:,:,c] + 128, 0, 255).astype(np.uint8), mode='L'
                ).resize((w, h), Image.BICUBIC)
                clip_noise[:,:,c] = np.array(ch, dtype=np.float32) - 128.0
            combined_noise += clip_noise
            attack_layers.append("CLIP_PGD")
            
            # === LAYER 2: DCT Frequency Poisoning ===
            print("ENGINE:Layer 2/3 — DCT frequency poisoning...")
            sys.stdout.flush()
            try:
                dct_result = dct_frequency_poison(orig_array)
                dct_noise = dct_result - orig_array
                combined_noise += dct_noise
                attack_layers.append("DCT_FREQUENCY")
                print("PROGRESS:70")
            except ImportError:
                print("ENGINE:scipy not available, skipping DCT layer.")
            sys.stdout.flush()
            
            # === LAYER 3: Nightshade Data Poisoning ===
            print("ENGINE:Layer 3/3 — Nightshade concept poisoning...")
            sys.stdout.flush()
            nightshade_delta_224 = nightshade_poison(img, model, mean, std)
            
            # Upscale Nightshade delta
            ns_noise = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                ch = Image.fromarray(
                    np.clip(nightshade_delta_224[:,:,c] + 128, 0, 255).astype(np.uint8), mode='L'
                ).resize((w, h), Image.BICUBIC)
                ns_noise[:,:,c] = np.array(ch, dtype=np.float32) - 128.0
            combined_noise += ns_noise
            attack_layers.append("NIGHTSHADE")
        else:
            # Fallback
            import random
            pixels = img.load()
            for x in range(w):
                for y in range(h):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (
                        max(0, min(255, r + random.randint(-8, 8))),
                        max(0, min(255, g + random.randint(-8, 8))),
                        max(0, min(255, b + random.randint(-8, 8)))
                    )
            attack_layers.append("FALLBACK")
        
        if use_pytorch:
            # Apply perceptual mask
            mask = compute_perceptual_mask(img)
            combined_noise = combined_noise * mask
            
            # Clamp total noise to ±6 per pixel (still invisible)
            combined_noise = np.clip(combined_noise, -6.0, 6.0)
            
            # Apply to image
            protected_array = np.clip(orig_array + combined_noise, 0, 255).astype(np.uint8)
            protected_img = Image.fromarray(protected_array)
            
            # Heatmap
            heatmap = np.abs(combined_noise).sum(axis=2)
            heatmap = np.clip(heatmap / max(heatmap.max(), 1) * 255, 0, 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap, mode='L')
        else:
            protected_img = img
            heatmap_img = None
        
        print("PROGRESS:95")
        sys.stdout.flush()
        
        # Save with C2PA metadata
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".png":
            from PIL.PngImagePlugin import PngInfo
            metadata = PngInfo()
            metadata.add_text("Copyright", "AI-Training-Opted-Out via PoisonPill")
            metadata.add_text("C2PA:Assertion", "c2pa.training-mining=notAllowed")
            metadata.add_text("PoisonPill:Version", "4.0.0")
            metadata.add_text("PoisonPill:Layers", "+".join(attack_layers))
            protected_img.save(output_path, pnginfo=metadata)
        else:
            from PIL.ExifTags import Base
            exif_dict = protected_img.getexif()
            exif_dict[Base.Copyright] = "AI-Training-Opted-Out via PoisonPill | C2PA:training-mining=notAllowed"
            protected_img.save(output_path, quality=95, exif=exif_dict.tobytes())
        
        if heatmap_img:
            heatmap_img.save(output_path + ".heatmap.png")
        
        # Stats
        protected_hash = sha256_file(output_path)
        prot_arr = np.array(protected_img.resize(img.size))
        modified = int(np.sum(np.any(np.array(img) != prot_arr, axis=2)))
        pix_pct = (modified / (w * h)) * 100
        
        report = {
            "status": "PROTECTED",
            "engine_version": "4.0.0",
            "attack_layers": attack_layers,
            "clip_distance": round(clip_distance, 4),
            "pixels_modified_pct": round(pix_pct, 1),
            "image_size": f"{w}x{h}",
            "original_hash": f"sha256:{original_hash[:16]}",
            "protected_hash": f"sha256:{protected_hash[:16]}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        with open(output_path + ".report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"REPORT:{json.dumps(report)}")
        print(f"SUCCESS:{output_path}")
        
    except Exception as e:
        print(f"FAILED:{str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python engine.py <input_image> <output_image>", file=sys.stderr)
        sys.exit(1)
    poison_image(sys.argv[1], sys.argv[2])
