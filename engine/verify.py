"""
PoisonPill Verification Tool
Compares original image vs shielded image to prove noise was injected.
Usage: python verify.py <original_image> <shielded_image>
"""
import sys
from PIL import Image

def verify(original_path, shielded_path):
    orig = Image.open(original_path).convert("RGB")
    shield = Image.open(shielded_path).convert("RGB")
    
    orig_pixels = orig.load()
    shield_pixels = shield.load()
    w, h = orig.size
    
    total_pixels = w * h
    changed_pixels = 0
    total_diff = 0
    
    for x in range(w):
        for y in range(h):
            r1, g1, b1 = orig_pixels[x, y]
            r2, g2, b2 = shield_pixels[x, y]
            diff = abs(r1-r2) + abs(g1-g2) + abs(b1-b2)
            if diff > 0:
                changed_pixels += 1
                total_diff += diff
    
    pct = (changed_pixels / total_pixels) * 100
    avg_diff = total_diff / max(changed_pixels, 1)
    
    print("=" * 50)
    print("  POISONPILL VERIFICATION REPORT")
    print("=" * 50)
    print(f"  Image Size:        {w} x {h}")
    print(f"  Total Pixels:      {total_pixels:,}")
    print(f"  Modified Pixels:   {changed_pixels:,}")
    print(f"  Coverage:          {pct:.1f}%")
    print(f"  Avg Noise/Pixel:   {avg_diff:.2f}")
    print("=" * 50)
    
    if pct > 50:
        print("  STATUS: PROTECTED - Anti-AI noise confirmed!")
    else:
        print("  STATUS: WARNING - Low noise coverage")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify.py <original_image> <shielded_image>")
        sys.exit(1)
    verify(sys.argv[1], sys.argv[2])
