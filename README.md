# 🛡️ PoisonPill — Anti-AI Image Shield

**Protect your art from AI training.** PoisonPill applies invisible adversarial perturbations to your images that confuse AI models like CLIP, Stable Diffusion, and Midjourney — while keeping your images looking perfect to human eyes.

## How It Works

1. **Select your image** → Click "Select & Protect" in the app
2. **PGD attack runs** → 30 iterations of Projected Gradient Descent against CLIP ViT-B/32
3. **Perceptual masking** → Noise hidden in textures, minimized on smooth areas (skin, sky)
4. **C2PA metadata** → `AI-Training-Opted-Out` watermark embedded in file metadata
5. **Shield Report** → CLIP distance score proves AI sees the image differently

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop App | [Tauri v2](https://tauri.app/) (Rust) |
| Frontend | Next.js 15, React, TypeScript |
| AI Engine | PyTorch + OpenCLIP ViT-B/32 |
| Attack | PGD (Projected Gradient Descent) with perceptual masking |

## Setup

```bash
# Install dependencies
npm install
pip install -r engine/requirements.txt

# Download CLIP model (first time only)
python engine/download_model.py

# Run the app
npx tauri dev
```

## Engine (Standalone)

```bash
python engine/engine.py input.png output.png
```

## How Protection Works

The engine uses **real gradient backpropagation** through CLIP ViT-B/32 to compute mathematically optimal pixel perturbations. This is the same technique used by [Glaze](https://glaze.cs.uchicago.edu/) (University of Chicago).

- **Epsilon:** ±4 pixels per channel (invisible to humans)
- **Perceptual mask:** Sobel edge detection concentrates noise on textured areas
- **CLIP distance:** ~18% embedding shift — AI perceives the image differently
- **C2PA metadata:** Industry-standard `training-mining=notAllowed` assertion

## Project Structure

```
sanfa/
├── engine/
│   ├── engine.py          # PyTorch PGD adversarial engine
│   ├── download_model.py  # CLIP model downloader
│   ├── verify.py          # Pixel comparison tool
│   └── requirements.txt
├── src/
│   └── app/page.tsx       # Premium React dashboard
├── src-tauri/
│   └── src/lib.rs         # Rust IPC + file watcher
└── package.json
```

## License

MIT — Built to protect artists. 🎨
