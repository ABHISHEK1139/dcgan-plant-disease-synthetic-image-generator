# 🌿 Crop Leaf Disease Generator

Generate synthetic crop leaf disease images using trained DCGAN models.

## Quick Start (Windows)

1. **Make sure Python 3.10+ is installed**

2. **Double-click `start.bat`** - it will:
   - Check for Python
   - Install dependencies if needed
   - Start the app at http://localhost:8501

That's it! The app will open in your browser.

## Manual Start

If start.bat doesn't work:
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Available Plants & Diseases

| Plant | Types Available |
|-------|----------------|
| 🍎 Apple | healthy, Apple_scab, Black_rot, Cedar_apple_rust |
| 🫐 Blueberry | healthy |
| 🍒 Cherry | healthy, Powdery_mildew |
| 🍇 Grape | healthy, Black_rot, Esca, Leaf_blight |
| 🍊 Orange | Haunglongbing |
| 🍑 Peach | Bacterial_spot |
| 🌶️ Pepper | healthy, Bacterial_spot |
| 🥔 Potato | healthy, Early_blight, Late_blight |
| 🫐 Raspberry | healthy |
| 🫘 Soybean | healthy |
| 🍓 Strawberry | healthy, Leaf_scorch |
| 🍅 Tomato | healthy, Tomato_mosaic_virus, Yellow_Leaf_Curl_Virus |

## How It Works

1. Select a plant from the dropdown
2. Select disease/type
3. Choose number of images (1-16)
4. Click "Generate Images"
5. Download images you like!

## Requirements

- Windows 10/11
- Python 3.10+
- ~2GB disk space
- GPU optional (faster generation)

## Model Parameters

The DCGAN models are trained with the following hyper-parameters optimized for quality and stable training (e.g. on an RTX 3050 4GB or similar GPUs):

### Image & Latent Space
- **Image Size:** 128x128 pixels (3 Channels - RGB)
- **Latent Dimension:** 100

### Training (Stage 1: Anatomy)
- **Epochs:** 200
- **Learning Rate:** 0.0002
- **Batch Size:** 64
- **Optimizer:** Adam (Beta1 = 0.5, Beta2 = 0.999)

### Fine-Tuning (Stage 2: Disease Patterns using TTUR)
- **Epochs:** 200
- **Generator Learning Rate:** 1e-5 (Preserves plant anatomy)
- **Discriminator Learning Rate:** 2e-4 (Quickly learns disease patterns)

### Stabilization Techniques
- **Label Smoothing (Real):** 0.9
- **Label Smoothing (Fake):** 0.0
- **Label Flip Rate:** 5% (0.05)
- **Noise Injection:** Initial standard deviation of 0.1 with 0.995 decay per epoch
- **Gradient Clipping:** 1.0
