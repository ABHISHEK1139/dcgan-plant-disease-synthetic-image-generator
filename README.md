<div align="center">
  <h1>🌿 Crop Leaf Disease Generator</h1>
  <p><strong>A Deep Convolutional Generative Adversarial Network (DCGAN) for Synthetic Agricultural Image Synthesis</strong></p>
</div>

---

## 📖 About The Project

This project leverages the power of Generative Adversarial Networks (specifically **DCGANs**) to artificially synthesize high-quality, realistic images of crop leaves with various diseases. 

Training robust AI models for agricultural disease detection often requires vast amounts of data, which can be difficult to source. By synthesizing new, unique diseased leaf images, this project aims to provide a reliable method for data augmentation, effectively increasing dataset diversity and improving downstream classifier robustness.

The architecture involves a highly optimized Two-Stage Training Process (TTUR) designed specifically for 128x128 high-resolution output on consumer-grade GPUs.

---

## 🚀 Getting Started (How to Run)

Follow these steps to run the interactive generator application on your local machine.

### Prerequisites
- **Operating System:** Windows 10/11, Linux, or macOS
- **Python:** Version `3.10` or higher
- **Storage:** ~3 GB free space (to hold the model weights)
- **Hardware:** CPU is sufficient for generating images, though a GPU will make generation near-instantaneous.

### 1. Clone the Repository
Download the project to your local machine:
```bash
git clone https://github.com/ABHISHEK1139/dcgan-plant-disease-synthetic-image-generator.git
cd dcgan-plant-disease-synthetic-image-generator
```

*(Note: Ensure you have Git LFS installed if you are pulling the heavy `.pth` model weights).*

### 2. Launch the Application

#### Option A: One-Click Start (Windows)
If you are on Windows, simply double-click the **`start.bat`** file in the root directory. 
- It will automatically verify your Python installation.
- It will install all required dependencies behind the scenes.
- It will launch the Streamlit web interface in your default browser at `http://localhost:8501`.

#### Option B: Manual Start (All Operating Systems)
If you prefer using the terminal or are on macOS/Linux:
```bash
# 1. Install required packages
pip install -r requirements.txt

# 2. Run the Streamlit application
python -m streamlit run app.py
```

---

## 🖥️ How to Use the Generator

Once the application is open in your browser:
1. **Select a Plant:** Choose your target crop from the sidebar dropdown.
2. **Select a Condition:** Choose whether you want to generate a "healthy" leaf or a specific disease.
3. **Set Image Count:** Use the slider to pick how many unique synthetic images to generate at once (1 to 16).
4. **Generate:** Click the **"Generate Images"** button. The DCGAN will map random noise to a realistic leaf image in real-time.
5. **Download:** Hover over any generated image you like and click the save icon to download it to your computer.

---

## 📊 Available Plants & Diseases

The generator has been trained on a wide variety of crop-disease pairs. Here are the available models:

| Plant | Types Available |
|-------|----------------|
| 🍎 **Apple** | healthy, Apple_scab, Black_rot, Cedar_apple_rust |
| 🫐 **Blueberry** | healthy |
| 🍒 **Cherry** | healthy, Powdery_mildew |
| 🍇 **Grape** | healthy, Black_rot, Esca, Leaf_blight |
| 🍊 **Orange** | Haunglongbing |
| 🍑 **Peach** | Bacterial_spot |
| 🌶️ **Pepper** | healthy, Bacterial_spot |
| 🥔 **Potato** | healthy, Early_blight, Late_blight |
| 🫐 **Raspberry** | healthy |
| 🫘 **Soybean** | healthy |
| 🍓 **Strawberry** | healthy, Leaf_scorch |
| 🍅 **Tomato** | healthy, Tomato_mosaic_virus, Yellow_Leaf_Curl_Virus |

---

## ⚙️ Model Architecture & Parameters

The DCGAN models were heavily tuned to maintain stable training and generate high-fidelity 128x128 images. Below are the core hyperparameters used during training.

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
- **Generator Learning Rate:** 1e-5 *(Preserves healthy plant anatomy while adding lesions)*
- **Discriminator Learning Rate:** 2e-4 *(Quickly learns to identify specific disease patterns)*

### Stabilization Techniques Used
- **Label Smoothing:** Real=0.9, Fake=0.0
- **Label Flip Rate:** 5% (0.05)
- **Noise Injection:** Initial standard deviation of 0.1 with a 0.995 decay per epoch
- **Gradient Clipping:** 1.0

---

## 📁 Repository Structure

- `app.py` - The main Streamlit web application.
- `start.bat` - Windows auto-start script.
- `config.py` - Central configuration for paths and model hyperparameters.
- `src/` - Contains the core machine learning logic (DCGAN Generator/Discriminator architectures, dataloaders, and training loops).
- `checkpoints/` - Holds the trained `.pth` generator models (tracked via Git LFS).
