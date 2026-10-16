"""Streamlit web app for the DCGAN plant-disease image generator.

Run with:  streamlit run app.py   (or: python -m streamlit run app.py)

Pick a plant + disease, load the latest trained generator checkpoint, and
synthesize new leaf images.
"""

import streamlit as st
import torch

from config import LATENT_DIM, PLANT_DISEASE_MAP
from src.models import Generator
from src.utils import denormalize, find_latest_checkpoint


@st.cache_resource
def load_generator(checkpoint_path: str, device: str) -> Generator:
    """Load a generator from checkpoint (cached across reruns)."""
    from src.utils import load_generator_only

    generator = Generator().to(device)
    load_generator_only(checkpoint_path, generator, device)
    generator.eval()
    return generator


def generate_images(generator: Generator, n_samples: int, device: str):
    """Generate n_samples images; returns a list of numpy arrays."""
    images = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, LATENT_DIM, 1, 1, device=device)
            fake_img = generator(z)
            images.append(denormalize(fake_img[0]))
    return images


def main() -> None:
    st.set_page_config(page_title="Plant Disease DCGAN", layout="wide")
    st.title("🌱 Plant Disease Image Generator (DCGAN)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Running on: {device}")

    col1, col2 = st.columns(2)
    with col1:
        plant = st.selectbox("Plant", sorted(PLANT_DISEASE_MAP.keys()))
    with col2:
        disease = st.selectbox("Disease", PLANT_DISEASE_MAP[plant])
    n_samples = st.slider("Number of samples", 1, 16, 4)

    checkpoint = find_latest_checkpoint(plant, disease)
    if checkpoint is None:
        st.warning(
            f"No checkpoint found for {plant} / {disease}. "
            "Train a model first (see README)."
        )
        return
    st.info(f"Using checkpoint: `{checkpoint}`")

    if st.button("Generate", type="primary"):
        with st.spinner("Generating images..."):
            generator = load_generator(checkpoint, device)
            images = generate_images(generator, n_samples, device)
        cols = st.columns(min(n_samples, 4))
        for i, img in enumerate(images):
            cols[i % len(cols)].image(img, caption=f"Sample {i + 1}",
                                      use_container_width=True)


if __name__ == "__main__":
    main()
