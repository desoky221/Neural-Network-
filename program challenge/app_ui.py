import io
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms, models

import streamlit as st


# ----------------------------
# Config
# ----------------------------
MODEL_PATH = Path("models/skin_cancer_cnn.pth")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    """Load the trained model architecture and weights."""
    base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(base_model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )

    model = base_model.to(DEVICE)

    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        st.warning(
            f"Model weights not found at {MODEL_PATH}. "
            "Train the model in the notebook and ensure the file is saved."
        )

    model.eval()
    return model


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict_image(model, image: Image.Image):
    """Run model prediction on a single PIL image."""
    tfm = get_transforms()
    tensor = tfm(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].item()

    return prob


def explain_prediction(prob: float) -> str:
    """Generate a simple natural language explanation for the prediction."""
    if prob >= 0.8:
        risk_desc = "high"
    elif prob >= 0.5:
        risk_desc = "moderate"
    elif prob >= 0.2:
        risk_desc = "low"
    else:
        risk_desc = "very low"

    explanation = (
        f"The model estimates a probability of about {prob:.1%} that this lesion is cancerous. "
        f"This corresponds to a **{risk_desc}** model-estimated cancer risk. "
        "This estimate is based purely on image patterns learned from the training dataset "
        "and does **not** consider other important clinical information such as patient history, "
        "age, or evolution over time.\n\n"
        "This tool is intended **only for educational and research purposes**. "
        "It must not be used as a substitute for an examination by a qualified dermatologist."
    )
    return explanation


def main():
    st.set_page_config(page_title="Skin Cancer Detection (Educational Demo)", layout="centered")

    st.title("Skin Cancer Detection")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a dermatoscopic image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Analyze image"):
            with st.spinner("Running model..."):
                prob = predict_image(model, image)
                label = "Cancer" if prob >= 0.5 else "No Cancer"

            st.subheader("Prediction")
            st.write(f"**Predicted class:** {label}")
            st.write(f"**Cancer probability:** {prob:.1%}")

            st.subheader("Explanation")
            st.markdown(explain_prediction(prob))

    st.markdown("---")
    st.caption(
        "Model: MobileNetV2 (transfer learning), trained in the accompanying Jupyter notebook. "
        "Dataset: HAM10000 (Kaggle)."
    )


if __name__ == '__main__':
    main()


