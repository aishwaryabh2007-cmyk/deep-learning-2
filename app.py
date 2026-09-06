import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load Class Names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load Model
model = models.resnet18()

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "model/disease_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Disease Information
disease_info = {
    "Potato___Early_blight":
        "Apply fungicides and remove infected leaves.",

    "Potato___healthy":
        "Plant is healthy. No treatment required.",

    "Tomato___Early_blight":
        "Use fungicides and avoid overhead watering.",

    "Tomato___healthy":
        "Plant is healthy. No treatment required."
}

st.title("🌱 AI-Powered Plant Disease Detection")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    img = transform(image)
    img = img.unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probabilities = torch.nn.functional.softmax(
            outputs[0],
            dim=0
        )

        confidence, predicted = torch.max(
            probabilities,
            0
        )

    disease = class_names[predicted.item()]

    st.success(
        f"Detected Disease: {disease}"
    )

    st.write(
        f"Confidence: {confidence.item()*100:.2f}%"
    )

    st.subheader("Recommended Action")

    if disease in disease_info:
        st.write(disease_info[disease])
    else:
        st.write(
            "Consult an agricultural expert."
        )