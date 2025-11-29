import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# -----------------------
# Charger le modèle YOLOv5
# -----------------------
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)
    return model

model = load_model()

st.title("🎓 Projet ENSA — Détection des Bouteilles")
st.subheader("Application de détection en temps réel utilisant YOLOv5 et Streamlit.")

# -----------------------
# Capture webcam
# -----------------------
run_live = st.checkbox("📷 Activer la webcam")

FRAME_WINDOW = st.image([])

if run_live:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("❌ Impossible d'accéder à la caméra")
            break

        # Conversion en RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -----------------------
        # Prédiction YOLOv5
        # -----------------------
        results = model(rgb)

        # Rendu dans l'image
        rendered = np.squeeze(results.render())

        FRAME_WINDOW.image(rendered)

    cap.release()
else:
    st.write("📌 Active la webcam pour commencer.")
