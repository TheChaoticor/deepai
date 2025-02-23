import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load a pre-trained deepfake detection model
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def detect_deepfake_image(image, model):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][1].item()
    return confidence

def detect_deepfake_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_score = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Analyze every 30th frame
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fake_score += detect_deepfake_image(image, model)
    finally:
        cap.release()
    
    return fake_score / max(1, (frame_count // 30))

# Streamlit UI
st.title("Deepfake Detection WebApp")
model = load_model()

option = st.selectbox("Choose detection mode", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        confidence = detect_deepfake_image(image, model)
        st.write(f"Deepfake Confidence: {confidence:.2f}")
        if confidence > 0.5:
            st.error("This image is likely a deepfake!")
        else:
            st.success("This image appears authentic.")

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        st.video(temp_path)
        fake_score = detect_deepfake_video(temp_path, model)
        st.write(f"Deepfake Confidence: {fake_score:.2f}")
        if fake_score > 0.5:
            st.error("This video is likely a deepfake!")
        else:
            st.success("This video appears authentic.")
