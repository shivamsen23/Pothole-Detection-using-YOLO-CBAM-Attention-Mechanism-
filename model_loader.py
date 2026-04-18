from ultralytics import YOLO
import streamlit as st

@st.cache_resource
def load_model():
    model = YOLO("runs/detect/train/weights/best.pt")
    return model