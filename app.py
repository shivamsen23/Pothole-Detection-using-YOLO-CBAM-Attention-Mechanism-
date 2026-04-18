import streamlit as st
from PIL import Image
from model_loader import load_model
from predictor import predict_image

# Page config (wide + no scroll feel)
st.set_page_config(page_title="Pothole Detection", layout="wide")

# Remove top padding & reduce spacing (IMPORTANT for single page feel)
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        img {
            max-height: 400px;
            object-fit: contain;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div.stButton > button {
    background: linear-gradient(135deg, #2c2c2c, #1a1a1a);
    color: #f5f5f5;
    font-size: 18px;
    font-weight: 600;
    padding: 12px 30px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
}

/* Hover effect */
div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #3a3a3a, #222222);
    box-shadow: 0px 6px 25px rgba(0,0,0,0.8);
    border: 1px solid rgba(255,255,255,0.2);
}

/* Click effect */
div.stButton > button:active {
    transform: scale(0.96);
    box-shadow: 0px 2px 10px rgba(0,0,0,0.7);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'> Pothole Detection System</h1>", unsafe_allow_html=True)

# Load model
model = load_model()

# Upload (centered)
_, center_col, _ = st.columns([1,2,1])
with center_col:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

st.markdown("---")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Layout: Left | Middle | Right
    col1, col_mid, col2 = st.columns([3,1,3])

    # LEFT SIDE
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=500)

        detect = st.button("Detect Pothole")

    # MIDDLE (Loader)
    with col_mid:
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        if detect:
            with st.spinner("In Processing..."):
                result_img, count = predict_image(model, image)

    # RIGHT SIDE
    with col2:
        st.subheader("Detection Result")

        if detect:
            st.image(result_img, width=500)
            st.success(f"Found {count} pothole(s)")
       