import streamlit as st
from PIL import Image
from model_loader import load_model
from predictor import predict_image

# New imports for video/live-camera support
import cv2
import tempfile
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase



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
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

st.markdown("---")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Layout: Left | Middle | Right
    col1, col_mid, col2 = st.columns([3, 1, 3])

    # LEFT SIDE
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=500)

        detect = st.button("Detect Pothole", key="detect_image")

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


# ============================================================
# NEW FEATURES — VIDEO FILE + LIVE CAMERA
# Existing image UI above is not changed.
# ============================================================

st.markdown("---")
st.markdown("## Video & Live Detection")
st.caption(
    "The same trained YOLOv8 + CBAM model is used for every video frame. "
    "No separate model is required."
)


# ------------------------------------------------------------
# Helper: process an uploaded video
# ------------------------------------------------------------
def process_video(input_path, output_path, model, confidence=0.5):
    """Process a video frame-by-frame with the trained model.

    Returns:
        total_frame_detections: total number of detections across all frames.
        max_detections_in_frame: maximum number of potholes detected in one frame.
        processed_frames: number of processed frames.
        fps: source video FPS.
    """

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions.")

    # mp4v is widely available with OpenCV for local testing.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    total_frame_detections = 0
    max_detections_in_frame = 0
    processed_frames = 0

    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Run the SAME trained YOLOv8 + CBAM model on this frame.
        results = model(frame, conf=confidence, verbose=False)

        result = results[0]

        # Number of detections in this frame.
        frame_count = len(result.boxes) if result.boxes is not None else 0

        total_frame_detections += frame_count
        max_detections_in_frame = max(max_detections_in_frame, frame_count)

        # Draw bounding boxes, labels and confidence scores.
        annotated_frame = result.plot()

        writer.write(annotated_frame)
        processed_frames += 1

        if total_frames > 0:
            progress.progress(min(processed_frames / total_frames, 1.0))

    cap.release()
    writer.release()
    progress.empty()

    return (
        total_frame_detections,
        max_detections_in_frame,
        processed_frames,
        fps,
    )


class PotholeVideoProcessor(VideoProcessorBase):
    """Receives webcam frames and runs the trained detector."""

    def __init__(self, model, confidence=0.5):
        self.model = model
        self.confidence = confidence

    def recv(self, frame):
        # WebRTC frame -> OpenCV BGR image
        img = frame.to_ndarray(format="bgr24")

        # Same trained YOLOv8 + CBAM model
        results = self.model(
            img,
            conf=self.confidence,
            verbose=False
        )

        # Draw detections
        annotated = results[0].plot()

        # OpenCV BGR image -> WebRTC frame
        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ============================================================
# VIDEO FILE + LIVE CAMERA TABS
# ============================================================

video_tab, live_tab = st.tabs(["Upload & Process Video", "Live Camera"])



with video_tab:
    st.subheader("Upload Recorded Road Video")

    video_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="video_confidence"
    )

    if video_file is not None:
        st.video(video_file)

        process_button = st.button(
            "Process Video",
            key="process_video"
        )

        if process_button:
            input_path = None
            output_path = None

            try:
                # Save uploaded video temporarily.
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(video_file.name)[1]
                ) as input_temp:
                    input_temp.write(video_file.getbuffer())
                    input_path = input_temp.name

                # Output file.
                output_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix="_pothole_result.mp4"
                )
                output_path = output_temp.name
                output_temp.close()

                with st.spinner("Processing video frame-by-frame..."):
                    (
                        total_detections,
                        max_in_frame,
                        processed_frames,
                        fps,
                    ) = process_video(
                        input_path,
                        output_path,
                        model,
                        confidence
                    )

                st.success("Video processing completed.")

                # Show processed video.
                st.subheader("Detection Result")
                with open(output_path, "rb") as f:
                    result_video_bytes = f.read()

                st.video(result_video_bytes)

                # Important: total_detections is frame-level, NOT unique potholes.
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Processed Frames", processed_frames)
                metric2.metric("Max Potholes / Frame", max_in_frame)
                metric3.metric("Frame-Level Detections", total_detections)

                st.download_button(
                    "Download Processed Video",
                    data=result_video_bytes,
                    file_name="pothole_detection_result.mp4",
                    mime="video/mp4",
                    key="download_video"
                )

                st.info(
                    "Frame-Level Detections is the sum of detections across frames. "
                    "It should not be interpreted as the number of unique physical potholes. "
                    "For unique pothole counting in production, add object tracking + GPS-based deduplication."
                )

            except Exception as e:
                st.error(f"Video processing failed: {e}")

            finally:
                # Clean temporary input file.
                if input_path and os.path.exists(input_path):
                    try:
                        os.remove(input_path)
                    except OSError:
                        pass

with live_tab:
    st.subheader("Live Pothole Detection")

    live_confidence = st.slider(
        "Live Detection Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="live_confidence"
    )

    st.info(
        "Click START and allow browser camera permission. "
        "Each camera frame is sent to the trained YOLOv8 + CBAM model."
    )

    try:
        webrtc_streamer(
            key="pothole-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: PotholeVideoProcessor(
                model,
                confidence=live_confidence
            ),
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )

    except Exception as e:
        st.error(f"Live camera could not start: {e}")
        st.info(
            "Make sure streamlit-webrtc is installed and the browser has camera permission."
        )
