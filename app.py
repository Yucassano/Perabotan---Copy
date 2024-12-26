from ultralytics import YOLO  # YOLOv8
import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Load YOLO model with caching
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Process and display the detection results
def display_results(image, results):
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    labels = results.boxes.cls.cpu().numpy()  # Class indices
    names = results.names  # Class names

    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Main Streamlit app
def main():
    st.title("Real-Time Object Detection")
    st.sidebar.title("Settings")

    # Load YOLO model
    model_path = "best.pt"  # Replace with your model path
    model = load_model(model_path)

    # Start/stop detection
    run_detection = st.sidebar.checkbox("Start Detection")

    if run_detection:
        cap = cv2.VideoCapture(0)  # Open default camera
        if not cap.isOpened():
            st.error("Failed to access camera. Ensure it's connected and not used by another application.")
            return

        st_frame = st.empty()  # Placeholder for video frames
        stop_button = st.sidebar.button("Stop Detection")  # Stop button

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture image. Check your camera connection.")
                break

            # Convert BGR to RGB for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLO detection
            results = model.predict(frame, imgsz=640)

            # Draw results on the frame
            frame = display_results(frame, results[0])

            # Display the frame in Streamlit
            st_frame.image(frame, channels="RGB", use_container_width=True)

            # Check for stop signal
            if stop_button:
                break

        cap.release()
        st.success("Detection stopped.")

if __name__ == "__main__":
    main()

