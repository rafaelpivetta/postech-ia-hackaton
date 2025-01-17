import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Knife Detection", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Detecção de Objetos Cortantes")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Detected Objects")
        try:
            # Run inference
            results = model(image_np)
            
            # Plot results
            plot = results[0].plot()
            
            # Display results
            st.image(plot, use_column_width=True)
            
            # Show detections
            if len(results[0].boxes) > 0:
                st.write("Detections:")
                for box in results[0].boxes:
                    confidence = box.conf.item()
                    st.write(f"- Confidence: {confidence:.2%}")
            else:
                st.write("No objects detected")
                
        except Exception as e:
            st.error(f"Error during inference: {e}")

# Add webcam support
if st.button("Use Webcam"):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam")
            st.stop()
            
        frame_placeholder = st.empty()
        stop_button = st.button("Stop")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read from webcam")
                break
                
            # Run inference
            results = model(frame)
            
            # Plot results
            plot = results[0].plot()
            
            # Convert BGR to RGB
            plot_rgb = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
            
            # Display frame
            frame_placeholder.image(plot_rgb, channels="RGB")
            
        cap.release()
        
    except Exception as e:
        st.error(f"Error with webcam: {e}")
    