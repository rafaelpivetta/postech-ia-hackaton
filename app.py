import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

MODEL_PATH = "cutting_objects_detect_model.tflite"
#MODEL_PATH = "knife_detection_densenet.tflite"

def predict_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        prediction = predict_image(frame_pil)
        stframe.image(frame, caption=f"Prediction: {prediction}", use_column_width=True)
    cap.release()

if st.button("Open Webcam"):
    predict_webcam()

@st.cache_resource
def load_interpreter():
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

# Load TFLite model and allocate tensors
interpreter, input_details, output_details = load_interpreter()

def predict_image(image):
    # Preprocess the image to fit the model input
    image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(image, axis=0)
    input_data = np.array(input_data, dtype=np.float32)

    # Perform the prediction
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def predict_video(video):
    cap = cv2.VideoCapture(video)
    predictions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        prediction = predict_image(frame)
        predictions.append(prediction)
    cap.release()
    return predictions

st.title("Detecção de Objetos Cortantes")
st.write("Arraste ou carregue uma imagem ou vídeo para detectar objetos cortantes.")

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem", use_column_width=True)
        prediction = predict_image(image)
        st.markdown(f"<h2>Prediction: {prediction[0][0] * 100:.2f}%</h2>", unsafe_allow_html=True)
    elif uploaded_file.type.startswith("video"):
        video = uploaded_file.name
        with open(video, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.video(video)
        predictions = predict_video(video)
        threshold = st.slider("Set the prediction threshold", 0.0, 1.0, 0.7)
        count_above_threshold = sum(1 for prediction in predictions if prediction[0][0] > threshold)
        st.markdown(f"<h2>Number of frames with prediction above threshold: {count_above_threshold}</h2>", unsafe_allow_html=True)
    