from flask import Flask, request, jsonify, send_file, render_template, make_response
from flask_cors import CORS
import os
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from tempfile import NamedTemporaryFile
import io
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLO model
model_name = 'best_finetunned.pt'

def load_model():
    model = YOLO(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model

try:
    model = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Error handling
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    confidence_threshold = float(request.form.get('confidence', 0.25))
    
    try:
        # Handle image files
        image = Image.open(file)
        image_np = np.array(image)
        
        results = model(image_np, conf=confidence_threshold)
        plot = results[0].plot()
        
        # Convert numpy array to PIL Image
        plot_image = Image.fromarray(plot)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        plot_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Get detections
        has_detections = len(results[0].boxes) > 0
        detections = []
        if has_detections:
            for box in results[0].boxes:
                confidence = box.conf.item()
                detections.append({
                    'confidence': float(confidence)
                })
        
        # Return both the image and detection data
        response = make_response(send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='detected_objects.jpg'
        ))
        response.headers['X-Detections'] = json.dumps({
            'has_detections': has_detections,
            'detections': detections
        })
        return response
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect_webcam', methods=['POST'])
def detect_webcam():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    confidence_threshold = float(request.form.get('confidence', 0.25))
    
    try:
        # Read image from request
        image = Image.open(file)
        image_np = np.array(image)
        
        # Run inference
        results = model(image_np, conf=confidence_threshold)
        plot = results[0].plot()
        
        # Convert numpy array to PIL Image
        plot_image = Image.fromarray(plot)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        plot_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='webcam_detected.jpg'
        )
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)