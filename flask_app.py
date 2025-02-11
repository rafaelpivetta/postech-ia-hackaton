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
from alertSMSNotification import send_twilio_sms_notification
from alertEmailNotification import send_email_notification
import tempfile
from Rastrear import *
import base64

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model only once when the container starts
model = None

def get_model():
    global model
    if model is None:
        model = YOLO('best_finetunned.pt')
        device = 'cpu'  # Force CPU for Cloud Run
        model.to(device)
    return model

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

@app.route('/projeto')
def projeto():
    return render_template('projeto.html')

def process_video(file, confidence_threshold):
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
        file.save(temp_input.name)
        temp_input_path = temp_input.name

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
        temp_output_path = temp_output.name

    try:
        # Open video capture
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        has_detections = False
        detections = []

        first_detection_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection on frame
            results = get_model()(frame, conf=confidence_threshold)
            annotated_frame = results[0].plot()
            
            # Check for detections
            if len(results[0].boxes) > 0:
                has_detections = True
                for box in results[0].boxes:
                    detections.append({
                        'confidence': float(box.conf.item())
                    })

                # Save first frame with detections
                if first_detection_frame is None:
                    first_detection_frame = annotated_frame.copy()
            
            # Write frame
            out.write(annotated_frame)

        # Release resources
        cap.release()
        out.release()
        
        return temp_output_path, has_detections, detections, first_detection_frame
        
    except Exception as e:
        # Clean up temp files in case of error
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)
        raise e
    finally:
        # Clean up input file
        os.unlink(temp_input_path)

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    confidence_threshold = float(request.form.get('confidence', 0.25))
    
    try:
        
        if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            
            output_video_path, has_detections, detections, first_detection_frame = process_video(file, confidence_threshold)
            
            # Return video file
            response = make_response(send_file(
                output_video_path,
                mimetype='video/mp4',
                as_attachment=True,
                download_name='detected_video.mp4'
            ))
            
            response.headers['X-Detections'] = json.dumps({
                'has_detections': has_detections,
                'detections': detections
            })
            
            if first_detection_frame is not None:
                # Convert frame to PIL Image
                frame_image = Image.fromarray(cv2.cvtColor(first_detection_frame, cv2.COLOR_BGR2RGB))
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                frame_image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                
                # Add image to response
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                response.headers['X-Detection-Image'] = json.dumps({
                    'image': img_base64
                })
            
            return response
        else:
            image = Image.open(file)
            image_np = np.array(image)
            
            results = get_model()(image_np, conf=confidence_threshold)
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
    
tracked_ids = set()

@app.route('/api/detect_webcam', methods=['POST'])
# Conjunto global para armazenar os IDs dos objetos rastreados


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
        results = get_model()(image_np, conf=confidence_threshold)  
        plot = results[0].plot()  

        # Get detections and track knives  
        has_detections, detections, trackers = ProcessarWEBCAM(results[0].boxes, confidence_threshold, image_np)  

        # Convert numpy array to PIL Image  
        plot_image = Image.fromarray(plot)  

        # Drawing bounding boxes with IDs on the plot
        for idx, detection in enumerate(detections):
            box = detection['box']  # Assuming box is in [x_min, y_min, x_max, y_max]
            detection_id = detection['id']  # Assuming each detection has a unique 'id'

            # Check if the ID is new
            if detection_id not in tracked_ids:
                # New ID found, send notification and add to tracked set
                tracked_ids.add(detection_id)
                # Aqui você pode chamar a função para enviar a notificação (ex: enviar_notificacao(detection_id))

                # Exemplo de envio de notificação
                print(f"Novo objeto detectado com ID: {detection_id}")

            # Label to display on the bounding box
            label = f"ID: {detection_id}"

            # Draw the box and the ID
            plot_image = Desenhar(plot_image, box, label)

        # Save to bytes  
        img_byte_arr = io.BytesIO()  
        plot_image.save(img_byte_arr, format='JPEG')  
        img_byte_arr.seek(0)  

        # Create directory for detected knives  
        #knife_dir = criar_pasta_para_facas()  

        # Save detected knives  
        #Guardar_facas_detectadas(detections, knife_dir, image_np)  

        # Return both the image and detection data  
        response = make_response(send_file(  
            img_byte_arr,  
            mimetype='image/jpeg',  
            as_attachment=True,  
            download_name='webcam_detected.jpg'  
        ))  

        # Add detection data to headers  
        response.headers['X-Detections'] = json.dumps({  
            'has_detections': has_detections,  
            'detections': detections,
            'ObjectID': [detection['id'] for detection in detections]
        })  

        return response  

    except Exception as e:  
        return jsonify({'error': str(e)}), 500


@app.route('/api/send_notification', methods=['POST'])
def send_notification():
    data = request.json
    detection_mode = data.get('detection_mode')  # (imagem, vídeo ou webcam)
    notification_type = data.get('notification_type')  # Tipo de notificação (push, por exemplo)
    sms_number = data.get('sms_number')  # Número de telefone para SMS
    email_address = data.get('email_address')  # Endereço de e-mail para notificação por e-mail
    image_base64 = data.get('image_base64')  # Imagem em base64 para notificação por e-mail
    
    if notification_type == 'sms' and sms_number:
        try:
            send_twilio_sms_notification(sms_number, detection_mode)
            return jsonify({"status": "success", "message": "SMS enviado com sucesso."}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": f"Falha ao enviar o SMS: {str(e)}"}), 500
    
    elif notification_type == 'email' and email_address:
        try:
            send_email_notification(email_address, detection_mode, image_base64)
            return jsonify({"status": "success", "message": "E-mail enviado com sucesso."}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": f"Falha ao enviar a notificação: {str(e)}"}), 500
        
    else:
        return jsonify({"status": "error", "message": "Dados inválidos ou faltando."}), 400



if __name__ == '__main__':
    # Use the PORT environment variable provided by Cloud Run
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)