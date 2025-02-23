// Add these variables at the top
let webcamStream = null;
let isWebcamActive = false;
const Objetos = new Set();

function showConfidenceSliderAndUploadCard() {
    const mode = document.getElementById('detectionMode').value;
    const confidenceSlider = document.getElementById('confidenceSlider');
    const uploadCard = document.getElementById('uploadCard');
    const webcamCard = document.getElementById('webcamCard');
    const notificationsCard = document.getElementById('notificationsCard');

    let slider = document.querySelector("input[type='range']");
    slider.value = "0.25";
    
    clearAlerts();
    clearFileInputAndPreviews();
    stopWebcam(); // Stop webcam if running
   
    document.getElementById('confidenceRangeValue').textContent = 0.25; 

    confidenceSlider.addEventListener('input', function(e) {
        document.getElementById('confidenceRangeValue').textContent = e.target.value;
        clearAlerts();
        clearFileInputAndPreviews();
    });
    confidenceSlider.style.display = 'block';
    notificationsCard.style.display = 'block';
    
    if (mode === 'webcam') {
        uploadCard.style.display = 'none';
        webcamCard.style.display = 'block';
        setupWebcam();
    } else {
        uploadCard.style.display = 'block';
        webcamCard.style.display = 'none';
    }

    document.getElementById('fileUpload').addEventListener('change', function(event) {
        clearAlerts();
        clearPreviews();
        const file = event.target.files[0];
          
        if (file) {
            detectObjects();
        }
    });
}

function triggerAlert(hasDetections, image_base64, detection_mode) {
    
    if (hasDetections) {
        document.getElementById('detectedObjectsAlert').classList.remove('d-none');
        document.getElementById('noDetectionsAlert').classList.add('d-none');

        const notificationType = document.querySelector('input[name="notificationType"]:checked')?.value;
        if (notificationType !== 'none' && notificationType !== undefined) {
            sendAlert(image_base64, detection_mode);
        }
    } else {
        document.getElementById('noDetectionsAlert').classList.remove('d-none');
        document.getElementById('detectedObjectsAlert').classList.add('d-none');
    }
    
}

async function sendAlert(image_base64, detection_mode) {

    const notificationType = document.querySelector('input[name="notificationType"]:checked')?.value;
    const smsNumber = "+55" + document.getElementById('smsNumber').value;
    const emailAddress = document.getElementById('emailAddress').value;

    const data = {
        detection_mode: detection_mode,
        notification_type: notificationType,
        sms_number: smsNumber,
        email_address: emailAddress,
        image_base64: image_base64
    };

    try {
        const response = await fetch('/api/send_notification', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Network response was not ok');

        // Lógica para processar a resposta, se necessário
        const result = await response.json();
        document.getElementById("notificationSentAlert").classList.remove("d-none");
        
    } catch (error) {
        console.error('Error:', error);
    }
}

function clearNotificationSentAlert() {
    const notificationAlert = document.getElementById('notificationSentAlert');
    if (notificationAlert) {
        notificationAlert.classList.add('d-none');
    }
}

function clearAlerts() {
    document.getElementById('detectedObjectsAlert').classList.add('d-none');
    document.getElementById('noDetectionsAlert').classList.add('d-none');
    document.getElementById('notificationSentAlert').classList.add('d-none');
}

function clearFileInputAndPreviews() {
    const fileInput = document.getElementById('fileUpload');
    fileInput.value = '';

    clearPreviews();
}

function clearPreviews() {
    const processedImage = document.getElementById('processedImage');
    const processedVideo = document.getElementById('processedVideo');
    processedImage.style.display = 'none';
    processedVideo.style.display = 'none';

    const downloadLink = document.getElementById('downloadLink');

    if (downloadLink) {
        downloadLink.remove();
    }
}

// Modify your existing detectObjects function
async function detectObjects() {

    document.getElementById('loadingIndicator').classList.remove('d-none');
    document.getElementById('uploadCardBody').classList.add('d-none');
    
    // Clear any existing alerts while processing
    clearAlerts();
    const fileInput = document.getElementById('fileUpload');
    const file = fileInput.files[0];
    if (!file) {
        alert('Por favor, selecione um arquivo primeiro');
        return;
    }

    // Check file size (limit to 70 MB)
    const fileSizeLimit = 70 * 1024 * 1024; // 70 MB in bytes
    if (file.size > fileSizeLimit) {
        alert('O arquivo excede o limite de tamanho de 70 MB');
        document.getElementById('loadingIndicator').classList.add('d-none');
        document.getElementById('uploadCardBody').classList.remove('d-none');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', document.getElementById('confidenceRange').value);

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        document.getElementById('loadingIndicator').classList.add('d-none');
        document.getElementById('uploadCardBody').classList.remove('d-none');
        if (!response.ok) throw new Error('Network response was not ok');

        // Get detection data from headers
        const detectionData = JSON.parse(response.headers.get('X-Detections'));

        const blob = await response.blob();

        let image_base64 = null;
        let detection_mode = "Imagem"

        if (file.type.startsWith('image/')) {
            image_base64 = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result.split(',')[1]);
                reader.onerror = reject;
                reader.readAsDataURL(blob);
            });
        }

        if (file.type.startsWith('video/')) {
            const imageDetection = JSON.parse(response.headers.get('X-Detection-Image'));
            if (imageDetection) {
                image_base64 = imageDetection.image;
            }
            detection_mode = "Vídeo"
        }

        // Update alert visibility based on detections
        triggerAlert(detectionData.has_detections, image_base64, detection_mode);

        const processedImage = document.getElementById('processedImage');
        const processedVideo = document.getElementById('processedVideo');
        
        processedImage.style.display = 'none';
        processedVideo.style.display = 'none';

        if (file.type.startsWith('image/')) {
            processedImage.src = URL.createObjectURL(blob);
            processedImage.style.display = 'block';
        } 
        else if (file.type.startsWith('video/')) {

            // Create download link for video
            const downloadLink = document.createElement('a');
            downloadLink.id = 'downloadLink';
            downloadLink.href = URL.createObjectURL(blob);
            downloadLink.download = 'detected_video.mp4';
            downloadLink.className = 'btn btn-primary mt-3';
            downloadLink.innerHTML = 'Baixar Video Processado';

            // Add download button after video
            processedVideo.parentNode.insertBefore(downloadLink, processedVideo.nextSibling);

        }
  

    } catch (error) {
        console.error('Error:', error);
        alert('Error processing the file');
    } 
}

// Add these new functions for webcam handling
async function setupWebcam() {
    const startButton = document.getElementById('startWebcam');
    const stopButton = document.getElementById('stopWebcam');
    
    startButton.onclick = startWebcam;
    stopButton.onclick = stopWebcam;
}

async function startWebcam() {
    try {
        const video = document.getElementById('webcamVideo');
        const startButton = document.getElementById('startWebcam');
        const stopButton = document.getElementById('stopWebcam');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        
        video.srcObject = stream;
        webcamStream = stream;
        isWebcamActive = true;
        
        startButton.style.display = 'none';
        stopButton.style.display = 'inline-block';
        
        // Start detection loop
        detectWebcam();
        
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Erro ao acessar a webcam. Verifique as permissões.');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
        isWebcamActive = false;
        
        const video = document.getElementById('webcamVideo');
        video.srcObject = null;
        
        const startButton = document.getElementById('startWebcam');
        const stopButton = document.getElementById('stopWebcam');
        startButton.style.display = 'inline-block';
        stopButton.style.display = 'none';
    }
}

async function detectWebcam() {
    if (!isWebcamActive) return;
    
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const ctx = canvas.getContext('2d');
    const gallery = document.getElementById('detectionGallery');
    
    // Wait for video to be ready
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(detectWebcam);
        return;
    }
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0);
    
    try {
        // Get blob directly using a Promise wrapper
        const blob = await new Promise((resolve) => {
            canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.9);
        });
        
        if (!blob) {
            console.error('Failed to create blob from canvas');
            requestAnimationFrame(detectWebcam);
            return;
        }

        const formData = new FormData();
        formData.append('file', blob, 'webcam.jpg');
        formData.append('confidence', document.getElementById('confidenceRange').value);
        
        const response = await fetch('/api/detect_webcam', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const detectionData = JSON.parse(response.headers.get('X-Detections'));
            
            let newDetections = false;
            
            detectionData.detections.forEach(detection => {
                if (!Objetos.has(detection.id)) {
                    Objetos.add(detection.id);
                    newDetections = true;
                }
            });
            
            const blob = await response.blob();

            if (newDetections) {
                const image_base64 = await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
                triggerAlert(detectionData.has_detections, image_base64, "Webcam");
            }

            const imgUrl = URL.createObjectURL(blob);
            const img = new Image();
            
            img.onload = () => {
                ctx.drawImage(img, 0, 0);
                
                // If detection found, add to gallery
                if (detectionData.has_detections) {
                    const container = document.createElement('div');
                    const detectedImg = document.createElement('img');
                    const timestamp = document.createElement('div');
                    
                    detectedImg.src = imgUrl;
                    timestamp.className = 'detection-timestamp';
                    timestamp.textContent = new Date().toLocaleTimeString();

                    if (newDetections) { 
                        container.appendChild(detectedImg);
                        container.appendChild(timestamp);
                        gallery.insertBefore(container, gallery.firstChild);
                    }

                    // Keep only last 10 detections
                    if (gallery.children.length > 10) {
                        gallery.removeChild(gallery.lastChild);
                    }
                } else {
                    URL.revokeObjectURL(imgUrl);
                }
                
                // Continue detection if webcam is still active
                if (isWebcamActive) {
                    requestAnimationFrame(detectWebcam);
                }
            };
            img.src = imgUrl;
        }
    } catch (error) {
        console.error('Error during webcam detection:', error);
        if (isWebcamActive) {
            setTimeout(detectWebcam, 1000);
        }
    }
}

