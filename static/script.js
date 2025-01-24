// Update confidence value display
document.getElementById('confidence').addEventListener('input', function(e) {
    document.getElementById('confidenceValue').textContent = e.target.value;
});

// Preview uploaded file
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const preview = document.getElementById('originalPreview');
    preview.innerHTML = '';

    if (file) {
        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            preview.appendChild(img);
        } else if (file.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(file);
            video.controls = true;
            preview.appendChild(video);
        }
    }
});

let mediaStream = null;
let isWebcamActive = false;
let detectionInterval = null;

async function startWebcam() {
    const video = document.getElementById('webcamVideo');
    const startButton = document.getElementById('startWebcam');
    const stopButton = document.getElementById('stopWebcam');

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        video.srcObject = mediaStream;
        video.style.display = 'block';
        isWebcamActive = true;
        startButton.style.display = 'none';
        stopButton.style.display = 'inline-block';
        
        // Start detection loop
        startWebcamDetection();
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Error accessing webcam: " + err.message);
    }
}

function stopWebcam() {
    const video = document.getElementById('webcamVideo');
    const startButton = document.getElementById('startWebcam');
    const stopButton = document.getElementById('stopWebcam');
    const canvas = document.getElementById('webcamCanvas');

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }

    video.style.display = 'none';
    canvas.style.display = 'none';
    isWebcamActive = false;
    startButton.style.display = 'inline-block';
    stopButton.style.display = 'none';
}

async function startWebcamDetection() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const ctx = canvas.getContext('2d');
    const confidenceThreshold = document.getElementById('confidence').value;

    canvas.style.display = 'block';

    detectionInterval = setInterval(async () => {
        if (!isWebcamActive) return;

        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Get canvas data as blob
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'webcam.jpg');
            formData.append('confidence', confidenceThreshold);

            try {
                const response = await fetch('/api/detect_webcam', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Detection failed');

                const result = await response.blob();
                const imgUrl = URL.createObjectURL(result);
                const img = new Image();
                img.onload = () => {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    URL.revokeObjectURL(imgUrl);
                };
                img.src = imgUrl;
            } catch (error) {
                console.error('Error during webcam detection:', error);
            }
        }, 'image/jpeg');
    }, 100); // Adjust interval as needed (currently 100ms)
}

// Add event listeners for webcam buttons
document.getElementById('startWebcam').addEventListener('click', startWebcam);
document.getElementById('stopWebcam').addEventListener('click', stopWebcam);

function updateAlertVisibility(hasDetections) {
    const alertDiv = document.getElementById('alert');
    if (hasDetections) {
        alertDiv.classList.remove('d-none');
    } else {
        alertDiv.classList.add('d-none');
    }
}

// Modify your existing detectObjects function
async function detectObjects() {
    document.getElementById('alert').classList.add('d-none');
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file first');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', document.getElementById('confidence').value);

    document.getElementById('loading').style.display = 'block';

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Network response was not ok');

        // Get detection data from headers
        const detectionData = JSON.parse(response.headers.get('X-Detections'));
        
        // Update alert visibility based on detections
        updateAlertVisibility(detectionData.has_detections);

        // Handle the image response
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        
        // Update the preview
        document.getElementById('resultPreview').innerHTML = 
            `<img src="${imageUrl}" alt="Detected Objects">`;

    } catch (error) {
        console.error('Error:', error);
        alert('Error processing the file');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}