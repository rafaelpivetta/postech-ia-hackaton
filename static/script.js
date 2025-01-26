function showConfidenceSliderAndUploadCard() {
    const confidenceSlider = document.getElementById('confidenceSlider');
    const uploadCard = document.getElementById('uploadCard');
    
    clearAlerts();
    clearFileInputAndPreviews();
    
    confidenceSlider.addEventListener('input', function(e) {
        document.getElementById('confidenceRangeValue').textContent = e.target.value;
        clearAlerts();

        const fileInput = document.getElementById('fileUpload');

        
        if (fileInput.files.length > 0) {
            detectObjects();
        }
    });
    confidenceSlider.style.display = 'block';
    uploadCard.style.display = 'block';

    document.getElementById('fileUpload').addEventListener('change', function(event) {
        const file = event.target.files[0];
          
        if (file) {
            detectObjects();
        }
    });
}

function triggerAlert(hasDetections) {
    
    if (hasDetections) {
        document.getElementById('detectedObjectsAlert').classList.remove('d-none');
        document.getElementById('noDetectionsAlert').classList.add('d-none');
    } else {
        document.getElementById('noDetectionsAlert').classList.remove('d-none');
        document.getElementById('detectedObjectsAlert').classList.add('d-none');
    }
}

function clearAlerts() {
    document.getElementById('detectedObjectsAlert').classList.add('d-none');
    document.getElementById('noDetectionsAlert').classList.add('d-none');
}

function clearFileInputAndPreviews() {
    const fileInput = document.getElementById('fileUpload');
    fileInput.value = '';

    const processedImage = document.getElementById('processedImage');
    const processedVideo = document.getElementById('processedVideo');
    processedImage.style.display = 'none';
    processedVideo.style.display = 'none';
}

// Modify your existing detectObjects function
async function detectObjects() {

    console.log('---detectObjects---');
    console.log(document.getElementById('fileUpload').value);
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
        
        // Update alert visibility based on detections
        triggerAlert(detectionData.has_detections);

        const processedImage = document.getElementById('processedImage');
        const processedVideo = document.getElementById('processedVideo');
        
        processedImage.style.display = 'none';
        processedVideo.style.display = 'none';

        const blob = await response.blob();

        if (file.type.startsWith('image/')) {
            processedImage.src = URL.createObjectURL(blob);
            processedImage.style.display = 'block';
        } 
        else if (file.type.startsWith('video/')) {

            // Create download link for video
            const downloadLink = document.createElement('a');
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

