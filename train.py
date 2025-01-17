from ultralytics import YOLO
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def train_knife_detector():
    # Get absolute path to the data.yaml file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(script_dir, 'data.yaml')
    
    
    # Initialize a new YOLO model
    model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model as starting point
    
    # Train the model
    results = model.train(
        data=data_yaml_path,  # Path to data config file
        epochs=1,  # Number of epochs
        imgsz=416,  # Image size
        batch=32,  # Batch size
        patience=20,  # Early stopping patience
        device='cpu',  # Use CPU (or 'cuda' if you have a GPU)
        project='runs/detect',  # Project name
        name='cutting_objects_detector'  # Run name
    )

if __name__ == "__main__":
    train_knife_detector()