from ultralytics import YOLO
import cv2
import os
import argparse
from datetime import datetime

def process_video(model_path, video_path, output_path=None, conf_threshold=0.25):
    """
    Process a video file with YOLOv8 model and save the results
    
    Args:
        model_path: Path to the trained model
        video_path: Path to input video
        output_path: Path to save the output video (optional)
        conf_threshold: Confidence threshold for detections
    """
    # Load the model
    model = YOLO(model_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize counters
    frames_with_detections = 0
    total_detections = 0
    frame_count = 0
    
    print(f"Processing video with {total_frames} frames...")
    start_time = datetime.now()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
        
        # Run detection
        results = model(frame, conf=conf_threshold)[0]
        
        # Count detections
        if len(results.boxes) > 0:
            frames_with_detections += 1
            total_detections += len(results.boxes)
        
        # Draw results on frame
        annotated_frame = results.plot()
        
        # Write frame if output path is provided
        if output_path:
            out.write(annotated_frame)
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    
    # Calculate statistics
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Print results
    print("\nVideo Analysis Results:")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Detection rate: {frames_with_detections/frame_count*100:.2f}%")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing speed: {frame_count/processing_time:.2f} FPS")
    
    if output_path:
        print(f"\nProcessed video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process video with YOLOv8 model')
    parser.add_argument('--model', default='best.pt',
                      help='Path to the trained model')
    parser.add_argument('--video', required=True,
                      help='Path to input video file')
    parser.add_argument('--output', default=None,
                      help='Path to save output video (optional)')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    try:
        process_video(args.model, args.video, args.output, args.conf)
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main() 