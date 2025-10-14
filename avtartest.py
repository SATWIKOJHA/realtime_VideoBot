import cv2
import face_detection
import torch
import numpy as np

# Load image and check if valid
img = cv2.imread('avtar.png')
if img is None:
    raise ValueError("Failed to load avtar.png. Ensure the file exists and is a valid image.")

# Initialize detector
try:
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        face_detector='sfd',
        path_to_detector='face_detection/detection/sfd/s3fd.pth'
    )
    print("Detector initialized successfully")
except Exception as e:
    print(f"Error initializing detector: {e}")
    raise

# Perform face detection
try:
    detections = detector.get_detections_for_batch(np.array([img]))
    print("Detections:", detections)
except Exception as e:
    print(f"Error during face detection: {e}")
    raise