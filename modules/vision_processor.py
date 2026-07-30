"""
Vision processing module using OpenCV and YOLO for object detection and AR overlays.
Detects parts and draws bounding boxes with risk color codes.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import io
import streamlit as st


# Cache the model loading so it only downloads ONCE on the server
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO('yolov8n.pt')
    except Exception as e:
        print(f"YOLO model load failed: {e}")
        return None


class VisionProcessor:
    def __init__(self):
        # Use the cached loader instead of calling YOLO directly
        self.model = load_yolo_model()

    def load_image(self, image_input):
        """
        Load image from file path, upload, or camera input.
        Returns a numpy array (BGR) for OpenCV.
        """
        if isinstance(image_input, str):
            # File path
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            # Assume it's a bytes-like object (from st.camera_input)
            img = Image.open(io.BytesIO(image_input))
            img = np.array(img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def draw_bounding_box(self, img, x1, y1, x2, y2, label, severity):
        """
        Draw bounding box with color based on severity.
        severity: 'CRITICAL' -> RED, 'WARNING' -> YELLOW, 'MODERATE' -> ORANGE, 'HEALTHY' -> GREEN
        """
        colors = {
            'CRITICAL': (0, 0, 255),   # RED
            'WARNING': (0, 255, 255),  # YELLOW
            'MODERATE': (0, 165, 255), # ORANGE
            'HEALTHY': (0, 255, 0)     # GREEN
        }
        color = colors.get(severity, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Add label text with background
        text = f"{label}: {severity}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def mock_detection(self, img):
        """
        Fallback mock detector using simple edge/color thresholding.
        Simulates detection of parts like 'belt', 'wire', 'chain', 'bulb'.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                # Randomly assign severity based on area (just for demo)
                if area > 3000:
                    severity = 'CRITICAL'
                elif area > 1500:
                    severity = 'WARNING'
                else:
                    severity = 'MODERATE'
                detections.append({
                    'bbox': (x, y, x+w, y+h),
                    'label': 'Part',
                    'severity': severity
                })
        return detections

    def run_detection(self, image_input, engine_type='car'):
        """
        Main detection pipeline.
        Returns annotated image and list of detections.
        """
        img = self.load_image(image_input)
        detections = []
        if self.model is not None:
            # Use YOLO
            results = self.model(img)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                # Map severity based on confidence and class (simplified)
                if conf > 0.7:
                    severity = 'CRITICAL'
                elif conf > 0.5:
                    severity = 'WARNING'
                else:
                    severity = 'MODERATE'
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'label': label,
                    'severity': severity,
                    'confidence': conf
                })
                img = self.draw_bounding_box(img, x1, y1, x2, y2, label, severity)
        else:
            # Mock detection
            detections = self.mock_detection(img)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                img = self.draw_bounding_box(img, x1, y1, x2, y2, det['label'], det['severity'])

        return img, detections
