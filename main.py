import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os

class ObjectDetector:
    def __init__(self):
        self.classes = ['raccoon', 'horse', 'dog', 'cat']
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Initialize model parameters
        self.input_size = 416
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
    def load_yolo_model(self, model_path):
        """Load the YOLO model"""
        try:
            self.model = load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def preprocess_image(self, image):
        """Preprocess image for YOLO model"""
        # Resize image
        image = cv2.resize(image, (self.input_size, self.input_size))
        # Normalize
        image = image.astype(np.float32) / 255.0
        # Expand dimensions
        image = np.expand_dims(image, axis=0)
        return image
    
    def detect_objects(self, image):
        """Detect objects in image"""
        original_image = image.copy()
        h, w = original_image.shape[:2]
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get predictions
        predictions = self.model.predict(processed_image)
        
        # Process predictions and draw boxes
        boxes, scores, classes = self.process_predictions(predictions, (h, w))
        
        # Draw detections
        for i in range(len(boxes)):
            if scores[i] > self.confidence_threshold:
                x1, y1, x2, y2 = boxes[i]
                class_id = int(classes[i])
                color = self.colors[class_id]
                
                # Draw bounding box
                cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f'{self.classes[class_id]}: {scores[i]:.2f}'
                cv2.putText(original_image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return original_image
    
    def process_predictions(self, predictions, image_shape):
        """Process YOLO predictions and return boxes, scores, and classes"""
        # This is a placeholder for actual YOLO prediction processing
        # The actual implementation will depend on your model's output format
        return [], [], []
    
    def detect_in_image(self, image_path):
        """Detect objects in a single image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return None
        
        return self.detect_objects(image)
    
    def detect_in_video(self, video_path=0):
        """Detect objects in video stream"""
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            processed_frame = self.detect_objects(frame)
            
            # Display result
            cv2.imshow('Object Detection', processed_frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = ObjectDetector()
    
    # Load model (you'll need to provide the path to your trained model)
    model_path = 'path_to_your_model.h5'
    if not detector.load_yolo_model(model_path):
        return
    
    # Example usage
    # For image detection
    # image_path = 'path_to_image.jpg'
    # result = detector.detect_in_image(image_path)
    # if result is not None:
    #     cv2.imshow('Detection Result', result)
    #     cv2.waitKey(0)
    
    # For video detection (0 for webcam)
    detector.detect_in_video(0)

if __name__ == "__main__":
    main()