# YOLO Object Detection with TensorFlow and OpenCV

A real-time object detection system implemented using YOLO (You Only Look Once) architecture with TensorFlow and OpenCV. This project provides capabilities for detecting and classifying objects in both images and video streams, with specific focus on detecting raccoons, horses, dogs, and cats.

## Features

- Real-time object detection in video streams
- Static image object detection
- Support for webcam input
- Pre-trained model for detecting 4 classes:
  - Raccoon
  - Horse
  - Dog
  - Cat
- Configurable confidence threshold
- Non-maximum suppression (NMS) for optimal detection
- Color-coded bounding boxes and labels

## Requirements

- Python 3.x
- TensorFlow >= 2.5.0
- OpenCV >= 4.5.0
- NumPy >= 1.19.0
- Pillow >= 8.0.0

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd Image-Processing
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model (you'll need to provide your own YOLO model file)
   - Place your model file in the project directory
   - Update the `model_path` in `main.py` to point to your model file

## Usage

### For Video Detection (Webcam)

```python
from main import ObjectDetector

detector = ObjectDetector()
detector.load_yolo_model('path_to_your_model.h5')

# Use webcam (0) or video file path
detector.detect_in_video(0)
```

### For Image Detection

```python
from main import ObjectDetector

detector = ObjectDetector()
detector.load_yolo_model('path_to_your_model.h5')

# Detect objects in an image
result = detector.detect_in_image('path_to_image.jpg')
```

## Configuration

You can modify the following parameters in the `ObjectDetector` class:

- `input_size`: Input image size for the YOLO model (default: 416)
- `confidence_threshold`: Minimum confidence for detection (default: 0.5)
- `nms_threshold`: Non-maximum suppression threshold (default: 0.4)

## Implementation Details

- Uses TensorFlow's Keras API for model loading and inference
- OpenCV for image processing and visualization
- Real-time object detection with bounding box visualization
- Color-coded detection boxes based on object class
- Confidence score display for each detection

## Controls

- Press 'q' to quit video detection mode
- For image detection, press any key to close the display window

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- YOLO (You Only Look Once) algorithm
- TensorFlow team
- OpenCV community