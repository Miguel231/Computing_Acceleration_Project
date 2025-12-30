#!/usr/bin/env python3
"""
Setup and test script for Security Camera with ResNet100-ArcFace
"""

import os
import sys
import subprocess
import cv2
import time

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    required = ['flask', 'opencv-python', 'onnxruntime', 'numpy']
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            return False
    return True

def check_model_files():
    """Verify that model files exist."""
    print("\nChecking model files...")
    
    models = {
        'YuNet Face Detector': 'models/face_detection_yunet_2023mar.onnx',
        'ResNet100-ArcFace': 'models/arcface_resnet100.onnx'
    }
    
    all_present = True
    for name, path in models.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: Missing at {path}")
            all_present = False
            
    return all_present

def test_camera():
    """Test if camera is accessible."""
    print("\nTesting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"✅ Camera working (frame: {frame.shape})")
        return True
    else:
        print("❌ Could not read frame from camera")
        return False

def test_face_detection():
    """Test YuNet face detector."""
    print("\nTesting face detection...")
    
    try:
        import cv2
        import numpy as np
        
        # Load detector
        detector = cv2.FaceDetectorYN.create(
            "models/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000
        )
        
        # Create a test image (black with a white rectangle as "face")
        test_img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 80), (220, 200), (255, 255, 255), -1)
        
        detector.setInputSize((320, 240))
        _, faces = detector.detect(test_img)
        
        if faces is not None:
            print(f"✅ Face detector working (detected {len(faces)} faces)")
            return True
        else:
            print("❌ Face detector didn't detect test face")
            return False
            
    except Exception as e:
        print(f"❌ Face detector test failed: {e}")
        return False

def create_directory_structure():
    """Create required directories."""
    print("\nCreating directory structure...")
    
    dirs = ['models', 'templates']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created: {dir_name}/")
        else:
            print(f"Exists: {dir_name}/")
    
    return True

def download_yunet_model():
    """Download YuNet face detector if missing."""
    yunet_path = "models/face_detection_yunet_2023mar.onnx"
    
    if not os.path.exists(yunet_path):
        print("\nDownloading YuNet face detector...")
        
        # YuNet ONNX model URL from OpenCV
        yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(yunet_url, yunet_path)
            print(f"✅ Downloaded: {yunet_path}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("Please download manually from:")
            print(yunet_url)
            return False
    else:
        print("✅ YuNet model already exists")
        return True

def main():
    print("=" * 60)
    print("Security Camera Setup & Test")
    print("=" * 60)
    
    # Create directories
    create_directory_structure()
    
    # Check and download models
    if not check_model_files():
        print("\nSome models are missing.")
        
        # Try to download YuNet
        if not os.path.exists("models/face_detection_yunet_2023mar.onnx"):
            download_yunet_model()
        
        # Check ArcFace model
        if not os.path.exists("models/arcface_resnet100.onnx"):
            print("\nPlease download ArcFace ResNet100 model:")
            print("Command: wget -O models/arcface_resnet100.onnx https://github.com/onnx/models/raw/main/archive/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx")
            print("\nOr visit: https://github.com/onnx/models/tree/main/archive/vision/body_analysis/arcface")
            return
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies:")
        print("pip install flask opencv-python onnxruntime numpy")
        return
    
    # Test camera
    if not test_camera():
        print("\nCamera may need to be configured.")
        print("On Raspberry Pi: sudo raspi-config -> Interface Options -> Camera")
    
    # Test face detection
    test_face_detection()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the system: python main.py")
    print("2. Open browser to: http://localhost:5000")
    print("3. Enroll faces using the web interface")
    print("4. Adjust recognition threshold in main.py if needed")
    print("\nTroubleshooting:")
    print("- If camera shows 'No signal', check camera connection")
    print("- If enrollment fails, ensure good lighting and face visibility")
    print("- Adjust ALARM_THRESHOLD in main.py (default: 0.60)")
    print("  Lower = more strict, Higher = more lenient")

if __name__ == "__main__":
    main()
