"""
Runs the Security Daemon and the Web Server simultaneously.
Updated for ResNet100-ArcFace integration.
"""

import threading
import time
import cv2
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from gpiozero import MotionSensor
from system_core import FaceSystem

# --- INIT ---
app = Flask(__name__)
system = FaceSystem()
camera_lock = threading.Lock()
cap = cv2.VideoCapture(0)

# Set camera resolution for better face detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# GPIO Setup (comment out if testing without hardware)
try:
    pir = MotionSensor(18)  # Connect OUT to GPIO 18
    USE_PIR = True
    print("PIR sensor initialized on GPIO 18")
except Exception as e:
    print(f"PIR sensor not available: {e}. Running in manual mode.")
    USE_PIR = False

# Global variables for manual testing
TEST_MODE = False
ALARM_THRESHOLD = 0.60  # Adjust this based on testing

def security_daemon():
    """Background thread for security monitoring."""
    print("Security Daemon: Active and monitoring...")
    
    while True:
        if USE_PIR:
            pir.wait_for_motion()
            system.log_event("MOTION", "PIR sensor triggered")
        elif TEST_MODE:
            time.sleep(5)  # Check every 5 seconds in test mode
            system.log_event("TEST", "Manual check cycle")
        else:
            # If no PIR and not in test mode, wait for external trigger
            time.sleep(1)
            continue
        
        is_family_present = False
        recognized_name = None
        
        # Take multiple shots when motion is detected
        for i in range(5):
            time.sleep(0.5)  # Wait between shots
            
            with camera_lock:
                ret, frame = cap.read()
            
            if not ret:
                system.log_event("ERROR", "Failed to capture frame")
                continue
            
            # Get face embedding
            vec = system.get_embedding(frame)
            if vec is not None:
                name = system.check_identity(vec, threshold=ALARM_THRESHOLD)
                if name:
                    system.log_event("ACCESS", f"Recognized: {name} (shot {i+1}/5)")
                    is_family_present = True
                    recognized_name = name
                    break
            else:
                system.log_event("INFO", f"No face detected in shot {i+1}/5")
        
        if not is_family_present:
            system.log_event("ALARM", "Intruder Alert! No family detected in 5 shots.")
            # INSERT YOUR ALARM WEBHOOK HERE (e.g., requests.post(...))
            # Example:
            # import requests
            # requests.post('https://your-webhook-url', json={'event': 'intruder'})
        
        if USE_PIR:
            # Wait for motion to stop before next detection
            pir.wait_for_no_motion()
            system.log_event("INFO", "Motion stopped")
        elif TEST_MODE:
            time.sleep(5)  # Wait before next test cycle

def gen_frames():
    """Generate video frames for streaming."""
    while True:
        with camera_lock:
            success, frame = cap.read()
            if not success:
                break
        
        # Draw detection rectangle if you want visual feedback
        h, w, _ = frame.shape
        system.detector.setInputSize((w, h))
        _, faces = system.detector.detect(frame)
        
        if faces is not None:
            for face in faces:
                box = face[0:4].astype(int)
                cv2.rectangle(frame, (box[0], box[1]), 
                             (box[0] + box[2], box[1] + box[3]), 
                             (0, 255, 0), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def dashboard():
    """Main dashboard page."""
    try:
        with open(system.log_path, "r") as f:
            logs = f.readlines()[-15:]  # Show last 15 entries
    except FileNotFoundError:
        logs = ["No logs found"]
    
    # Load enrolled names for display
    try:
        with open(system.db_path, "r") as f:
            db = json.load(f)
            enrolled_names = list(db.keys())
    except (FileNotFoundError, json.JSONDecodeError):
        enrolled_names = []
    
    return render_template('index.html', 
                         logs=reversed(logs), 
                         enrolled_names=enrolled_names,
                         threshold=ALARM_THRESHOLD)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/enroll', methods=['POST'])
def enroll():
    """Enroll a new face."""
    name = request.form.get('name')
    
    if not name:
        return jsonify({"error": "No name provided"}), 400
    
    with camera_lock:
        ret, frame = cap.read()
    
    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500
    
    result = system.enroll_face(name, frame)
    
    if "Success" in result:
        return jsonify({"success": result}), 200
    else:
        return jsonify({"error": result}), 400

@app.route('/manual_check', methods=['POST'])
def manual_check():
    """Manually trigger face check (for testing without PIR)."""
    with camera_lock:
        ret, frame = cap.read()
    
    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500
    
    vec = system.get_embedding(frame)
    if vec is not None:
        name = system.check_identity(vec, threshold=ALARM_THRESHOLD)
        if name:
            system.log_event("MANUAL_CHECK", f"Recognized: {name}")
            return jsonify({"status": "recognized", "name": name}), 200
        else:
            system.log_event("MANUAL_CHECK", "Unknown person")
            return jsonify({"status": "unknown"}), 200
    else:
        system.log_event("MANUAL_CHECK", "No face detected")
        return jsonify({"status": "no_face"}), 200

@app.route('/get_logs', methods=['GET'])
def get_logs():
    """Get recent logs as JSON."""
    try:
        with open(system.log_path, "r") as f:
            logs = f.readlines()[-50:]  # Last 50 entries
        return jsonify({"logs": logs}), 200
    except FileNotFoundError:
        return jsonify({"logs": []}), 200

@app.route('/delete_enrollment/<name>', methods=['DELETE'])
def delete_enrollment(name):
    """Delete an enrolled face."""wget https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true -O models/arcface.onnx
    try:
        with open(system.db_path, "r") as f:
            db = json.load(f)
        
        if name in db:
            del db[name]
            with open(system.db_path, "w") as f:
                json.dump(db, f, indent=2)
            
            system.log_event("ADMIN", f"Deleted enrollment: {name}")
            return jsonify({"success": f"Deleted {name}"}), 200
        else:
            return jsonify({"error": f"Name '{name}' not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_camera', methods=['GET'])
def test_camera():
    """Test if camera is working."""
    with camera_lock:
        ret, frame = cap.read()
    
    if ret:
        # Return a small preview
        frame = cv2.resize(frame, (320, 240))
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if ret:
            system.log_event("TEST", "Camera test successful")
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    system.log_event("ERROR", "Camera test failed")
    return jsonify({"error": "Camera not working"}), 500

if __name__ == "__main__":
    # Start the background security thread
    if USE_PIR or TEST_MODE:
        t = threading.Thread(target=security_daemon, daemon=True)
        t.start()
        print("Security daemon started")
    
    # Start the web interface
    print(f"Starting web server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
