"""
This file handles the AI models and the JSON file storing the face embeddings.
Updated for ResNet100-ArcFace ONNX model.
"""

import cv2
import numpy as np
import onnxruntime as ort
import json
import os
import datetime

class FaceSystem:
    def __init__(self):
        # Load YuNet Face Detector
        self.detector = cv2.FaceDetectorYN.create(
            "models/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000
        )
        
        # Load ResNet100-ArcFace Recognizer
        self.rec_sess = ort.InferenceSession(
            "models/arcface_resnet100.onnx", 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.rec_sess.get_inputs()[0].name
        
        # Model-specific parameters
        self.input_size = (112, 112)  # Required input size
        self.embedding_size = 512     # Output embedding size
        
        self.db_path = "family_db.json"
        self.log_path = "security_logs.csv"
        
        # Load existing family embeddings or create new database
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump({}, f)
        
        # Create logs file if it doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("Timestamp,Event,Details\n")
        
        print("System initialized with ResNet100-ArcFace model")
        print(f"Model input: {self.input_size}, Output: {self.embedding_size}D embeddings")

    def preprocess_face(self, face_img):
        """
        Preprocess face image for ResNet100-ArcFace model.
        Required preprocessing: BGR -> Resize -> Normalize -> Transpose
        """
        # Resize to 112x112
        face_img = cv2.resize(face_img, self.input_size)
        
        # Convert to float32
        face_img = face_img.astype(np.float32)
        
        # Normalize: (image - 127.5) / 128.0
        # This is specific to ResNet100-ArcFace
        face_img = (face_img - 127.5) / 128.0
        
        # Transpose from HWC (112, 112, 3) to CHW (3, 112, 112)
        face_img = np.transpose(face_img, (2, 0, 1))
        
        # Add batch dimension: (1, 3, 112, 112)
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img

    def get_embedding(self, frame):
        """
        Extract face embedding from a frame.
        Returns: 512-dimensional numpy array or None if no face detected
        """
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        
        if faces is not None and len(faces) > 0:
            # Take the first face detected (faces[0])
            # YuNet returns: [x, y, w, h, conf, landmarks...]
            f = faces[0][:4].astype(int)
            
            # Extract face region with boundary checks
            x1, y1 = max(0, f[0]), max(0, f[1])
            x2, y2 = min(w, f[0] + f[2]), min(h, f[1] + f[3])
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            face_img = frame[y1:y2, x1:x2]
            
            # Preprocess for ArcFace model
            preprocessed = self.preprocess_face(face_img)
            
            # Run inference
            try:
                embedding = self.rec_sess.run(None, {self.input_name: preprocessed})[0]
                # Return the 512D embedding (flatten if needed)
                return embedding.flatten()
            except Exception as e:
                print(f"Inference error: {e}")
                return None
        return None

    def check_identity(self, current_vec, threshold=0.60):
        """
        Compare current embedding with database.
        Note: Threshold may need adjustment for ArcFace (typically higher than GhostFaceNet).
        """
        if not os.path.exists(self.db_path):
            return None
            
        try:
            with open(self.db_path, 'r') as f:
                db = json.load(f)
        except json.JSONDecodeError:
            print("Database corrupted, creating new one")
            db = {}
            with open(self.db_path, 'w') as f:
                json.dump(db, f)
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for name, saved_vec in db.items():
            # Calculate cosine similarity (more appropriate for ArcFace)
            current_norm = np.linalg.norm(current_vec)
            saved_norm = np.linalg.norm(saved_vec)
            
            if current_norm == 0 or saved_norm == 0:
                continue
                
            # Cosine similarity: 1.0 = identical, 0.0 = orthogonal
            similarity = np.dot(current_vec, saved_vec) / (current_norm * saved_norm)
            distance = 1 - similarity  # Convert to distance metric
            
            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match = name
        
        return best_match

    def log_event(self, event, details):
        """Log security events to CSV file."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(f"{now},{event},{details}\n")
        print(f"[{now}] {event}: {details}")

    def enroll_face(self, name, frame):
        """
        Enroll a new face from the given frame.
        Returns: Success message or error
        """
        embedding = self.get_embedding(frame)
        
        if embedding is None:
            return "Error: No face detected in the image."
        
        # Load existing database
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                try:
                    db = json.load(f)
                except json.JSONDecodeError:
                    db = {}
        else:
            db = {}
        
        # Check if name already exists
        if name in db:
            return f"Warning: {name} already exists. Overwriting..."
        
        # Store the embedding
        db[name] = embedding.tolist()
        
        # Save database
        with open(self.db_path, 'w') as f:
            json.dump(db, f, indent=2)
        
        return f"Success: {name} enrolled with embedding size {len(embedding)}"
