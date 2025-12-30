
from sys import argv, exit
import cv2
import numpy as np

from face_detection import FaceDetector
from embeddings import Embeddings
from comunication import NtfyNotifier


def cosine_sim_from_l2(emb1, emb2):
    return float(np.dot(emb1, emb2))


def interpret_similarity(sim):
    # Starting points; tune for your model/data.
    if sim >= 0.55:
        return "Very likely the same person"
    elif sim >= 0.40:
        return "Possibly the same person"
    elif sim >= 0.30:
        return "Unlikely the same person"
    else:
        return "Different people"
    

def main():
    if len(argv) < 5:
        print("Usage: python3 pipeline.py <face_detector.tflite> <embed_model.tflite> <img1> <img2>")
        exit(0)
    
    detector_model_path = argv[1]
    embed_model_path = argv[2]
    img1_path = argv[3]
    img2_path = argv[4]

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("Failed to load images. Check paths.")
        exit(1)
    
    # Detection
    detector = FaceDetector(detector_model_path)
    # Embedding
    embedder = Embeddings(embed_model_path)
    # Communication
    notifier = NtfyNotifier(topic="home-door-83f9a2")

    cropped_face1 = detector.detect_and_crop_largest_face_tasks(img1, expand=0.0)
    cropped_face2 = detector.detect_and_crop_largest_face_tasks(img2, expand=0.0)

    emb1 = embedder.get_embedding(cropped_face1, normalization="arcface")
    emb2 = embedder.get_embedding(cropped_face2, normalization="arcface")

    detector.close()

    sim = cosine_sim_from_l2(emb1, emb2)
    print("Cosine similarity:", sim)
    print("Interpretation:", interpret_similarity(sim))

    notifier.send(
        title="Test notification",
        message="Hello from Raspberry Pi!",
        priority=4 # 4 = high priority
    )

if __name__ == "__main__":
    main()

    
