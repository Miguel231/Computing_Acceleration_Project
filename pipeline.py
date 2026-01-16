import cv2
import numpy as np
import os, time

from face_detection import FaceDetector
from embeddings import Embeddings
from comunication import NtfyNotifier
from utils import load_db


def main():
    OUT_DIR = "/home/miserasp/Desktop/projecte/shared/"
    SLOTS = 10
    THRESH = 0.8 # similarity threshold
    EXPAND = 0.2

    detector_model_path = "/home/miserasp/Desktop/projecte/Computing_Acceleration_Project/data_projecte/detector.tflite"
    embed_model_path = "/home/miserasp/Desktop/projecte/Computing_Acceleration_Project/data_projecte/mobilefacenet_int8.tflite"
    db_embs_path = "/home/miserasp/Desktop/projecte/Computing_Acceleration_Project/data_projecte/storage.npz"

    detector = FaceDetector(detector_model_path)
    embedder = Embeddings(embed_model_path)
    notifier = NtfyNotifier(topic="home-door-83f9a2")

    names, db_embs = load_db(db_embs_path)  # names: list, db_embs: (N,D)

    last_alert_time = 0.0
    ALERT_COOLDOWN_SEC = 10  # prevent spam

    try:
        while True:
            did_work = False

            for slot in range(1, SLOTS + 1):
                path = os.path.join(OUT_DIR, f"slot_{slot:02d}.jpg")
                if not os.path.exists(path):
                    continue

                img = cv2.imread(path)
                if img is None:
                    time.sleep(0.01)
                    continue
                
                start_t = time.perf_counter()

                try:
                    cropped_face = detector.detect_and_crop_largest_face_tasks(img, expand=EXPAND)
                    emb_face = embedder.get_embedding(cropped_face, normalization="arcface")

                    # Best match against DB
                    scores = db_embs @ emb_face
                    best_i = int(np.argmax(scores))
                    best_sim = float(scores[best_i])
                    best_name = names[best_i]

                    if best_sim < THRESH:
                        now = time.time()
                        if now - last_alert_time >= ALERT_COOLDOWN_SEC:
                            notifier.send(
                                title="Unknown person detected!",
                                message=f"Unknown at door. Best similarity: {best_sim:.4f}",
                                priority=4
                            )
                            last_alert_time = now
                        print(f"UNKNOWN (best_sim={best_sim:.4f})")
                    else:
                        print(f"KNOWN: {best_name} (sim={best_sim:.4f})")

                except Exception as e:
                    print("Error processing", path, ":", e)

                end_t = time.perf_counter()
                elapsed_ms = (end_t - start_t) * 1000
                print(f"Processed in {elapsed_ms:.2f} ms")
                # delete image after processing
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

                did_work = True

            if not did_work:
                time.sleep(0.05)

    finally:
        detector.close()


if __name__ == "__main__":
    main()
