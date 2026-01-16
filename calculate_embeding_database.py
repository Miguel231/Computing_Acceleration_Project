from sys import argv, exit
import os
import cv2
import numpy as np

from face_detection import FaceDetector
from embeddings import Embeddings


from utils import load_db


def save_db(npz_path: str, names, embs):
    np.savez(npz_path, names=np.array(names), embs=embs.astype(np.float32))


def main():
    if len(argv) < 6:
        print("Usage:")
        print("  python3 enroll.py <face_detector.tflite> <embed_model.tflite> <storage.npz> <label> <img1> [img2 ... imgN]")
        exit(0)

    detector_model_path = argv[1]
    embed_model_path = argv[2]
    db_path = argv[3]
    label = argv[4]
    image_paths = argv[5:]

    detector = FaceDetector(detector_model_path)
    embedder = Embeddings(embed_model_path)

    # Load existing DB (if any)
    names, embs = load_db(db_path)
    new_embs = []

    try:
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[SKIP] Could not read image: {p}")
                continue

            try:
                crop = detector.detect_and_crop_largest_face_tasks(img, expand=0.2)
                emb = embedder.get_embedding(crop, normalization="arcface")  # already L2-normalized
                # the normalization is done inside get_embedding
                new_embs.append(emb)
                print(f"[OK] Added embedding for '{label}' from: {p}")
            except Exception as e:
                print(f"[SKIP] {p} -> {e}")

        if not new_embs:
            print("No embeddings were added. Nothing to save.")
            return

        new_embs = np.vstack(new_embs).astype(np.float32)  # (K,D)

        if embs is None:
            # first time creating DB
            embs = new_embs
            names = [label] * new_embs.shape[0]
        else:
            # append to existing
            embs = np.vstack([embs, new_embs]).astype(np.float32)
            names.extend([label] * new_embs.shape[0])

        save_db(db_path, names, embs)
        print(f"Saved {len(new_embs)} new embeddings to: {db_path}")
        print(f"DB now contains {len(names)} embeddings total.")

    finally:
        detector.close()


if __name__ == "__main__":
    main()
