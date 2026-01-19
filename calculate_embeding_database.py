from sys import argv, exit
import os
import cv2
import numpy as np

from face_detection import FaceDetector
from embeddings import Embeddings
from utils import load_db


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_db(npz_path: str, names, embs):
    np.savez(npz_path, names=np.array(names), embs=embs.astype(np.float32))


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def iter_image_paths(folder: str, recursive: bool = True):
    if recursive:
        for root, _, files in os.walk(folder):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    yield os.path.join(root, fn)
    else:
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            if os.path.isfile(p):
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    yield p


def main():
    if len(argv) < 6:
        print("Usage:")
        print("  python3 enroll_folder.py <face_detector.tflite> <embed_model.tflite> <storage.npz> <label> <folder>")
        exit(0)

    detector_model_path = argv[1]
    embed_model_path = argv[2]
    db_path = argv[3]
    label = argv[4]
    folder = argv[5]

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        exit(1)

    detector = FaceDetector(detector_model_path)
    embedder = Embeddings(embed_model_path)

    # Load existing DB (if any)
    names, embs = load_db(db_path)

    collected = []
    total = 0
    used = 0

    try:
        for p in iter_image_paths(folder, recursive=True):
            total += 1
            img = cv2.imread(p)
            if img is None:
                print(f"[SKIP] Could not read image: {p}")
                continue

            try:
                crop = detector.detect_and_crop_largest_face_tasks(img, expand=0.0)
                emb = embedder.get_embedding(crop, normalization="arcface")  # typically already L2-normalized
                collected.append(emb.astype(np.float32))
                used += 1
                print(f"[OK] {label}: {p}")
            except Exception as e:
                print(f"[SKIP] {p} -> {e}")

        if not collected:
            print("No embeddings were added. Nothing to save.")
            return

        # (K, D)
        collected = np.vstack(collected).astype(np.float32)

        # Average template, then L2-normalize so it matches your cosine/L2 comparisons well
        template = collected.mean(axis=0)
        template = l2_normalize(template).astype(np.float32)  # (D,)

        template = template.reshape(1, -1)  # (1, D)

        if embs is None:
            embs = template
            names = [label]
        else:
            embs = np.vstack([embs, template]).astype(np.float32)
            names.append(label)

        save_db(db_path, names, embs)

        print("\n=== Summary ===")
        print(f"Scanned images: {total}")
        print(f"Used images:    {used}")
        print(f"Saved 1 template embedding for label '{label}' to: {db_path}")
        print(f"DB now contains {len(names)} identities/templates total.")

    finally:
        detector.close()


if __name__ == "__main__":
    main()

