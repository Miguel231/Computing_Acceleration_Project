# import tensorflow as tf
import tflite_runtime.interpreter as tflite


import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sys import argv, exit


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def create_mediapipe_tasks_face_detector(detector_model_path: str):
    """
    Creates a MediaPipe Tasks FaceDetector.
    detector_model_path should point to a FaceDetector task model .tflite
    (e.g., BlazeFace short/long range task model).
    """
    base_options = python.BaseOptions(model_asset_path=detector_model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.FaceDetector.create_from_options(options)


def detect_and_crop_largest_face_tasks(detector, image_bgr, expand=0.20):
    """
    Detect faces using MediaPipe Tasks FaceDetector and return the largest face crop.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if not result.detections:
        raise ValueError("No face detected in the image (MediaPipe Tasks).")

    best = None
    best_area = 0

    for det in result.detections:
        bbox = det.bounding_box  # absolute pixels
        x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        if bw <= 0 or bh <= 0:
            continue
        area = bw * bh
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)

    if best is None:
        raise ValueError("Faces found but none were valid.")

    x, y, bw, bh = best

    pad_x = int(bw * expand)
    pad_y = int(bh * expand)

    x0 = clamp(x - pad_x, 0, w - 1)
    y0 = clamp(y - pad_y, 0, h - 1)
    x1 = clamp(x + bw + pad_x, 0, w)
    y1 = clamp(y + bh + pad_y, 0, h)

    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Face crop failed (empty).")

    return crop


def preprocess_for_tflite(face_bgr, input_details, normalization="arcface"):
    """
    normalization:
      - "arcface": (x - 127.5) / 128.0   -> approx [-1, 1]
      - "0_1": x / 255.0
    """
    shape = input_details[0]["shape"]  # usually [1,H,W,3] (NHWC)
    dtype = input_details[0]["dtype"]

    if len(shape) != 4:
        raise ValueError(f"Unexpected model input shape: {shape}")

    # Infer layout
    if shape[3] == 3:  # NHWC
        in_h, in_w = int(shape[1]), int(shape[2])
        channel_last = True
    elif shape[1] == 3:  # NCHW (rare for TFLite)
        in_h, in_w = int(shape[2]), int(shape[3])
        channel_last = False
    else:
        raise ValueError(f"Cannot infer channel position from input shape: {shape}")

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    x = face_resized.astype(np.float32)

    if normalization == "arcface":
        x = (x - 127.5) / 128.0
    elif normalization == "0_1":
        x = x / 255.0
    else:
        raise ValueError("normalization must be 'arcface' or '0_1'")

    if channel_last:
        x = np.expand_dims(x, axis=0)  # [1,H,W,3]
    else:
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)  # [1,3,H,W]

    # Quantize if needed
    if dtype == np.float32:
        return x.astype(np.float32)

    if dtype == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        if scale == 0:
            raise ValueError("Model is int8 but quantization params invalid (scale=0).")
        q = x / scale + zero_point
        q = np.clip(np.round(q), -128, 127).astype(np.int8)
        return q

    raise ValueError(f"Unsupported input dtype: {dtype}")


def dequantize_output_if_needed(out, output_details):
    dtype = output_details[0]["dtype"]
    if dtype == np.float32:
        return out.astype(np.float32)

    if dtype == np.int8:
        scale, zero_point = output_details[0]["quantization"]
        if scale == 0:
            raise ValueError("Output is int8 but quantization params invalid (scale=0).")
        return (out.astype(np.float32) - zero_point) * scale

    raise ValueError(f"Unsupported output dtype: {dtype}")


def l2_normalize(v, eps=1e-10):
    v = v.astype(np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + eps)


def extract_embedding(detector, image_bgr, interpreter, input_details, output_details,
                      save_debug_path=None, normalization="arcface"):
    # Use Tasks detector to crop face
    face_crop = detect_and_crop_largest_face_tasks(detector, image_bgr)

    if save_debug_path:
        cv2.imwrite(save_debug_path, face_crop)

    inp = preprocess_for_tflite(face_crop, input_details, normalization=normalization)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]["index"])
    out = dequantize_output_if_needed(out, output_details)

    emb = l2_normalize(out)
    return emb


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

    # TFLite embedder (MobileFaceNet)
    # interpreter = tf.lite.Interpreter(model_path=embed_model_path, num_threads=2)
    interpreter = tflite.Interpreter(model_path=embed_model_path, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # MediaPipe Tasks face detector
    detector = create_mediapipe_tasks_face_detector(detector_model_path)

    try:
        emb1 = extract_embedding(detector, img1, interpreter, input_details, output_details,
                                 save_debug_path="face1.jpg", normalization="arcface")
        emb2 = extract_embedding(detector, img2, interpreter, input_details, output_details,
                                 save_debug_path="face2.jpg", normalization="arcface")
    finally:
        detector.close()

    sim = cosine_sim_from_l2(emb1, emb2)
    print("Cosine similarity:", sim)
    print("Interpretation:", interpret_similarity(sim))


if __name__ == "__main__":
    main()
