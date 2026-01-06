import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


class FaceDetector:
    def __init__(self, model_path:str):
        self.model_path = model_path
        self.detector = self._load_model()
    
    def _load_model(self):
        """
        Creates a MediaPipe Tasks FaceDetector.
        detector_model_path should point to a FaceDetector task model .tflite
        (e.g., BlazeFace short/long range task model).
        """
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        return vision.FaceDetector.create_from_options(options)

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
  
    def detect_and_crop_largest_face_tasks(self, image_bgr, expand=0.0): # expand=0.20
        """
        Detect faces using MediaPipe Tasks FaceDetector and return the largest face crop.
        """
        if self.detector is None:
            raise RuntimeError("Detector is closed. Create a new FaceDetector instance.")
        
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)

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

        x0 = self.clamp(x - pad_x, 0, w - 1)
        y0 = self.clamp(y - pad_y, 0, h - 1)
        x1 = self.clamp(x + bw + pad_x, 0, w)
        y1 = self.clamp(y + bh + pad_y, 0, h)

        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            raise ValueError("Face crop failed (empty).")

        return crop