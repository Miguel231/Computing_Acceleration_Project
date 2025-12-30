
# !! For laptop
import tensorflow as tf
# !! For resberry Pi
# import tflite_runtime.interpreter as tflite


import cv2
import numpy as np



class Embeddings:

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()
    
    def _load_model(self):
        # TFLite embedder (MobileFaceNet)
        # !! For laptop
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads=2)
        # !! For resberry Pi
        # interpreter = tflite.Interpreter(model_path=embed_model_path, num_threads=2)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def preprocess_for_tflite(self, face_bgr, normalization="arcface"):
        """
        normalization:
        - "arcface": (x - 127.5) / 128.0   -> approx [-1, 1]
        - "0_1": x / 255.0
        """
        shape = self.input_details[0]["shape"]  # usually [1,H,W,3] (NHWC)
        dtype = self.input_details[0]["dtype"]

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
        # We have to resize to the exact size expected by the model
        face_resized = cv2.resize(face_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

        # Normalize
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
            
            # Quantize parameters, these are learned during model training
            scale, zero_point = self.input_details[0]["quantization"]
            if scale == 0:
                raise ValueError("Model is int8 but quantization params invalid (scale=0).")
            q = x / scale + zero_point
            q = np.clip(np.round(q), -128, 127).astype(np.int8)
            return q

        raise ValueError(f"Unsupported input dtype: {dtype}")
    
    def dequantize_output_if_needed(self, out):
        dtype = self.output_details[0]["dtype"]
        if dtype == np.float32:
            return out.astype(np.float32)


        if dtype == np.int8:
            scale, zero_point = self.output_details[0]["quantization"]
            if scale == 0:
                raise ValueError("Output is int8 but quantization params invalid (scale=0).")
            return (out.astype(np.float32) - zero_point) * scale

        raise ValueError(f"Unsupported output dtype: {dtype}")

    @staticmethod
    def l2_normalize(v, eps=1e-10):
        v = v.astype(np.float32).reshape(-1)
        return v / (np.linalg.norm(v) + eps)


    def get_embedding(self, cropped_face, normalization="arcface"):

        if cropped_face is None or cropped_face.size == 0:
            raise ValueError("Empty face crop passed to get_embedding().")

        inp = self.preprocess_for_tflite(cropped_face, normalization=normalization)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()

        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        out = self.dequantize_output_if_needed(out)

        emb = self.l2_normalize(out)
        return emb








