"""Face detection + recognition built on OpenCV's DNN models.

- Detector:   YuNet  (face_detection_yunet_2023mar.onnx)
- Recognizer: SFace  (face_recognition_sface_2021dec.onnx) -> 128-d embedding

Both run on CPU through opencv-python; no dlib / CUDA needed.
"""
from __future__ import annotations

import base64
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

MAX_SIDE = 1280  # downscale big uploads before detection


@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]  # x, y, w, h (in original image coords)
    confidence: float
    embedding: np.ndarray  # L2-normalised 128-d float32
    raw: np.ndarray  # YuNet row (15 values) in working-image coords


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


class FaceEngine:
    def __init__(self, det_threshold: float = 0.8):
        missing = [p.name for p in (DETECTOR_PATH, RECOGNIZER_PATH) if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Missing model files in {MODELS_DIR}: {missing}. Run `python download_models.py`."
            )
        self._detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH), "", (320, 320), det_threshold, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")
        self._lock = threading.Lock()  # OpenCV DNN objects are not thread-safe

    def analyse(self, img: np.ndarray) -> list[DetectedFace]:
        h, w = img.shape[:2]
        with self._lock:
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(img)
            if faces is None:
                return []
            out: list[DetectedFace] = []
            for row in faces:
                aligned = self._recognizer.alignCrop(img, row)
                feat = self._recognizer.feature(aligned).flatten().astype(np.float32)
                feat /= np.linalg.norm(feat) + 1e-9
                x, y, bw, bh = (int(round(v)) for v in row[:4])
                out.append(DetectedFace((x, y, bw, bh), float(row[14]), feat, row))
        # largest face first
        out.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        return out


def portrait_data_uri(img: np.ndarray, box: tuple[int, int, int, int], size: int = 360) -> str:
    """Square crop around the face with some breathing room, as a JPEG data URI."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2.2
    side = max(w, h) * 1.9
    H, W = img.shape[:2]
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img
    ch, cw = crop.shape[:2]
    s = min(ch, cw)
    crop = crop[(ch - s) // 2 : (ch - s) // 2 + s, (cw - s) // 2 : (cw - s) // 2 + s]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class Gallery:
    """In-memory matrix of every enrolled embedding for fast cosine matching."""

    def __init__(self):
        self._lock = threading.Lock()
        self._owners: list[str] = []
        self._matrix = np.empty((0, 128), dtype=np.float32)

    def rebuild(self, students) -> int:
        owners, rows = [], []
        for doc in students:
            for emb in doc.get("embeddings", []):
                owners.append(str(doc["_id"]))
                rows.append(emb)
        with self._lock:
            self._owners = owners
            self._matrix = (
                np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 128), dtype=np.float32)
            )
        return len(owners)

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        with self._lock:
            if not self._owners:
                return None, 0.0
            sims = self._matrix @ embedding
            i = int(np.argmax(sims))
            return self._owners[i], float(sims[i])

    @property
    def size(self) -> int:
        return len(self._owners)
