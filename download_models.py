"""Download the OpenCV Zoo face models (YuNet detector + SFace recognizer)."""
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

out = Path(__file__).parent / "models"
out.mkdir(exist_ok=True)
for name, url in MODELS.items():
    dest = out / name
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"✓ {name} already present")
        continue
    print(f"↓ {name}")
    urllib.request.urlretrieve(url, dest)
print("done")
