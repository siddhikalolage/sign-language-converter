import base64
import os
from pathlib import Path
from threading import Lock

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gesture_pipeline import (
    create_face_hands_landmarker,
    detect_holistic_landmarks,
    extract_landmarks,
    format_label,
    has_landmark_signal,
    landmark_quality_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"

LANDMARK_MODEL_PATH = PROJECT_ROOT / "text_phrase_image_model.pkl"
LANDMARK_ENCODER_PATH = PROJECT_ROOT / "text_phrase_image_label_encoder.pkl"
YOLO_MODEL_PATH = PROJECT_ROOT / "text_phrase_yolo_cls.pt"
YOLO_CLASS_MAP_PATH = PROJECT_ROOT / "yolo_phrase_dataset" / "class_name_map.json"
LANDMARK_FALLBACK_MIN_CONFIDENCE = 0.30
LANDMARK_FALLBACK_MIN_MARGIN = 0.08
LANDMARK_FALLBACK_MIN_QUALITY = 0.45


class PredictRequest(BaseModel):
    image_data: str
    mode: str = "landmark"


class LandmarkRuntime:
    def __init__(self) -> None:
        self.model = joblib.load(LANDMARK_MODEL_PATH)
        self.label_encoder = joblib.load(LANDMARK_ENCODER_PATH)
        self.landmarker = create_face_hands_landmarker(static_image_mode=True)
        self.lock = Lock()

    def predict(self, frame: np.ndarray) -> dict:
        with self.lock:
            results = detect_holistic_landmarks(self.landmarker, frame)

        landmarks = extract_landmarks(results)
        if not has_landmark_signal(landmarks):
            raise HTTPException(status_code=422, detail="No face or hand landmarks were detected.")

        quality = landmark_quality_score(results)
        probabilities = self.model.predict_proba([landmarks])[0]
        ordered = np.argsort(probabilities)[::-1]
        top_index = int(ordered[0])
        confidence = float(probabilities[top_index])
        runner_up = float(probabilities[int(ordered[1])]) if len(ordered) > 1 else 0.0
        label = str(self.label_encoder.inverse_transform([top_index])[0])
        return {
            "label": format_label(label),
            "confidence": confidence,
            "quality": quality,
            "margin": confidence - runner_up,
            "backend": "landmark",
        }


class YoloRuntime:
    def __init__(self) -> None:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR.resolve()))
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "YOLO runtime is unavailable. Install the web requirements to use YOLO mode."
            ) from exc

        self.model = YOLO(str(YOLO_MODEL_PATH))
        if YOLO_CLASS_MAP_PATH.exists():
            import json

            self.class_map = json.loads(YOLO_CLASS_MAP_PATH.read_text(encoding="utf-8"))
        else:
            self.class_map = {}
        self.lock = Lock()

    def predict(self, frame: np.ndarray) -> dict:
        with self.lock:
            results = self.model.predict(frame, verbose=False)

        if not results or results[0].probs is None:
            raise HTTPException(status_code=422, detail="YOLO could not classify the image.")

        probs = results[0].probs
        top_index = int(probs.top1)
        confidence = float(probs.top1conf.item())
        raw_label = str(results[0].names[top_index])
        label = self.class_map.get(raw_label, raw_label).replace("_", " ")
        return {
            "label": label,
            "confidence": confidence,
            "quality": None,
            "margin": None,
            "backend": "yolo",
        }


class RuntimeRegistry:
    def __init__(self) -> None:
        self._landmark: LandmarkRuntime | None = None
        self._yolo: YoloRuntime | None = None

    @property
    def landmark(self) -> LandmarkRuntime:
        if self._landmark is None:
            self._landmark = LandmarkRuntime()
        return self._landmark

    @property
    def yolo(self) -> YoloRuntime:
        if self._yolo is None:
            self._yolo = YoloRuntime()
        return self._yolo


def decode_image(image_data: str) -> np.ndarray:
    payload = image_data
    if "," in image_data:
        _, payload = image_data.split(",", 1)
    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image data.")
    return frame


def should_fallback_to_yolo(result: dict) -> bool:
    quality = float(result.get("quality") or 0.0)
    confidence = float(result.get("confidence") or 0.0)
    margin = float(result.get("margin") or 0.0)
    return (
        quality < LANDMARK_FALLBACK_MIN_QUALITY
        or confidence < LANDMARK_FALLBACK_MIN_CONFIDENCE
        or margin < LANDMARK_FALLBACK_MIN_MARGIN
    )


runtime = RuntimeRegistry()
app = FastAPI(title="Sign Language Converter")
app.mount("/web", StaticFiles(directory=WEB_ROOT), name="web")


@app.get("/")
def serve_index():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
def health_check():
    yolo_runtime_ready = True
    try:
        import ultralytics  # noqa: F401
    except Exception:
        yolo_runtime_ready = False

    return {
        "status": "ok",
        "landmark_model": LANDMARK_MODEL_PATH.exists(),
        "yolo_model": YOLO_MODEL_PATH.exists(),
        "yolo_runtime_ready": yolo_runtime_ready,
    }


@app.post("/api/predict")
def predict(payload: PredictRequest):
    try:
        frame = decode_image(payload.image_data)
        mode = payload.mode.strip().lower()

        if mode == "landmark":
            result = runtime.landmark.predict(frame)
            if should_fallback_to_yolo(result):
                try:
                    yolo_result = runtime.yolo.predict(frame)
                except Exception:
                    pass
                else:
                    if yolo_result["confidence"] >= max(result["confidence"] + 0.10, 0.55):
                        result = {
                            **yolo_result,
                            "requested_mode": "landmark",
                            "backend": "yolo-fallback",
                        }
        elif mode == "yolo":
            result = runtime.yolo.predict(frame)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {payload.mode}")

        return {
            "mode": result.get("backend", mode),
            "requested_mode": mode,
            "label": result["label"],
            "confidence": result["confidence"],
            "quality": result["quality"],
            "margin": result.get("margin"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


def main() -> None:
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
