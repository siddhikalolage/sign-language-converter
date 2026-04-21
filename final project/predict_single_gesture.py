import argparse
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from gesture_pipeline import (
    DEFAULT_MIN_LANDMARK_QUALITY,
    EXPECTED_FEATURE_COUNT,
    create_holistic_landmarker,
    detect_holistic_landmarks,
    draw_landmarks,
    extract_landmarks,
    format_label,
    has_landmark_signal,
    landmark_quality_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FACE_FEATURE_COUNT = 18
HAND_FEATURE_COUNT = 63


class SpeechWorker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._backend = None
        self._failed = False

        if not enabled:
            return

        try:
            import pyttsx3  # noqa: F401

            self._backend = "pyttsx3"
        except Exception:
            try:
                import winsound  # noqa: F401

                self._backend = "winsound"
            except Exception:
                self._failed = True
                return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        engine = None
        winsound_module = None

        try:
            if self._backend == "pyttsx3":
                import pyttsx3

                engine = pyttsx3.init()
            elif self._backend == "winsound":
                import winsound

                winsound_module = winsound
        except Exception:
            self._failed = True
            return

        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break

            try:
                if self._backend == "pyttsx3":
                    engine.say(text)
                    engine.runAndWait()
                elif self._backend == "winsound":
                    print(f"[audio] {text}")
                    winsound_module.MessageBeep()
            except Exception:
                self._failed = True
            finally:
                self._queue.task_done()

        if engine is not None:
            engine.stop()

    def say(self, text: str) -> None:
        if self.enabled and not self._failed:
            self._queue.put(text)

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one isolated gesture at a time from the webcam."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--classifier-path",
        default=str(PROJECT_ROOT / "text_phrase_image_model.pkl"),
        help="Path to the trained sklearn gesture model.",
    )
    parser.add_argument(
        "--label-encoder-path",
        default=str(PROJECT_ROOT / "text_phrase_image_label_encoder.pkl"),
        help="Path to the fitted label encoder.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.28,
        help="Only allow capture when the averaged class confidence reaches this threshold.",
    )
    parser.add_argument(
        "--min-landmark-quality",
        type=float,
        default=min(DEFAULT_MIN_LANDMARK_QUALITY, 0.35),
        help="Ignore frames where tracking quality is below this threshold.",
    )
    parser.add_argument(
        "--stability-frames",
        type=int,
        default=3,
        help="Require this many stable high-quality frames before a capture is considered ready.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=6,
        help="Average prediction probabilities over this many recent frames.",
    )
    parser.add_argument(
        "--max-motion",
        type=float,
        default=0.035,
        help="Treat frames above this motion level as unstable.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.06,
        help="Require the best class to beat the runner-up by at least this margin.",
    )
    parser.add_argument(
        "--consensus-window",
        type=int,
        default=8,
        help="How many recent stable frame labels to consider before committing a word.",
    )
    parser.add_argument(
        "--consensus-ratio",
        type=float,
        default=0.55,
        help="Portion of recent stable labels that must agree before a word is shown.",
    )
    parser.add_argument(
        "--release-frames",
        type=int,
        default=4,
        help="Require this many weak frames before unlocking for the next word.",
    )
    parser.add_argument(
        "--show-landmarks",
        action="store_true",
        help="Draw debug landmarks on top of the camera feed.",
    )
    parser.add_argument(
        "--debug-ui",
        action="store_true",
        help="Show tracking and confidence details on screen.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak the captured word.",
    )
    return parser.parse_args()


def canonicalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3).copy()
    mask = np.any(np.abs(points) > 1e-6, axis=1)
    if not np.any(mask):
        return points.reshape(-1)

    face = points[: FACE_FEATURE_COUNT // 3]
    face_mask = mask[: FACE_FEATURE_COUNT // 3]
    center = None
    scale = 1.0

    if np.any(face_mask):
        visible_face = face[face_mask]
        center = visible_face.mean(axis=0, keepdims=True)
        if face_mask[1] and face_mask[4]:
            scale = float(np.linalg.norm(face[1, :2] - face[4, :2]))

    if center is None:
        visible_points = points[mask]
        center = visible_points.mean(axis=0, keepdims=True)

    if scale < 1e-6:
        visible_points = points[mask]
        point_extent = visible_points.max(axis=0) - visible_points.min(axis=0)
        scale = float(max(point_extent[0], point_extent[1], 1.0))

    points[mask] = (points[mask] - center) / max(scale, 1e-6)
    return points.reshape(-1)


def compute_motion(previous_landmarks: np.ndarray | None, current_landmarks: np.ndarray) -> float:
    if previous_landmarks is None:
        return 0.0
    previous_landmarks = np.asarray(previous_landmarks, dtype=np.float32)
    current_landmarks = np.asarray(current_landmarks, dtype=np.float32)
    return float(np.mean(np.abs(current_landmarks - previous_landmarks)))


def top_prediction_with_margin(probabilities: np.ndarray) -> tuple[int, float, float]:
    ordered = np.argsort(probabilities)[::-1]
    top_index = int(ordered[0])
    top_confidence = float(probabilities[top_index])
    runner_up = float(probabilities[int(ordered[1])]) if len(ordered) > 1 else 0.0
    return top_index, top_confidence, top_confidence - runner_up


def draw_prediction_only(frame, prediction: str) -> None:
    if not prediction:
        return

    label = format_label(prediction)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    frame_height, frame_width = frame.shape[:2]
    x = max(20, (frame_width - text_width) // 2)
    y = max(text_height + 20, frame_height - 40)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 20, y - text_height - 20),
        (x + text_width + 20, y + baseline + 18),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    classifier_path = Path(args.classifier_path)
    label_encoder_path = Path(args.label_encoder_path)

    model = joblib.load(classifier_path)
    label_encoder = joblib.load(label_encoder_path)
    expected_model_features = getattr(model, "n_features_in_", EXPECTED_FEATURE_COUNT)
    if expected_model_features != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Model expects {expected_model_features} features, but this pipeline extracts "
            f"{EXPECTED_FEATURE_COUNT}. Use a classifier trained from the current landmarks."
        )

    landmarker, resolved_model_path = create_holistic_landmarker(running_mode="video")
    print(f"[info] Using classifier: {classifier_path}")
    print(f"[info] Using holistic model: {resolved_model_path}")
    print("[info] Hold one gesture steady. A word appears only after stable consensus.")
    print("[info] Press C to clear the current word, Q to quit.")

    speech_worker = SpeechWorker(enabled=args.speak)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    probability_window: deque[np.ndarray] = deque(maxlen=max(1, args.smoothing_window))
    landmark_window: deque[np.ndarray] = deque(maxlen=max(1, args.smoothing_window))
    label_window: deque[str] = deque(maxlen=max(1, args.consensus_window))
    previous_signature: np.ndarray | None = None
    stable_frames = 0
    release_frames = 0
    candidate_label = ""
    candidate_confidence = 0.0
    candidate_margin = 0.0
    current_prediction = ""
    locked_prediction = ""
    status_text = "Show one gesture and hold it"

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            results = detect_holistic_landmarks(
                landmarker,
                frame,
                frame_timestamp_ms=timestamp_ms,
            )
            if args.show_landmarks:
                draw_landmarks(frame, results)

            landmarks = extract_landmarks(results)
            has_signal = has_landmark_signal(landmarks)
            tracking_quality = landmark_quality_score(results) if has_signal else 0.0
            landmark_array = np.asarray(landmarks, dtype=np.float32)
            landmark_signature = canonicalize_landmarks(landmark_array) if has_signal else None
            motion = (
                compute_motion(previous_signature, landmark_signature)
                if landmark_signature is not None
                else 0.0
            )
            previous_signature = landmark_signature

            ready = False
            if has_signal and tracking_quality >= args.min_landmark_quality and motion <= args.max_motion:
                landmark_window.append(landmark_array)
                smoothed_landmarks = np.median(np.asarray(landmark_window, dtype=np.float32), axis=0)
                probabilities = model.predict_proba([smoothed_landmarks.tolist()])[0]
                probability_window.append(probabilities)
                averaged = np.mean(probability_window, axis=0)
                prediction, candidate_confidence, candidate_margin = top_prediction_with_margin(averaged)
                candidate_label = str(label_encoder.inverse_transform([prediction])[0])
                if candidate_margin >= args.min_margin:
                    label_window.append(candidate_label)
                else:
                    label_window.append("")
                stable_frames += 1
                release_frames = 0
                if locked_prediction:
                    current_prediction = locked_prediction
                    status_text = "Hold until release"
                else:
                    required_votes = max(
                        1,
                        math.ceil(len(label_window) * min(max(args.consensus_ratio, 0.0), 1.0)),
                    )
                    consensus_hits = sum(1 for label in label_window if label == candidate_label)
                    ready = (
                        stable_frames >= args.stability_frames
                        and candidate_confidence >= args.min_confidence
                        and candidate_margin >= args.min_margin
                        and consensus_hits >= required_votes
                    )
                    status_text = "Stable" if ready else "Hold steady"
                    if ready and candidate_label:
                        current_prediction = candidate_label
                        locked_prediction = candidate_label
                        print(
                            f"[done] Predicted: {candidate_label} "
                            f"({candidate_confidence * 100:.1f}%)"
                        )
                        if args.speak:
                            speech_worker.say(format_label(candidate_label))
            else:
                probability_window.clear()
                landmark_window.clear()
                label_window.clear()
                stable_frames = 0
                candidate_label = ""
                candidate_confidence = 0.0
                candidate_margin = 0.0
                if has_signal:
                    status_text = "Hold steadier"
                else:
                    status_text = "Show one gesture"
                if locked_prediction:
                    release_frames += 1
                    if release_frames >= max(1, args.release_frames):
                        locked_prediction = ""
                else:
                    release_frames = 0

            if args.debug_ui:
                cv2.putText(
                    frame,
                    f"Status: {status_text}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if ready else (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Candidate: {format_label(candidate_label) if candidate_label else '-'}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {candidate_confidence * 100:.0f}%",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Margin: {candidate_margin * 100:.0f}%",
                    (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Tracking: {tracking_quality * 100:.0f}%",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Stable: {stable_frames}",
                    (10, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )
            else:
                draw_prediction_only(frame, current_prediction)

            cv2.imshow("Single Gesture Prediction", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                current_prediction = ""
                locked_prediction = ""
                release_frames = 0
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_worker.close()
        landmarker.close()


if __name__ == "__main__":
    main()
