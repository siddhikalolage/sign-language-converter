import argparse
import json
import os
import time
from pathlib import Path

import cv2


def configure_ultralytics_workspace() -> Path:
    config_dir = Path(".ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one phrase at a time from the webcam using a YOLO classification model."
    )
    parser.add_argument(
        "--model-path",
        default="text_phrase_yolo_cls.pt",
        help="Path to the trained YOLO classification checkpoint.",
    )
    parser.add_argument(
        "--class-map",
        default="yolo_phrase_dataset/class_name_map.json",
        help="JSON mapping from sanitized class names back to user-facing phrases.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Only accept predictions above this confidence threshold.",
    )
    parser.add_argument(
        "--predict-every-ms",
        type=int,
        default=250,
        help="Run the classifier at most once this often.",
    )
    return parser.parse_args()


def load_class_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    configure_ultralytics_workspace()
    from ultralytics import YOLO

    class_map = load_class_map(Path(args.class_map))
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    last_predict_at = 0.0
    candidate_label = ""
    candidate_confidence = 0.0
    last_captured = ""
    status_text = "Show one phrase and press SPACE to confirm"

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            if (now - last_predict_at) * 1000.0 >= max(1, args.predict_every_ms):
                last_predict_at = now
                results = model.predict(frame, verbose=False)
                if results:
                    probs = results[0].probs
                    if probs is not None:
                        top_index = int(probs.top1)
                        top_conf = float(probs.top1conf.item())
                        raw_label = str(results[0].names[top_index])
                        candidate_label = class_map.get(raw_label, raw_label).replace("_", " ")
                        candidate_confidence = top_conf
                        status_text = (
                            "Ready: press SPACE"
                            if top_conf >= args.min_confidence
                            else "Hold steadier"
                        )

            cv2.putText(
                frame,
                f"Candidate: {candidate_label or '-'}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Confidence: {candidate_confidence * 100:.0f}%",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Status: {status_text}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if last_captured:
                cv2.putText(
                    frame,
                    f"Last: {last_captured}",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("YOLO Phrase Prediction", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if candidate_label and candidate_confidence >= args.min_confidence:
                    last_captured = candidate_label
                    print(f"[done] Captured: {candidate_label} ({candidate_confidence * 100:.1f}%)")
                else:
                    print("[warn] Confidence is too low. Try holding the phrase more clearly.")
            elif key == ord("c"):
                last_captured = ""
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
