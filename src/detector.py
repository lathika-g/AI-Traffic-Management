"""YOLO-based vehicle detector."""
import cv2
import numpy as np

from src import config

# BGR colours for each class
_COLOURS = {
    "car": (0, 200, 0),
    "motorcycle": (0, 165, 255),
    "bus": (255, 120, 0),
    "truck": (0, 0, 255),
}


class VehicleDetector:
    """Loads a YOLO model and detects cars, motorcycles, buses and trucks."""

    def __init__(self, model_name: str = config.MODEL_NAME,
                 confidence: float = config.CONFIDENCE):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The 'ultralytics' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        try:
            self.model = YOLO(model_name)
        except Exception as exc:  # download or file errors
            raise RuntimeError(
                f"Could not load YOLO model '{model_name}'. "
                "The first run needs internet to download the weights. "
                f"Details: {exc}"
            ) from exc

        self.confidence = confidence

    def detect(self, frame: np.ndarray):
        """Detect vehicles in one BGR frame.

        Returns:
            annotated_frame: copy of the frame with boxes drawn
            counts: dict such as {"car": 5, "motorcycle": 2, "bus": 0, "truck": 1}
        """
        counts = {name: 0 for name in config.VEHICLE_CLASSES.values()}
        annotated = frame.copy()

        result = self.model.predict(
            frame,
            classes=list(config.VEHICLE_CLASSES.keys()),
            conf=self.confidence,
            verbose=False,
        )[0]

        for box in result.boxes:
            class_id = int(box.cls[0])
            name = config.VEHICLE_CLASSES.get(class_id)
            if name is None:
                continue
            counts[name] += 1

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            colour = _COLOURS[name]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(annotated, f"{name} {conf:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        return annotated, counts
