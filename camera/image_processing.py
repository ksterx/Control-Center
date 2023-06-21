from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


class ImageProcessing:
    def __init__(self, type, url, model_name: Optional[str] = None):
        if type == "streaming":
            return Streaming(url)
        elif type == "object_detection":
            return ObjectDetection(url, model_name)
        elif type == "pose_estimation":
            return PoseEstimation(url, model_name)
        else:
            raise ValueError(f"Unknown type: {type}")


class Streaming(ABC):
    def __init__(self, url, model_name: Optional[str] = None):
        self.video = cv2.VideoCapture(url)
        if model_name is None:
            self.model = None

    def __del__(self):
        self.video.release()

    @abstractmethod
    def get_frame(self):
        _, results = self.video.read()
        img = results
        _, im = cv2.imencode(".jpg", img)
        return im.tobytes()


class ObjectDetection(Streaming):
    YOLO_DIR = "camera/nn/YOLO/"
    THRESHOLD = 0.4

    def __init__(self, url, model_name: str):
        super().__init__(url)
        if model_name.startswith("YOLO"):
            self.model = YOLO(self.YOLO_DIR + model_name.lower() + ".pt")
            self.names = self.model.names
            self.colors = np.random.randint(0, 255, size=(len(self.names), 3), dtype="uint8")
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def get_frame(self):
        _, results = self.video.read()
        results = self.model.predict(source=results)
        img = results[0].orig_img
        for x1, y1, x2, y2, conf, cls in results[0].boxes.data:
            x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)

            if conf > self.THRESHOLD:
                color = tuple(self.colors[cls].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cls_name = self.names[int(cls)]
                cv2.putText(
                    img,
                    f"{cls_name} {conf:.2f}",
                    (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

        _, im = cv2.imencode(".jpg", img)
        return im.tobytes()


class PoseEstimation(Streaming):
    def get_frame(self):
        super().get_frame()
