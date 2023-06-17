import cv2
import numpy as np
from ultralytics import YOLO


class VideoStream:
    YOLO_DIR = "camera/nn/YOLO/"
    WEIGHTS_v8n = YOLO_DIR + "yolov8n.pt"
    WEIGHTS_v8s = YOLO_DIR + "yolov8s.pt"
    WEIGHTS_v8m = YOLO_DIR + "yolov8m.pt"
    WEIGHTS_v8l = YOLO_DIR + "yolov8l.pt"
    WEIGHTS_v8x = YOLO_DIR + "yolov8x.pt"
    THRESHOLD = 0.4

    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.model = YOLO(self.WEIGHTS_v8n)
        self.names = self.model.names
        self.colors = np.random.randint(0, 255, size=(len(self.names), 3), dtype="uint8")

    def __del__(self):
        self.video.release()

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
