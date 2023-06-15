import cv2


class VideoStream:
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()
        _, im = cv2.imencode('.jpg', frame)
        return im.tobytes()
