from config import STREAM_URL_002, app
from flask import Response, render_template
from image_processing import VideoStream


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(
        gen(VideoStream(STREAM_URL_002)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


video = VideoStream(STREAM_URL_002)
if __name__ == "__main__":
    app.run(debug=True)
