from typing import Optional

from flask import Response, render_template, request

from camera import STREAM_URL_002, app

from .image_processing import ImageProcessing

processor = ImageProcessing("streaming", STREAM_URL_002)

proc_status = {
    "streaming": "true",
    "pose_estimation": "false",
    "object_detection": "false",
}

yolos = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]

options = {
    "model_name": None,
    "checked": "streaming",
    "unchecked": ["pose_estimation", "object_detection"],
}


@app.route("/", methods=["GET", "POST"])
def index():
    global processor

    if request.method == "POST":
        model_name = request.form["model_name"]
        options["model_name"] = model_name
        im_proc = request.form.get("im_proc")
        processor = ImageProcessing(im_proc, STREAM_URL_002, model_name)
        update_status(im_proc)
        return render_template("index.html", options=options, yolos=yolos, proc_status=proc_status)

    # When the user first visits the page
    else:
        update_status("streaming")
        return render_template("index.html", options=options, yolos=yolos, proc_status=proc_status)


@app.route("/stream")
def stream():
    return Response(
        gen(processor),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def gen(procs):
    while True:
        frame = processor.get_frame(procs)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


def update_status(proc):
    unchecked = []
    for p in proc_status:
        if p == proc:
            proc_status[p] = "true"
            checked = p
        else:
            proc_status[p] = "false"
            unchecked.append(p)

    options["checked"] = checked
    options["unchecked"] = unchecked
