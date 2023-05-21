from bottle import route, run, response
from bottle import TEMPLATE_PATH, jinja2_template as template

from camera import VideoCamera
import cv2

TEMPLATE_PATH.append("./templates/")


@route("/")
def index():
    return template("index.html")


def generatePng(camera):
    while True:
        a = camera.get_frame()  # ここでcap.readしている
        # ret, frame = cap.read()
        yield (b"--frame\r\n" + b"Content-Type: image/jpeg\r\n\r\n" + a + b"\r\n\r\n")


@route("/video_feed")
def video_feed():
    response.content_type = "multipart/x-mixed-replace; boundary=frame"
    return generatePng(VideoCamera())


if __name__ == "__main__":
    run(host="0.0.0.0", port=8080)
