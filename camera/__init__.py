from flask import Flask

STREAM_URL_002 = "http://10.3.0.102:8080/?action=stream"


app = Flask(__name__)


import camera.main
