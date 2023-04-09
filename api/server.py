from flask import Flask, abort, jsonify, request
from flask_cors import CORS

from predict import Predict

app = Flask(__name__)
CORS(app)

pred = Predict()


@app.get("/text")
def get_prediction():
    comment_text = request.args.get("q")
    if comment_text is None:
        return abort(400)
    return jsonify(pred.get_prediction(comment_text))


@app.get("/video")
def get_video():
    videoId = request.args.get("v")
    try:
        count = int(request.args.get("count"))
    except:
        count = 100

    if videoId is None:
        return abort(400)
    return jsonify(pred.get_video_predictions(videoId=videoId, count=count))

app.run("127.0.0.1", 8080, True)