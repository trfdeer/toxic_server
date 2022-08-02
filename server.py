from flask import Flask, request, jsonify, abort
from predict import Predict

app = Flask(__name__)

pred = Predict()


@app.get("/get-prediction")
def get_prediction():
    comment_text = request.args.get("text")
    if comment_text is None:
        return abort(400)
    return jsonify(pred.get_prediction(comment_text))


app.run("localhost", 8080)
