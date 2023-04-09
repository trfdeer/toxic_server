from detoxify import Detoxify
from flask import Flask, abort, jsonify, request
from flask_cors import CORS

from predict import Predict

app = Flask(__name__)
CORS(app)


model = Detoxify("multilingual")


@app.get("/text")
def get_prediction():
    comment_text = request.args.get("q")
    if comment_text is None:
        return abort(400)
    return jsonify({"toxic": 1 if model.predict(comment_text)["toxicity"] >= 0.5 else 0})


app.run("127.0.0.1", 8888)
