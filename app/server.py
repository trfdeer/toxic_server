from deep_translator import LibreTranslator
from detoxify import Detoxify
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

translator = LibreTranslator(source="auto", target="en")
model = Detoxify()


@app.get("/text")
def get_prediction():
    comment_text = request.args.get("q")
    if comment_text is None:
        return abort(400)

    translated_text = translator.translate(comment_text)
    return jsonify(
        {"toxic": 1 if model.predict(translated_text)["toxicity"] >= 0.5 else 0}
    )


app.run("127.0.0.1", 8888)
