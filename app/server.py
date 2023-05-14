import joblib
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from flask_cors import CORS
from keras.models import load_model
from keras.utils import pad_sequences

load_dotenv()

app = Flask(__name__)
CORS(app)


model_lstm = load_model("./saved/model")
tokenizer = joblib.load("./saved/tokenizer")


def predict(text):
    data_tokenized = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=224)
    pred = model_lstm.predict(data_tokenized)[0]
    return dict(
        zip(
            [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ],
            pred,
        )
    )


@app.get("/text")
def get_prediction():
    comment_text = request.args.get("q")
    if comment_text is None:
        return abort(400)

    preds = predict(comment_text)
    print(preds)
    return jsonify({"toxic": 1 if preds["toxic"] >= 0.5 else 0})


app.run("127.0.0.1", 8888)
