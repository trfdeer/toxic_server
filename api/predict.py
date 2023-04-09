import re
import string
from typing import *
import uuid

import joblib
import nltk
import pandas as pd
import textblob
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from scrape import get_comments, instance


class Predict:
    def __init__(self) -> None:
        self.labels = ["toxic", "severe_toxic", "obscene",
                       "threat", "insult", "identity_hate"]

        nltk.download("stopwords")
        self.sn = SnowballStemmer(language="english")

        self.vectorizer = joblib.load("./model/vectroize2_jlib")

        self.classifier = joblib.load("./model/classifier2_jlib")

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"\r", "", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\"'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\'#/@;:<>{}`+=~|.!?,]", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub("(\\W)", " ", text)
        text = re.sub("\S*\d\S*\s*", "", text)

        return text

    def stemmer(self, text):
        words = text.split()
        train = [self.sn.stem(word) for word in words if not word in set(
            stopwords.words("english"))]
        return " ".join(train)

    def make_test_predictions(self, df):
        df.comment_text = df.comment_text.apply(self.clean_text)
        df.comment_text = df.comment_text.apply(self.stemmer)
        X_test = df.comment_text
        X_test = X_test.to_numpy()
        X_test_transformed = self.vectorizer.transform(X_test)
        y_test_pred = self.classifier.predict_proba(X_test_transformed)
        return y_test_pred

    def make_response(self, id: str, comment: str, polarity: float, predictions: List[float]) -> Dict[str, object]:
        resp: Dict[str, object] = {}
        resp["id"] = id
        if polarity >= 0.1:
            resp["label"] = "positive"
        elif polarity >= -0.1:
            resp["label"] = "neutral"
        else:
            resp["label"] = "negative"
            resp["toxic"] = any([x >= 0.5 for x in predictions])
            resp["toxicity"] = dict(zip(self.labels, predictions))

        resp["polarity"] = polarity

        resp["text"] = comment

        return resp

    def get_prediction(self, comment: str) -> Dict[str, object]:
        id = str(uuid.uuid4())
        data = pd.DataFrame(
            {"id": [id], "comment_text": comment})

        pred = self.make_test_predictions(data)
        polarity = textblob.TextBlob(comment).polarity

        return self.make_response(id, comment, polarity, pred[0])

    def get_predictions(self, comments: List[str]) -> List[Dict[str, object]]:
        data = pd.DataFrame(comments, columns=["id", "comment_text"])
        pred = self.make_test_predictions(data)

        polarities = [
            textblob.TextBlob(comment[1])
            .polarity for comment in comments
        ]
        return list(map(lambda x: self.make_response(x[0][0], x[0][1], x[1], x[2]), zip(comments, polarities, pred)))

    def get_video_predictions(self, videoId: str, count: int):
        commentList = []
        for comments in get_comments(instance=instance, videoId=videoId, maxCount=count):
            commentList += comments
        return self.get_predictions(commentList)
