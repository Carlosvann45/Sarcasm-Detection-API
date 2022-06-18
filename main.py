from flask import *
import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

app = Flask(__name__)


@app.route('/containsSarcasm', methods=['POST'])
def run_sarcasm_check():
    try:
        json_data = request.get_json()
        phrase = json_data['phrase']

        # reads json from file nad sets it up for model
        data = pd.read_json("Sarcasm.json", lines=True)
        data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not Sarcasm", 1: "Sarcasm"})
        data = data[["headline", "is_sarcastic"]]
        x = np.array(data["headline"])
        y = np.array(data["is_sarcastic"])

        # Trains model on values based from Bernoulli's algorithm
        cv = CountVectorizer()
        x = cv.fit_transform(x)  # Fit the Data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        model = BernoulliNB()
        model.fit(x_train, y_train)

        # checks trained model with phrase sent through request
        data = cv.transform([phrase]).toarray()
        output = model.predict(data)

        answer = {'answer': output[0]}

    except BaseException as err:
        error = {'timestamp': datetime.datetime.now(),
                 'code': 503,
                 'error type': 'Server Unavailable',
                 'error message': err}

        return jsonify(error), 503

    return jsonify(answer), 200


if __name__ == '__main__':
    app.run(debug=True)
