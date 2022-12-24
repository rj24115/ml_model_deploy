# RJ 23dec2023
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb")) #RF
model = pickle.load(open("model2.pkl", "rb")) #LR


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    prediction_text = f"The flower species is {prediction}, with probability {prediction_proba}"
    return render_template("index.html", prediction_text=prediction_text)

# API enpoints

incomes = [
    { 'description': 'salary', 'amount': 5000 }
]


@flask_app.route('/incomes')
def get_incomes():
    return jsonify(incomes)


@flask_app.route('/incomes', methods=['POST'])
def add_income():
    incomes.append(request.get_json())
    return 'added income', 204



 # **********

if __name__ == "__main__":
    flask_app.run(debug=True)

print('heeeee')