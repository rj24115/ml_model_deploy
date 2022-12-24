# RJ 23dec2023
import numpy as np
from flask import Flask, request, jsonify, render_template
import json
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb")) #RF
model = pickle.load(open("model2.pkl", "rb")) #LR

# routing for displaying webpage
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

# API enpoints - IRIS

iris_dict = [
    {'Sepal_Length':5,
    'Sepal_Width':4,
    'Petal_Length':1,
    'Petal_Width':0.5 }
]

@flask_app.route('/iris')
def get_iris():
    return jsonify(iris_dict)

@flask_app.route('/iris', methods=['POST'])
def add_iris():
    iris_dict.append(request.get_json())
    return '', 204

@flask_app.route('/iris_predict', methods=['POST'])
def predict1():
    received = request.get_json()
    print(received.values())
    # float_features = [float(x) for x in received]
    # features = [np.array(float_features)]
    features = np.array(list(received.values())).reshape(1, -1)
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    prediction_text = "some text"
    prediction_text = f"The flower species is {prediction}, with probability {prediction_proba}"
    
    return prediction_text

@flask_app.route('/iris_predict2', methods=['POST'])
def predict2():
    received = request.get_json()
    print(received.values())
    # float_features = [float(x) for x in received]
    # features = [np.array(float_features)]
    features = np.array(list(received.values())).reshape(1, -1)
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    prediction_text = "some text"
    prediction_text = f"The flower species is {prediction}, with probability {prediction_proba}"
    
    return {
        "prediction": prediction[0],
        "pred probablity": prediction_proba[0][0]
        # "text": prediction_text
    }




# API enpoints for income value
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