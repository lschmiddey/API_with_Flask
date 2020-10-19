from flask import Flask, request, jsonify
import csv
import pandas as pd

from utils import transform_data, get_prediction

app = Flask(__name__)

@app.route('/predict', methods=["GET"])
def predict():
    if request.method == 'GET':
        Pclass = request.args.get('Pclass')
        Age = request.args.get('Age')
        Sex = request.args.get('Sex')
        Parch = request.args.get('Parch')
        raw_data = pd.DataFrame({'Pclass': [Pclass], 'Age': [Age], 'Sex':[Sex], 'Parch':[Parch]})
        transf_data = transform_data(raw_data)
        prediction = get_prediction(transf_data)
        prediction = prediction[0][1].item()
        # We take the first value of our predictions, representing the probability not to churn.
        data = {'prediction': prediction}
        return jsonify(data)
    else:
        return jsonify({'error': 'Only GET requests possible'})
