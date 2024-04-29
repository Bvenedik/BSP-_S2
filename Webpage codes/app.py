from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from joblib import load
import numpy as np

app = Flask(__name__)
CORS(app) 


rf_model = load('random_forest_model.joblib')
nb_model = load('naive_bayes_model.joblib')
svm_model = load('svm_model.joblib')
symptom_index = load('symptom_index.joblib')

@app.route('/')
def index():
    return render_template('predicting.html')

@app.route('/predict', methods=['POST'])
def predict():
    response = jsonify({
        'message': 'Invalid payload',
    })
    response.status_code = 400

    data = request.get_json()  
    if not data or 'symptoms' not in data:
        return response

    symptoms = data['symptoms']
    input_data = [0] * len(symptom_index)
    for symptom in symptoms:
        symptom_key = symptom.lower().replace(" ", "_") 
        if symptom_key in symptom_index:
            input_data[symptom_index[symptom_key]] = 1

    input_data = np.array(input_data).reshape(1, -1)
    predictions = {
        'Random Forest': int(rf_model.predict(input_data)[0]),
        'Naive Bayes': int(nb_model.predict(input_data)[0]),
        'SVM': int(svm_model.predict(input_data)[0])
    }
    response = jsonify(predictions)
    response.status_code = 200
    return response

if __name__ == '__main__':
    app.run(debug=True)

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Headers'] = 'Content-Type'
    header['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

if __name__ == '__main__':
    app.run(debug=True)
