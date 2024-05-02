from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from joblib import load
from sklearn.preprocessing import LabelEncoder
import numpy as np


app = Flask(__name__)
CORS(app)  


rf_model = load('models/random_forest_model.joblib')
nb_model  = load('models/naive_bayes_model.joblib')
svm_model = load('models/svm_model.joblib')
rc_nb_model = load('models/rc_nb_model.joblib')
rc_rf_model = load('models/rc_rf_model.joblib')


@app.route('/page1')
def page1():
    return render_template('predicting.html')

@app.route('/page2')
def page2():
    return render_template('recommending.html')

@app.route('/predict_model1', methods=['POST'])

def predict_model1():
    response = jsonify({
        'message': 'Invalid payload',
    })
    response.status_code = 400

    data = request.get_json() 
    if not data or 'symptoms' not in data:
        return response

    symptoms = data['symptoms']
    
    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([str(i).capitalize() for i in str(value).split("_")])
        symptom_index[symptom] = index

    encoder = LabelEncoder()
    encoder.fit(symptoms)
    data_dict = {
        "symptom_index":symptom_index,
        "predictions_classes":encoder.classes_
    }

    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    print("Input data array:", input_data) 
    input_data = np.array(input_data).reshape(1,-1)
    
    print("Input data shape:", input_data.shape)
    print("Input data:", input_data)

    predictions = {
        'Random Forest': data_dict["predictions_classes"][rf_model.predict(input_data)[0]],
        'Naive Bayes': data_dict["predictions_classes"][nb_model.predict(input_data)[0]],
        'SVM': data_dict["predictions_classes"][svm_model.predict(input_data)[0]]
    }
    response = jsonify(predictions)
    response.status_code = 200
    return response


@app.route('/predict_model2', methods=['POST'])

def predict_drug(model, disease, gender, age):
    test_data = np.array([[disease, gender, age]])
    prediction = model.predict(test_data)
    return prediction[0]

def predict_model2():
    data = request.json()

    disease = int(data['disease'])
    gender = int(data['gender'])
    age = int(data['age'])

    print(disease)
    
    rf_prediction = predict_drug(rc_rf_model, disease, gender, age)
    nb_prediction = predict_drug(rc_nb_model, disease, gender, age)

    predictions = {
    'Random Forest': rf_prediction,
    'Naive Bayes': nb_prediction
    }
    response = jsonify(predictions)
    response.status_code = 200
    return response

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Headers'] = 'Content-Type'
    header['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

if __name__ == '__main__':
    app.run(debug=True)
