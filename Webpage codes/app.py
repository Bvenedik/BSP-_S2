from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib as jl
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


app = Flask(__name__)
CORS(app)  


rf_model = jl.load('models/random_forest_model.joblib')
nb_model  = jl.load('models/naive_bayes_model.joblib')
svm_model = jl.load('models/svm_model.joblib')

DATA_PATH = "Training.csv"
training_data = pd.read_csv(DATA_PATH).dropna(axis = 1)


encoder = LabelEncoder()
training_data["prognosis"] = encoder.fit_transform(training_data["prognosis"])


@app.route('/page1')
def page1():
    return render_template('predicting.html')

@app.route('/predict_model1', methods=['POST'])

def predict_model1():
    response = jsonify({
        'message': 'Invalid payload',
    })
    response.status_code = 400
    
    data = request.get_json()
    input = data['symptoms']
    
    
    
    if not data or 'symptoms' not in data:
        return response
    
    
    symptoms = jl.load('models/symptoms_list.joblib')
    #print(symptoms)
    
    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index
        #print(symptom)
        #print(symptom_index)
        
    data_dict = {
        "symptom_index":symptom_index,
        "predictions_classes":encoder.classes_
    }

    print(data_dict)


    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in input:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
        

    #print("Input data array:", input_data) 
    input_data = np.array(input_data).reshape(1,-1)
    
    #print("Input data shape:", input_data.shape)
    #print("Input data:", input_data)


    predictions = {
        'Random Forest': data_dict["predictions_classes"][rf_model.predict(input_data)[0]],
        'Naive Bayes': data_dict["predictions_classes"][nb_model.predict(input_data)[0]],
        'SVM': data_dict["predictions_classes"][svm_model.predict(input_data)[0]]
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
