from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib as jl
from sklearn.preprocessing import LabelEncoder
import numpy as np


app2 = Flask(__name__)
CORS(app2)  

rc_nb_model = jl.load('models/rc_nb_model.joblib')
rc_rf_model = jl.load('models/rc_rf_model.joblib')



@app2.route('/page2')
def page2():
    return render_template('recommending.html')

@app2.route('/predict_model_2', methods=['POST'])

def predict_model2():
    response = jsonify({
        'message': 'Invalid payload',
    })
    response.status_code = 400

    
    print('AAAAAAAAAAAAAAAAAAAAA')

    data = request.get_json()

    disease = int(data['disease'])
    gender = int(data['gender'])
    age = int(data['age'])
    
    rf_prediction = predict_drug(rc_rf_model, disease, gender, age)
    nb_prediction = predict_drug(rc_nb_model, disease, gender, age)

    #print(rf_prediction)
    #print(nb_prediction)

    predictions = {
    'Random Forest': rf_prediction,
    'Naive Bayes': nb_prediction
    }

    response = jsonify(predictions)
    response.status_code = 200
    return response


def predict_drug(model, disease, gender, age):
    test_data = np.array([[disease, gender, age]])
    prediction = model.predict(test_data)
    return prediction[0]

@app2.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Headers'] = 'Content-Type'
    header['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

if __name__ == '__main__':
    app2.run(debug=True, port=5500)