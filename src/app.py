from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict_water_quality():
    data = request.get_json()
    ph = data['ph']
    turbidity = data['turbidity']
    dissolved_oxygen = data['dissolved_oxygen']

    # Input features as a numpy array
    input_features = np.array([[ph, turbidity, dissolved_oxygen]])

    # Scale the input features
    input_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Map prediction to output
    result = 'Good Quality' if prediction == 1 else 'Bad Quality'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
