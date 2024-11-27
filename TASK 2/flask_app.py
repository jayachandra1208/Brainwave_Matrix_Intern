# Import libraries
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Load pre-trained model
model = joblib.load("fraud_detection_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        features = pd.DataFrame([data])

        # Ensure features match the model's expected format
        prediction = model.predict(features)
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
