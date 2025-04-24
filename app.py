from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Load model and scaler
try:
    model_path = "rf_model.pkl"
    scaler_path = "scaler.pkl"
    rf_grid = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model type:", type(rf_grid), "Has best_estimator_:", hasattr(rf_grid, 'best_estimator_'))
except Exception as e:
    print(f"Failed to load model or scaler: {e}")
    raise

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug: Print request details
        print("Request headers:", request.headers)
        print("Content-Type:", request.headers.get('Content-Type'))

        # Check if request is JSON
        if not request.is_json:
            print("Request is not JSON, Content-Type:", request.headers.get('Content-Type'))
            return jsonify({'error': 'Content-Type must be application/json'}), 415

        # Get JSON data
        data = request.get_json(force=True)
        print("Received data:", data)
        print("Received feature keys:", list(data.keys()))

        # Define expected features
        expected_features = [
            'SpMax_L', 'J_Dz_e', 'nHM', 'F01_N_N', 'F04_C_N', 'NssssC', 'nCb_', 'C_percent', 
            'nCp', 'nO', 'F03_C_N', 'SdssC', 'HyWi_B_m', 'LOC', 'SM6_L', 'F03_C_O', 'Me', 
            'Mi', 'nN_N', 'nArNO2', 'nCRX3', 'SpPosA_B_p', 'nCIR', 'B01_C_Br', 'B03_C_Cl', 
            'N_073', 'SpMax_A', 'Psi_i_1d', 'B04_C_Br', 'SdO', 'TI2_L', 'nCrt', 'C_026', 
            'F02_C_N', 'nHDon', 'SpMax_B_m', 'Psi_i_A', 'nN', 'SM6_B_m', 'nArCOOR', 'nX'
        ]
        if not all(key in data for key in expected_features):
            missing = [key for key in expected_features if key not in data]
            print("Missing features:", missing)
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Convert features to array
        features = np.array([data[feature] for feature in expected_features]).reshape(1, -1)
        print("Features array:", features, "Shape:", features.shape)

        # Validate feature count and values
        if features.shape[1] != 41:
            print(f"Expected 41 features, got {features.shape[1]}")
            return jsonify({'error': f'Expected 41 features, got {features.shape[1]}'}), 400
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print("Invalid feature values (NaN or inf):", features)
            return jsonify({'error': 'Feature values contain NaN or infinity'}), 400

        # Scale and predict
        features_scaled = scaler.transform(features)
        print("Scaled features:", features_scaled)
        prediction = rf_grid.best_estimator_.predict(features_scaled) if hasattr(rf_grid, 'best_estimator_') else rf_grid.predict(features_scaled)
        print("Prediction:", prediction)

        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)