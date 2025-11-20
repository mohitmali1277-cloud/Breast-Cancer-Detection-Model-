import pickle
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

# Ignore warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- 1. Load Artifacts (Model, Scaler, and Feature Names) ---
try:
    # Load the Keras model (using the standard Keras format)
    model = tf.keras.models.load_model("Breast_model.keras")
    
    # Load feature names and the scaler object
    with open("feature_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
        FEATURE_NAMES = artifacts['feature_names']
        scaler = artifacts['scaler']

    print("Model, Scaler, and Features loaded successfully!")
    print(f"Model requires {len(FEATURE_NAMES)} features.")

except Exception as e:
    print(f"ERROR loading artifacts: {e}")
    model = None
    FEATURE_NAMES = []
    scaler = None

# --- 2. Define Routes ---
@app.route('/')
def home():
    """Renders the initial input form page, passing the feature names."""
    if not FEATURE_NAMES:
         return "Initialization Error: Could not load features.", 500
         
    return render_template('BreastCancer.html', feature_names=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not FEATURE_NAMES:
        return render_template('BreastCancer.html', prediction_text="Error: System initialization failed.", result_class="error")

    try:
        form_data = request.form
        
        # 1. Prepare data for the model (ensuring correct order)
        # Check if any required feature is missing
        if not all(name in form_data for name in FEATURE_NAMES):
             raise ValueError("Missing one or more required input fields.")
             
        input_data = [float(form_data[name]) for name in FEATURE_NAMES]
        
        # Convert to DataFrame
        final_features = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # 2. Scale the data using the loaded scaler
        scaled_features = scaler.transform(final_features)
        
        # 3. Reshape for the Conv1D model (1 sample, N_FEATURES steps, 1 feature)
        reshaped_features = scaled_features.reshape(1, len(FEATURE_NAMES), 1)
        
        # 4. Make Prediction (returns a probability)
        prediction_prob = model.predict(reshaped_features)[0][0]
        
        # 5. Determine the result string (0.5 threshold)
        if prediction_prob >= 0.5:
    # High probability now corresponds to the BENIGN class
            result_text = f"Benign (B) - Probability: {prediction_prob:.2f}"
            result_class = "benign"
        else:
    # Low probability now corresponds to the MALIGNANT class
            result_text = f"Malignant (M) - Probability: {(1 - prediction_prob):.2f}"
            result_class = "malignant"

        # Pass the result back to the template
        return render_template('BreastCancer.html', 
                               prediction_text=result_text,
                               result_class=result_class,
                               feature_names=FEATURE_NAMES)

    except Exception as e:
        return render_template('BreastCancer.html', 
                               prediction_text=f"Error processing input: {e}. Check values.",
                               result_class="error",
                               feature_names=FEATURE_NAMES)

if __name__ == "__main__":
    app.run(debug=True)