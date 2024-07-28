import warnings
import pickle
import numpy as np
from flask import Flask, render_template, request

# Suppress warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the trained model
model = pickle.load(open('churn.pkl', 'rb'))

# Define feature names 
feature_names = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Tenure', 'Phone Service', 'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming TV', 'Streaming Movies', 'Paperless Billing',
    'Monthly Charges', 'Total Charges', 'No Phone Service', 'Fiber Optic',
    'No Internet Service', 'Contract: One Year', 'Contract: Two Years', 'Payment Method: Credit Card',
    'Payment Method: Electronic Check', 'Payment Method: Mailed Check', 'Encoded Customer ID'
]

# Initialize Flask application
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form for all 24 features
        features = []
        for i in range(1, 25):  # 24 inputs (1-based indexing in form)
            value = request.form[f'feature{i}']

            # Convert categorical features to integer
            if i in {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23}:
                value = int(value)
            else:
                value = float(value)
            
            features.append(value)
        
        # Add default values for the missing 'Yes' and 'No' inputs
        # These are placeholder values to ensure the model gets 26 features
        features.insert(15, 0)  # Placeholder for 'Yes' feature
        features.insert(16, 0)  # Placeholder for 'No' feature

        # Make prediction using the loaded model
        prediction = model.predict([features])[0]

        output = 'Yes' if prediction == 1 else 'No'

        return render_template('index.html', prediction_text=f'Will the customer churn? {output}', feature_names=feature_names)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred. Please check your input and try again."

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
