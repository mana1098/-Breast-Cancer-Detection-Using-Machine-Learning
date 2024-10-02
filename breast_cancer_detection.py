import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify, render_template
import joblib

# Data Preprocessing
def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
    
    # Convert diagnosis to binary values
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Split features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Model Training
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Save model and scaler
def save_model_and_scaler(model, scaler):
    joblib.dump(model, 'breast_cancer_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

# Main function to run the ML pipeline
def run_ml_pipeline():
    # Load data
    data = pd.read_csv('breast_cancer_data.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    # Save model and scaler
    save_model_and_scaler(model, scaler)

# Flask Web Application
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('breast_cancer_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features_scaled = scaler.transform(final_features)
    prediction = model.predict(final_features_scaled)
    output = "Malignant" if prediction[0] == 1 else "Benign"
    return render_template('index.html', prediction_text=f'The tumor is predicted to be: {output}')

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = "Malignant" if prediction[0] == 1 else "Benign"
    return jsonify(output)

if __name__ == "__main__":
    run_ml_pipeline()
    app.run(debug=True)
