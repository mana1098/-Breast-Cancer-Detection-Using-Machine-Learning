# Breast Cancer Detection Using Machine Learning

## Project Overview
This project implements a machine learning model for breast cancer detection, achieving over 95% accuracy in classifying breast tumors as benign or malignant. It utilizes a Random Forest classifier and includes a Flask web interface for real-time predictions, facilitating improved diagnostic processes in healthcare.

## Key Features
- High accuracy breast tumor classification (>95% accuracy)
- Random Forest classifier implementation
- Extensive data preprocessing for optimal model performance
- Flask web interface for real-time predictions
- Easy-to-use API for integration into existing healthcare systems

## Installation

### Prerequisites
- Python 3.7+
- pip

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/breast-cancer-detection.git
   cd breast-cancer-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the model and start the Flask server, run:
```
python breast_cancer_detection.py
```

This will:
1. Load and preprocess the data
2. Train the Random Forest model
3. Evaluate the model's performance
4. Save the trained model and scaler
5. Start the Flask web server

### Using the Web Interface
Once the server is running, open a web browser and go to `http://localhost:5000`. You'll see a form where you can input tumor characteristics for classification.

### API Usage
You can also use the API endpoint for programmatic access:

```python
import requests

url = 'http://localhost:5000/results'
data = {
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    # ... include all required features
}
response = requests.post(url, json=data)
print(response.json())
```

## Data
The model is trained on the Breast Cancer Wisconsin (Diagnostic) Data Set. Ensure you have the rights to use this dataset and include appropriate attributions.

## Model Performance
- Accuracy: >95%
- For detailed performance metrics, refer to the console output after training or check the `model_evaluation.log` file.

## Contributing
Contributions to improve the model's performance or expand its capabilities are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Your Name - jainmanali1003.email@example.com

Project Link: [https://github.com/yourusername/breast-cancer-detection](https://github.com/yourusername/breast-cancer-detection)

## Acknowledgements
- [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [Scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
