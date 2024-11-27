Weather Prediction Model
This repository contains a weather prediction model that forecasts weather conditions in Dehradun, India based on real-time temperature, humidity, and weather descriptions. The model uses 2 years of historical weather data, handles class imbalance with SMOTE Tomek, and is built using the CatBoost algorithm. The final model is deployed with a Streamlit GUI.

Table of Contents
Overview

Dataset

Data Preprocessing

Model Training

Model Evaluation

Deployment

Usage

Installation

Contributing

License

Overview
This project aims to predict weather conditions in Dehradun, India using machine learning techniques. The model achieves an accuracy of 80% with excellent AUC ROC scores, precision, and recall. The project includes data collection, preprocessing, model training, evaluation, and deployment using Streamlit.

Dataset
Source: The weather data is collected in real-time from an API.

Duration: The dataset spans over 2 years.

Features:

Temperature

Humidity

Weather Description

The data is stored in CSV files.

Data Preprocessing
Class Imbalance Handling:

Used SMOTE Tomek to increase class counts and handle imbalanced classes.

Feature Engineering:

Extracted relevant features from the raw data.

Normalization:

Normalized the features for better model performance.

Model Training
Algorithm: CatBoost (Gradient Boosting)

Handling Imbalance: Applied class weights during model training.

Accuracy: 80%

Other Metrics: High AUC ROC, precision, and recall.

Model Evaluation
Evaluated the model using various metrics:

Accuracy

AUC ROC

Precision

Recall

Deployment
GUI: Built using Streamlit.

Model Serialization: Used joblib to load the trained model.

Usage
Clone the repository:

sh
git clone https://github.com//weather-prediction-model.git
cd weather-prediction-model
Install the required packages:

sh
pip install -r requirements.txt
Run the Streamlit app:

sh
streamlit run app.py
Installation
Clone the repository:

sh
git clone https://github.com/yourusername/weather-prediction-model.git
Navigate to the project directory:

sh
cd weather-prediction-model
Install dependencies:

sh
pip install -r requirements.txt
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License - see the LICENSE file for details
