import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os


model = joblib.load("D:\\c++\\ML\\balanced_random_forest_model.joblib")


# Define the feature names as they were in training
train_feature_names = [
    'Temperature', 'Humidity', 'DayOfYear', 
    'WeekOfYear', 'sin_month', 'cos_month'
]

# Define reverse mapping (Numeric to String)
reverse_mapping = {
    0: 'Clear',
    1: 'Partially cloudy',
    2: 'Rain',
    3: 'Rain, Overcast',
    4: 'Rain, Partially cloudy'
}

# Streamlit App Interface
st.title('Weather Prediction Model')

# User Inputs for Features
temperature = st.number_input("Enter Temperature (°C)", min_value=-30, max_value=50, value=20)
humidity = st.number_input("Enter Humidity (%)", min_value=0, max_value=100, value=60)

# Input a Date
date_input = st.date_input("Select a Date")

# Process the Date
try:
    # Parse the date
    date_object = datetime.strptime(str(date_input), '%Y-%m-%d')
    day_of_year = date_object.timetuple().tm_yday  # Day of the year
    week_of_year = date_object.isocalendar()[1]    # ISO week number
    month = date_object.month                      # Month

    # Transform 'Month' into cyclic features (sin_month, cos_month)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'DayOfYear': [day_of_year],
        'WeekOfYear': [week_of_year],
        'sin_month': [sin_month],
        'cos_month': [cos_month]
    })

    # Reorder columns to match the training feature names
    input_data = input_data[train_feature_names]

    # Display the prepared input data (optional debug)
    st.write("Input Data for Prediction:")
    st.write(input_data)

    # Make the prediction using the loaded model
    if st.button('Predict'):
        prediction = model.predict(input_data)
        
        # Ensure prediction output is properly formatted
        if isinstance(prediction, np.ndarray) and len(prediction) > 0:
            predicted_class_numeric = int(prediction[0])  # Convert to integer
        else:
            predicted_class_numeric = prediction  # Handle any edge case

        # Get the predicted class as a string
        predicted_class_string = reverse_mapping.get(predicted_class_numeric, "Unknown")  # Map to string
        
        # Display the predicted class
        st.success(f"Predicted Class (Numeric): {predicted_class_numeric}")
        st.success(f"Predicted Class (String): {predicted_class_string}")
except Exception as e:
    st.error(f"An error occurred: {e}")
