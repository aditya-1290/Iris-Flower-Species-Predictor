import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open('best_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    target_names = model_data['target_names']

# Streamlit app title
st.title("Iris Flower Species Predictor")

# User input for features
st.header("Input Measurements")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prepare input data for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_data_scaled = scaler.transform(input_data)

# Make prediction
if hasattr(model, "predict_proba"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
else:
    prediction = model.predict(input_data_scaled)
    prediction_proba = np.array([[1.0 if i == prediction else 0.0 for i in range(len(target_names))]])

# Display prediction results
st.header("Prediction Results")
st.write(f"Predicted Species: {target_names[prediction][0]}")
st.write("Prediction Probabilities:")
for i, species in enumerate(target_names):
    st.write(f"{species}: {prediction_proba[0][i]:.2f}")

# Run the app
if __name__ == "__main__":
    st.write("Run this app using `streamlit run app.py` in the terminal.")
