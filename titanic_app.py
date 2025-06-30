import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgboost_titanic_model.pkl")
scaler = joblib.load("scaler_titanic.pkl")  # Save this from your training script

# Define prediction function
def predict_survival(input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction, probability

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival chance on Titanic.")

# --- Input Fields ---
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Fare Paid", 0.0, 600.0, 30.0)
embarked = st.selectbox("Embarkation Port", ['S', 'C', 'Q'])
title = st.selectbox("Title", ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])

# Encode
sex = 0 if sex == 'male' else 1
embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked]
title_map = {'Mr': 3, 'Miss': 1, 'Mrs': 2, 'Master': 0, 'Rare': 4}
title = title_map[title]
family_size = sibsp + parch + 1

# Form input array
input_array = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, family_size, title]])

# Predict
if st.button("Predict Survival"):
    pred, prob = predict_survival(input_array)
    if pred == 1:
        st.success(f"Likely to Survive! (Probability: {prob:.2f})")
    else:
        st.error(f"Unlikely to Survive. (Probability: {prob:.2f})")

## test it using 1 , female,25,0,0,100,c,miss