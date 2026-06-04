import streamlit as st
import pandas as pd
import joblib

# 1. SETUP & MODEL LOADING
# Configure the page layout
st.set_page_config(page_title="PrimeVisit AI", page_icon="🌍", layout="centered")

# Load the trained machine learning model
# @st.cache_resource ensures the model only loads once
@st.cache_resource
def load_model():
    return joblib.load('prime_visit_model.pkl')

model = load_model()

# 2. FRONT-END USER INTERFACE
st.title("🌍 PrimeVisit AI")
st.subheader("Optimal Travel Window Predictor for Lisbon, Portugal")
st.write("Adjust the parameters below to see if your planned travel dates offer the perfect balance of good weather, low crowds, and fair pricing.")

st.divider()

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Travel Dates")
    month = st.selectbox("Month of Travel", options=list(range(1, 13)), index=6) # Default to July
    day = st.selectbox("Day of the Month", options=list(range(1, 32)), index=14)
    
    st.markdown("### 💰 Budget")
    price = st.number_input("Average Daily Hotel Rate (ADR)", min_value=50.0, max_value=500.0, value=120.0, step=10.0)

with col2:
    st.markdown("### 🌤️ Weather Forecast")
    
    # Create a dropdown with fixed string ranges for Temperature
    temp_selection = st.selectbox(
        "Expected Temperature", 
        options=[
            "Cool (0°C - 15°C)", 
            "Prime (15°C - 25°C)", 
            "Warm (25°C - 35°C)", 
            "Hot (35°C - 45°C)"
        ],
        index=1 # Defaults to the 'Prime' option
    )
    
    # Create a dropdown with fixed string ranges for Rainfall
    precip_selection = st.selectbox(
        "Expected Rainfall", 
        options=[
            "Dry (0 - 2 mm)", 
            "Light Rain (2 - 5 mm)", 
            "Heavy Rain (5 - 20 mm)"
        ],
        index=0 # Defaults to the 'Dry' option
    )
st.divider()

# 3. BACK-END PREDICTION ENGINE
# When the user clicks the button, run the model
if st.button("Predict Suitability 🚀", use_container_width=True):
    
    # 1. Map the user's text selection to the numerical midpoint
    temp_map = {
        "Cool (0°C - 15°C)": 7.5,
        "Prime (15°C - 25°C)": 20.0,
        "Warm (25°C - 35°C)": 30.0,
        "Hot (35°C - 45°C)": 40.0
    }
    
    precip_map = {
        "Dry (0 - 2 mm)": 1.0,
        "Light Rain (2 - 5 mm)": 3.5,
        "Heavy Rain (5 - 20 mm)": 12.5
    }
    
    # 2. Extract the exact numbers using the dictionaries
    model_temp = temp_map[temp_selection]
    model_precip = precip_map[precip_selection]
    
    # 3. Format the inputs into a DataFrame matching our training data
    input_data = pd.DataFrame({
        'Month': [month],
        'Day': [day],
        'Temperature': [model_temp],       # Pass the mapped number
        'Precipitation_mm': [model_precip], # Pass the mapped number
        'Average_Price': [price]
    })
    
    # 4. Make the prediction
    prediction = model.predict(input_data)
    
    # 5. Display the results beautifully
    if prediction[0] == 1:
        st.success("### ✅ Prime Time to Visit!")
        st.write("The algorithm predicts excellent weather and manageable crowds for these dates.")
        st.balloons() 
    else:
        st.error("### ❌ Sub-optimal Travel Window")
        st.write("The algorithm predicts either heavy crowds, uncomfortable temperatures, or high rainfall. Consider shifting your dates!")
