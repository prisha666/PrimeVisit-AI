# PrimeVisit AI: Optimal Travel Window Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-STREAMLIT-APP-LINK-HERE)

## Project Overview
This project is a Machine Learning pipeline and interactive web application designed to predict the optimal time to visit a destination by analyzing historical crowd density and weather patterns. By engineering a custom target variable ('Is_Suitable'), the model classifies whether a specific day of the year offers the ideal balance of low crowds, comfortable temperatures, and minimal rainfall.

## The Data Challenge & Solution (Annual Profiling)
Real-world data rarely aligns perfectly. This project utilizes two distinct datasets with entirely different timeframes:
1. **Hotel Booking Demand:** ~119,000 rows of historical demand (2015-2017)
2. **Global Weather Repository:** Daily climate data (2024-2026)

**The Solution:** To merge these asynchronous datasets, I stripped the year data and aggregated metrics strictly by `Month` and `Day`. This created a 365-day "Annual Profile" that successfully mapped historical footfall trends against historical climate averages for Lisbon, Portugal.

## Methodology & Tech Stack
* **Language/Libraries:** Python, Pandas, NumPy, Scikit-Learn, Streamlit
* **Feature Engineering:** Extracted temporal variables and aggregated daily guest counts to measure tourism density.
* **Target Variable:** Engineered a custom binary classification ('Is_Suitable') where $1 =$ Temp between 15°C-25°C **+** Precipitation < 2mm **+** Crowd Density below the 75th percentile.
* **Model:** Random Forest Classifier (100 estimators) chosen for its ability to capture complex, non-linear seasonal relationships.

## Results
The Random Forest model achieved an overall accuracy of **87%**. The model successfully identified complex relationships between the time of year, pricing trends, and weather suitability, prioritizing precision to ensure recommended travel days genuinely reflect optimal conditions.

## How to Run the App Locally
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
   pip install -r requirements.txt