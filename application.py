import streamlit as st
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import matplotlib.pyplot as plt
import os  # For checking file existence

# Ignore warnings
warnings.filterwarnings('ignore')

# Cache the dataset loading function
@st.cache_data
def load_data(file_path):
    try:
        if os.path.exists(file_path):  # Check if the file exists
            df = pd.read_csv(file_path)
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
            df.dropna(subset=['Month'], inplace=True)
            df.set_index('Month', inplace=True)
            return df
        else:
            raise FileNotFoundError  # Raise exception if file is not found
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please upload the correct file or check the path.")
        return None  # Return None if file is not found

# Cache the model fitting function
@st.cache_resource
def fit_sarima_model(df, best_pdq):
    return SARIMAX(df['Price'], order=best_pdq, seasonal_order=(1, 2, 1, 12)).fit()

# Cache the ARIMA grid search function
@st.cache_data
def find_best_arima_params(df):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_pdq = None
    for param in pdq:
        try:
            temp_model = ARIMA(df['Price'], order=param)
            temp_model_fit = temp_model.fit()
            if temp_model_fit.aic < best_aic:
                best_aic = temp_model_fit.aic
                best_pdq = param
        except:
            continue
    return best_pdq

# CSS to add background image and change header colors
custom_css = """
<style>
    .stApp {
        background-image: url('https://static.vecteezy.com/system/resources/previews/027/004/037/non_2x/green-natural-leaves-background-free-photo.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stMarkdown h1 {
        color: black !important;
    }
    .stMarkdown h2 {
        color: black !important;
    }
    .stMarkdown h3 {
        color: black !important;
    }
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit app title and description
st.title("AGRI-FORECAST")

st.header("Predict Future Values of Agri-Horticulture Commodities")
st.subheader("Enter the commodity, month, and year for which you need predictions, and get the predicted average sales for that month.")

state = st.selectbox(
    "STATE",
    ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", 
     "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", 
     "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", 
     "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", 
     "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", 
     "West Bengal"]
)


season = st.selectbox(
    "SEASON",
    ["Winter", "Spring", "Summer", "Monsoon", "Autumn", "Pre-Winter"]
)

# Commodity selection
com = st.selectbox(
    "COMMODITY", 
    ["Gram Dal", "Sugar", "Gur", "Wheat", "Tea", "Milk", "Salt", "Atta", "Tur/Arhar Dal", "Urad Dal", "Moong Dal", 
     "Masoor Dal", "Groundnut Oil", "Mustard Oil", "Vanaspati", 
     "Sunflower Oil", "Soya Oil", "Palm Oil","Rice", "Potato", 
     "Onion", "Tomato"]
)

# File paths for each commodity
file_paths = {
    "Gram Dal": "Dal_price.csv",
    "Sugar": "Chini.csv",
    "Gur": "Gur.csv",
    "Tea": "Tea - Sheet1.csv",
    "Milk": "Milk - Sheet1.csv",
    "Salt":"Salt - Sheet1.csv",
    "Wheat": "Wheat - Sheet1.csv",
    
}

# Initialize df as None
df = None

# Load data and fit model based on selected commodity
if com in file_paths:
    with st.spinner(f"Loading model for {com}..."):
        df = load_data(file_paths[com])
        if df is not None:
            best_pdq = find_best_arima_params(df)
            seasonal_model_fit = fit_sarima_model(df, best_pdq)

        # Input fields for Year and Month
        year = st.number_input("Enter Year:", min_value=2014, max_value=2034, step=1)
        month = st.selectbox("Select Month:", options=list(range(1, 13)))

        # Convert year and month into a date
        input_date = pd.to_datetime(f"{year}-{month:02d}-01")

        # Forecast till 2034
        if df is not None:
            future_start_date = '2024-09-01'
            future_end_date = '2034-12-01'
            future_dates = pd.date_range(start=future_start_date, end=future_end_date, freq='MS')

            # Forecast for future period
            future_forecast_steps = len(future_dates)
            future_forecast = seasonal_model_fit.forecast(steps=future_forecast_steps)

            # Combine original and predicted values into a single Pandas Series
            combined_series = df['Price'].copy()
            future_forecast_series = pd.Series(future_forecast, index=future_dates)
            combined_series = pd.concat([combined_series, future_forecast_series])

            # Display predicted price on button click
            if st.button("Get Value"):
                if input_date in combined_series.index:
                    predicted_price = combined_series[input_date]
                    st.write(f"The predicted price for {input_date.strftime('%Y-%m')} is: **₹{predicted_price:.2f}**")
                else:
                    st.write("Price data is not available for the selected date.")

                # Plot original and predicted prices together
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df.index, df['Price'], label="Original Price", color='blue')
                ax.plot(future_forecast_series.index, future_forecast_series, label="Predicted Price", color='red', linestyle='dashed')
                ax.axvline(input_date, color='green', linestyle='--', label="Prediction Point")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.set_title(f"Price Forecast for {com}")
                ax.legend()

                # Display the plot in Streamlit
                st.pyplot(fig)
else:
    st.error("Model is in Progress.")
