import streamlit as st
import boto3
import pandas as pd
import io
import numpy as np
from sklearn.linear_model import LinearRegression

BUCKET_NAME = "databucketforprocessing"
PROCESSED_FILE_KEY = "processed_data/car_price_dataset.csv"
FORECASTING_FILE_KEY = "processed_data/car_price_forecast.csv"
LINEAR_REGRESSION_FILE_KEY = "processed_data/Forecast_using _linear_regg.csv"

@st.cache_data
def load_data(file_key):
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        return df
    except Exception as e:
        st.error(f"❌ Error loading data from S3: {e}")
        return pd.DataFrame()

def save_data_to_s3(df, file_key):
    try:
        s3 = boto3.client("s3")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=BUCKET_NAME, Key=file_key, Body=csv_buffer.getvalue())  
        st.success(f"✅ Data successfully saved to S3: {file_key}")
    except Exception as e:
        st.error(f"❌ Error saving data to S3: {e}")

st.title("🚗 Car Price Forecasting with Trend Analysis")

df_original = load_data(PROCESSED_FILE_KEY)
df_forecast = load_data(FORECASTING_FILE_KEY)

if not df_original.empty:
    st.subheader("🔍 Original Data Preview")
    st.dataframe(df_original.head())

    st.subheader("📊 Original Data Summary")
    st.write(df_original.describe())

    st.subheader("📈 Price Distribution")
    st.area_chart(df_original["price"].value_counts().sort_index())

    st.subheader("⛽ Fuel Type vs. Average Price")
    if "fuel_type" in df_original.columns:
        fuel_avg_price = df_original.groupby("fuel_type")["price"].mean().reset_index()
        st.line_chart(fuel_avg_price.set_index("fuel_type"))
    else:
        st.warning("⚠️ 'fuel_type' column not found in dataset!")

    st.subheader("📈 Price Trend Analysis using Linear Regression")
    if "price" in df_original.columns:
        df_original = df_original.reset_index()

        X = np.array(df_original.index).reshape(-1, 1)  
        y = df_original["price"].values.reshape(-1, 1)  

        model = LinearRegression()
        model.fit(X, y)

        df_original["trend"] = model.predict(X).flatten()

        st.line_chart(df_original[["price", "trend"]])

        save_data_to_s3(df_original[["index", "price", "trend"]], LINEAR_REGRESSION_FILE_KEY)

    else:
        st.warning("⚠️ 'price' column not found in dataset!")

if not df_forecast.empty:
    st.subheader("📈 Forecasted Price Trend")
    if "step" in df_forecast.columns and "forecasted_price" in df_forecast.columns:
        st.area_chart(df_forecast.set_index("step")["forecasted_price"])
    else:
        st.warning("⚠️ Columns 'step' and 'forecasted_price' not found in forecast dataset!")
else:
    st.error("❌ No forecast data available for visualization!")
