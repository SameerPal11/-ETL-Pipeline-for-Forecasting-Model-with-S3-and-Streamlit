import streamlit as st
import boto3
import pandas as pd
import io

BUCKET_NAME = "databucketforprocessing"
PROCESSED_FILE_KEY = "processed_data/processed_car_price.csv"

@st.cache_data
def load_data():
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=PROCESSED_FILE_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    return df

st.title("Car Price Forecasting Data")

df = load_data()

st.subheader("Data Preview")
st.dataframe(df.head())

st.subheader("Data Summary")
st.write(df.describe())



st.subheader(" Price Distribution")
st.bar_chart(df["Price"].value_counts())


st.subheader("Fuel Type vs. Average Price")
fuel_avg_price = df.groupby("Fuel_Type")["Price"].mean().reset_index()
st.bar_chart(fuel_avg_price.set_index("Fuel_Type"))
