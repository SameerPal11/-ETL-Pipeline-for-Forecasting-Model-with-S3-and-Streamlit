import boto3
import io
import pandas as pd

s3 = boto3.client("s3")

BUCKET_NAME = "databucketforprocessing"
RAW_FILE = "raw_data/car_price_dataset.csv"
PROCESSED_FILE = "processed_data/car_price_dataset.csv"
FORECAST_FILE = "processed_data/car_price_forecast.csv"

def process_data(data):
    df = pd.read_csv(io.BytesIO(data))

    df.columns = df.columns.str.lower().str.replace(" ", "_")  
    df.dropna(inplace=True) 

    output = io.StringIO()
    df.to_csv(output, index=False)
    return df, output.getvalue().encode("utf-8")

def moving_average_forecast(df, window=3, steps=5000):
    if "price" not in df.columns:
        raise ValueError("No 'price' column found for forecasting!")

    df["price"] = df["price"].astype(float) 

    df["moving_avg"] = df["price"].rolling(window=window).mean()

    last_avg = df["moving_avg"].dropna().iloc[-1]
    forecast = [last_avg] * steps  
    forecast_df = pd.DataFrame({"step": range(1, steps + 1), "forecasted_price": forecast})

    output = io.StringIO()
    forecast_df.to_csv(output, index=False)
    
    return output.getvalue().encode("utf-8")

def lambda_handler(event, context):
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=RAW_FILE)
        raw_data = response["Body"].read()

        df, processed_data = process_data(raw_data)

        forecast_data = moving_average_forecast(df, window=3, steps=10)

        s3.put_object(Bucket=BUCKET_NAME, Key=PROCESSED_FILE, Body=processed_data)

        s3.put_object(Bucket=BUCKET_NAME, Key=FORECAST_FILE, Body=forecast_data)

        return {
            "statusCode": 200,
            "body": f"File {RAW_FILE} processed and forecast saved as {PROCESSED_FILE} & {FORECAST_FILE}."
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error processing file: {str(e)}"
        }
