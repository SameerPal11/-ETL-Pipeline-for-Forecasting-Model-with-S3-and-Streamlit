import boto3
import io
import pandas as pd
import streamlit as st

s3 = boto3.client("s3")

BUCKET_NAME = "databucketforprocessing"
RAW_FILE = "raw_data/car_price_dataset.csv"
PROCESSED_FILE = "processed_data/"

def process_data(data):
    df = pd.read_csv(io.BytesIO(data))

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    df.dropna(inplace=True)

    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")

def lambda_handler(event, context):
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=RAW_FILE)
        raw_data = response["Body"].read()

        processed_data = process_data(raw_data)

        s3.put_object(Bucket=BUCKET_NAME, Key=PROCESSED_FILE, Body=processed_data)

        return {
            "statusCode": 200,
            "body": f"File {RAW_FILE} processed and saved as {PROCESSED_FILE}."
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error processing file: {str(e)}"
        }












