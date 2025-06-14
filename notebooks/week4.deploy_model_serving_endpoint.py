# Databricks notebook source
# MAGIC %pip install doordash_eta-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import os
import time

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from doordash_eta.config import ProjectConfig
from doordash_eta.serving.model_serving import ModelServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.doordash_eta_model_basic",
    endpoint_name="doordash-eta-model-serving",
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = [
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    "estimated_order_place_duration",
    "estimated_store_to_consumer_driving_duration",
    "market_id",
    "store_id",
    "store_primary_category",
    "order_protocol",
]

# Sample 1000 records from the training set
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set"
).toPandas()

# Sample 100 records from the training set
sampled_records = (
    test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
)
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{   "total_items":1,
    "subtotal": 500,
    "num_distinct_items": 1,
    "min_item_price": 500,
    "max_item_price": 500,
    "total_onshift_dashers": 49.0,
    "total_busy_dashers": 66.0,
    "total_outstanding_orders": 53.0,
    "estimated_order_place_duration": 251,
    "estimated_store_to_consumer_driving_duration": 291.0,
    "market_id": 2.0,
    "store_id": 5400,
    "store_primary_category": "fast",
    "order_protocol": 4.0
}]
"""


def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/house-prices-model-serving/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
