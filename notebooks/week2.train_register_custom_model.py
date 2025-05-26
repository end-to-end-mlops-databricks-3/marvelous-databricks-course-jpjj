# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from doordash_eta.config import ProjectConfig, Tags
from doordash_eta.models.custom_model import CustomModel
import os
from doordash_eta import __version__ as doordash_eta_v

# COMMAND ----------
# Default profile:
profile = os.environ["PROFILE"]
mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_registry_uri(f"databricks-uc://{profile}")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(
    **{"git_sha": "a6a59e6da4d63e98ddd87275bda463846cac9e26", "branch": "week2"}
)

# COMMAND ----------
# Initialize model with the config path
custom_model = CustomModel(
    config=config,
    tags=tags,
    spark=spark,
    code_paths=[f"../dist/doordash_eta-{doordash_eta_v}-py3-none-any.whl"],
)

# COMMAND ----------
custom_model.load_data()
custom_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
custom_model.train()
custom_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(experiment_names=["/Shared/doordash-eta-custom"]).run_id[0]

model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-doordash-eta-model")

# COMMAND ----------
# Retrieve dataset for the current run
custom_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
custom_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
custom_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = custom_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
