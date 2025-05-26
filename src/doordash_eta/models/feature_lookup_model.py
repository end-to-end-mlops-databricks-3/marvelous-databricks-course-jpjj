"""FeatureLookUp model implementation."""

import mlflow
from catboost import CatBoostRegressor
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from doordash_eta.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.doordash_eta_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_dashers_per_order"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the doordash_eta_features table and populate it.

        This table stores features related to houses.
        """
        self.spark.sql(
            f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (id STRING NOT NULL, total_items DOUBLE, subtotal DOUBLE, num_distinct_items DOUBLE);
        """
        )
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT order_pk PRIMARY KEY(id);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT id, total_items, subtotal, num_distinct_items FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT id, total_items, subtotal, num_distinct_items FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate dashers_per_order.

        This function devides the number of dashers by the number of outstanding orders.
        """
        self.spark.sql(
            f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(total_onshift_dashers DOUBLE, total_outstanding_orders DOUBLE)
        RETURNS DOUBLE
        LANGUAGE PYTHON AS
        $$
        return total_onshift_dashers / (total_outstanding_orders + 1e-5)
        $$
        """
        )
        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables."""
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "total_items", "subtotal", "num_distinct_items"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("id", self.train_set["id"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["total_items", "subtotal", "num_distinct_items"],
                    lookup_key="id",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="dashers_per_order",
                    input_bindings={
                        "total_onshift_dashers": "total_onshift_dashers",
                        "total_outstanding_orders": "total_outstanding_orders",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["dashers_per_order"] = self.test_set["total_onshift_dashers"] / (
            self.test_set["total_outstanding_orders"] + 1e-5
        )

        self.X_train = self.training_df[self.num_features + self.cat_features + ["dashers_per_order"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features + ["dashers_per_order"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and CatBoost regressor.
        """
        logger.info("🚀 Starting training...")

        # Define a function to convert categorical features to strings
        def to_string(x: float) -> str:
            """Doc string."""
            return x.astype(str)

        # Create a transformer that applies the to_string function
        string_transformer = FunctionTransformer(to_string)

        # Use the transformer in your ColumnTransformer
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", string_transformer, self.cat_features)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        self.preprocessor.set_output(transform="pandas")
        catboost_regressor = CatBoostRegressor(**self.parameters, cat_features=self.cat_features)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (
                    "regressor",
                    catboost_regressor,
                ),
            ]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            mlflow.log_param("model_type", "CatBoost with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="catboost-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/catboost-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.doordash_eta_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.doordash_eta_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.doordash_eta_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
