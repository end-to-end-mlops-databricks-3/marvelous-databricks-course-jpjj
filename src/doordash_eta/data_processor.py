"""Data preprocessing module."""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from doordash_eta.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(
        self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession
    ) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        # Handle missing values and convert data types as needed
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            median_value = self.df[col].median()
            self.df[col].fillna(median_value, inplace=True)

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # convert timestamp features to appropriate type:
        dt_format = "%Y-%m-%d %H:%M:%S"
        self.df["actual_delivery_time"] = pd.to_datetime(
            self.df["actual_delivery_time"], format=dt_format
        )
        self.df["created_at"] = pd.to_datetime(self.df["created_at"], format=dt_format)

        # Extract target and relevant features
        target = self.config.target
        self.df[target] = (
            self.df["actual_delivery_time"] - self.df["created_at"]
        ).dt.total_seconds()
        self.df.drop(columns=["actual_delivery_time"], inplace=True)
        # No id in original df, use index as id
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={"index": "id"})
        time_features = self.config.time_features
        relevant_columns = (
            cat_features + num_features + time_features + [target] + ["id"]
        )
        self.df = self.df[relevant_columns]
        self.df["id"] = self.df["id"].astype("str")

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(
            self.df, test_size=test_size, random_state=random_state
        )
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
