import os
from typing import List

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


POSTGRES_DBNAME = os.environ["POSTGRES_DBNAME"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]

MLFLOW_BUCKET_NAME = os.environ["MLFLOW_BUCKET_NAME"]


# Check if the experiment already exists
def experiment_exists(name):
    list_exp = mlflow.search_experiments(filter_string=f"name = '{name}'")
    return len(list_exp) > 0


mlflow.set_tracking_uri(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@postgres:5432/{POSTGRES_DBNAME}"
)

# Create an experiment if it doesn't exist
EXPERIMENT_NAME = "mlops-zoomcamp-ml-exp"
if not experiment_exists(EXPERIMENT_NAME):
    mlflow.create_experiment(
        EXPERIMENT_NAME, 
        artifact_location=f"s3://{MLFLOW_BUCKET_NAME}/experiments/"
    )

mlflow.set_experiment(EXPERIMENT_NAME)


@transformer
def transform(list_df: List[pd.DataFrame], *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    X_train, y_train, X_val, y_val = list_df

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        clf = RandomForestClassifier(
            bootstrap=True,
            ccp_alpha=0.0,
            class_weight=None,
            criterion="gini",
            max_depth=None,
            max_features="sqrt",
            max_leaf_nodes=None,
            max_samples=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            monotonic_cst=None,
            n_estimators=100,
            n_jobs=-1,
            oob_score=False,
            random_state=123,
            verbose=0,
            warm_start=False,
        )

        clf.fit(X_train, y_train)

    return clf


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
