from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.s3 import S3
from pandas import DataFrame
from os import path

import os
import pickle
import click
import mlflow
import pathlib

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


EXPERIMENT_NAME = "mlops-zoomcamp-ml-exp"


@data_exporter
def run_register_model(model, data_2, **kwargs) -> None:

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.roc_auc_score_X_val DESC"]
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="clf-best-model")

    #model_info = mlflow.sklearn.log_model(
    #    sk_model=model,
    #    artifact_path="sk_model",
    #    # signature=signature,
    #    registered_model_name="clf-best-model",
    #)

    mlflow.end_run()