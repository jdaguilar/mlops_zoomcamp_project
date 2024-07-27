import os
import pickle
from typing import List

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


mlflow.set_tracking_uri("postgresql://user:password@postgres:5432/mlflowdb")
mlflow.set_experiment("mlops-zoomcamp-ml-exp")


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
            criterion='gini', 
            max_depth=None, 
            max_features='sqrt',
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
            warm_start=False
        )

        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_val)

        # print(roc_auc_score(y_val, y_pred))
        # 0.857142857143

    return clf


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
