import os
import pickle

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-train")


@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here
    mlflow.sklearn.autolog()
    # X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    # X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
