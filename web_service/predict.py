import os

from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)

from extras.utils import prepare_features
from extras.queries import prep_db, record_metrics_postgresql


POSTGRES_DBNAME = os.environ["POSTGRES_DBNAME"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
MLFLOW_MODEL_NAME = "clf-best-model"
MODEL_VERSION = "latest"

mlflow.set_tracking_uri(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DBNAME}"
)

# Load the model from the Model Registry
model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

reference_data = pd.read_csv("s3://data/train.csv")
reference_data = prepare_features(reference_data)
numerical_features = ["CreditScore", "Balance", "EstimatedSalary"]
categorical_features = [
    "HasCrCard",
    "IsActiveMember",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
]
column_mapping = ColumnMapping(
    prediction="Exited",
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    target=None,
)
report = Report(
    metrics=[
        ColumnDriftMetric(column_name="Exited"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)

prep_db()

def predict(features):
    current_data = features

    preds = model.predict(current_data)
    current_data["Exited"] = preds

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]

    record_metrics_postgresql(
        prediction_drift,
        num_drifted_columns,
        share_missing_values,
    )

    return int(preds[0])


app = Flask("churn-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    features = prepare_features(df)
    pred = predict(features)
    result = {"churn": pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
