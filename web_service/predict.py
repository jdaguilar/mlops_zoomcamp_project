import os

from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

from extras.utils import prepare_features
from extras.queries import prep_db, record_predictions


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

prep_db()


def predict(features):
    current_data = features

    # Return predicted label
    y_pred = model.predict(current_data)
    y_pred = int(y_pred[0])

    # Return predicted probability
    y_pred_prob = model.predict_proba(current_data)
    y_pred_prob = max(y_pred_prob[0])

    return y_pred, y_pred_prob


app = Flask("churn-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    features = prepare_features(df)
    y_pred, y_pred_prob = predict(features)
    record_predictions(df, y_pred, y_pred_prob)

    result = {
        "churn": y_pred,
        "prob": y_pred_prob,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
