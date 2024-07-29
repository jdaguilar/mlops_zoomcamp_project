import os

from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler


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


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=["id", "CustomerId", "Surname"])

    # Convert data types
    data["Age"] = data["Age"].astype(int)
    data["HasCrCard"] = data["HasCrCard"].astype(int)
    data["IsActiveMember"] = data["IsActiveMember"].astype(int)

    # One-hot encode categorical variables
    data["Geography_Germany"] = data.apply(
        lambda x: 1 if x["Geography"] == "Germany" else 0, axis=1
    )
    data["Geography_Spain"] = data.apply(
        lambda x: 1 if x["Geography"] == "Spain" else 0, axis=1
    )
    data["Gender_Male"] = data.apply(
        lambda x: 1 if x["Gender"] == "Male" else 0, axis=1
    )

    data = data.drop(columns=["Gender", "Geography"])

    scaler = StandardScaler()
    numerical_features = ["CreditScore", "Balance", "EstimatedSalary"]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


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
