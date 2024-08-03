import pandas as pd

from sklearn.preprocessing import StandardScaler


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
