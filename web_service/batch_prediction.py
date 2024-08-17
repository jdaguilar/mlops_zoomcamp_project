import os
import json
import pandas as pd

import requests


WEB_SERVICE_URL = "http://localhost:9696/predict"


def send_params_to_predictions_api(args):

    headers_list = {
        "Content-Type": "application/json"
    }

    response = requests.request(
        "POST",
        WEB_SERVICE_URL,
        data=json.dumps(args),
        headers=headers_list
    )

    print(response.text)


if __name__ == "__main__":

    data = pd.read_csv("data/test.csv")

    data["results"] = data.apply(
        lambda s: send_params_to_predictions_api(s.to_dict()),
        axis=1,
    )
