import requests

churn_params = {
  "id": 165034,
  "CustomerId": 15773898,
  "Surname": "Lucchese",
  "CreditScore": 586,
  "Geography": "France",
  "Gender": "Female",
  "Age": 23,
  "Tenure": 2,
  "Balance": 0,
  "NumOfProducts": 2,
  "HasCrCard": 0,
  "IsActiveMember": 1,
  "EstimatedSalary": 160976.75
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=churn_params)
print(response.json())
