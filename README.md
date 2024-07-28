# mlops_zoomcamp_project

## Problem description

This repository is an approximation to develop a solution for the Kaggle competition called [Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1/data?select=test.csv)

The goal is to build a MLOps platform to train, evaluate, save a deploy a classification model to predict the churn of a client.

## Dataset

> The bank customer churn dataset is a commonly used dataset for predicting customer churn in the banking industry. It contains information on bank customers who either left the bank or continue to be a customer. The dataset includes the following attributes:

| Field | Description |
|--|--|
|Customer ID| A unique identifier for each customer|
|Surname| The customer's surname or last name|
|Credit Score| A numerical value representing the customer's credit score|
|Geography| The country where the customer resides (France, Spain or Germany)|
|Gender| The customer's gender (Male or Female)|
|Age| The customer's age.|
|Tenure| The number of years the customer has been with the bank|
|Balance| The customer's account balance|
|NumOfProducts| The number of bank products the customer uses (e.g., savings account, credit card)|
|HasCrCard| Whether the customer has a credit card (1 = yes, 0 = no)|
|IsActiveMember| Whether the customer is an active member (1 = yes, 0 = no)|
|EstimatedSalary| The estimated salary of the customer|
|Exited| Whether the customer has churned (1 = yes, 0 = no)|

Reference: [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)

## Cloud

:warning: In order to save cost, this project is not deployed in cloud using IaC o similar tools. However, some services were adapted to be used by containers.

This project use Docker Compose to orchestrate a cluster with its own networks. The services used are the following:

- PostgreSQL: Database server which is used by MLFlow to save metadata.

- MinIO: Open source alternative to AWS S3, it's crucial to save datasets and the artifacts of MLFlow.

- Mage: Workflow Orchestration service to run the ML pipeline.

- MLFlow: Service used for Experiment tracking and model registry. It was adapted to use PostgreSQL database and MinIO server.


Model deployment
- 0 points: Model is not deployed
- 2 points: Model is deployed but only locally
- 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used

Model monitoring

- 0 points: No model monitoring
- 2 points: Basic model monitoring that calculates and reports metrics
- 4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated

Reproducibility

- 0 points: No instructions on how to run the code at all, the data is missing

- 2 points: Some instructions are there, but they are not complete OR instructions are clear and complete, the code works, but the data is missing

- 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.

Best practices
- There are unit tests (1 point)
- There is an integration test (1 point)
- Linter and/or code formatter are used (1 point)
- There's a Makefile (1 point)
- There are pre-commit hooks (1 point)
- There's a CI/CD pipeline (2 points)
