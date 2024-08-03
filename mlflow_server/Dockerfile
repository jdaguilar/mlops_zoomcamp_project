# https://github.com/minio/blog-assets/blob/main/mlflow-minio-setup/mlflow/Dockerfile

FROM python:3.12.4

RUN apt-get update && apt-get install -y curl
RUN pip install mlflow[extras]==2.15.0 psycopg2-binary boto3 cryptography pymysql

EXPOSE 5000
