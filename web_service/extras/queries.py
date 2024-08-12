import datetime
import os

import mysql.connector
from sqlalchemy import create_engine


MYSQL_DATABASE = os.environ["MYSQL_DATABASE"]
MYSQL_USER = os.environ["MYSQL_USER"]
MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]
MYSQL_HOST = os.environ["MYSQL_HOST"]

MLFLOW_MODEL_NAME = "clf-best-model"
MODEL_VERSION = "latest"


CREATE_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS prediction (
        timestamp DATETIME,
        id bigint,
        CustomerId bigint,
        Surname VARCHAR(100),
        CreditScore bigint,
        Geography VARCHAR(100),
        Gender VARCHAR(100),
        Age bigint,
        Tenure bigint,
        Balance bigint,
        NumOfProducts bigint,
        HasCrCard bigint,
        IsActiveMember TINYINT,
        EstimatedSalary double,
        Exited TINYINT,
        ExitedProbability double
    )
"""

INSERT_RECORD_QUERY = """
    insert into public.evidently_metrics(
        timestamp, prediction_drift, num_drifted_columns, share_missing_values
    ) values (
        %s, %s, %s, %s
    )
"""


def prep_db():

    cnx = mysql.connector.connect(
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        host=MYSQL_HOST,
        database=MYSQL_DATABASE,
    )
    cursor = cnx.cursor()

    try:
        cursor.execute(CREATE_TABLE_QUERY)
        cnx.close()
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit()


def record_predictions(df, y_pred, y_pred_prob):
    df["Exited"] = y_pred
    df["ExitedProbability"] = y_pred_prob
    df["timestamp"] = datetime.datetime.now()

    engine = create_engine(
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
    )

    df.to_sql("prediction", con=engine, if_exists="append", index=False)
