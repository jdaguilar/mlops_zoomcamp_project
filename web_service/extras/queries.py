import datetime
import os

import psycopg2


POSTGRES_DBNAME = os.environ["POSTGRES_DBNAME"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
MLFLOW_MODEL_NAME = "clf-best-model"
MODEL_VERSION = "latest"

DROP_TABLE_QUERY = "drop table if exists evidently_metrics;"

CREATE_TABLE_QUERY = """
    create table public.evidently_metrics(
        timestamp timestamp,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_values float
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
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=5432,
        dbname=POSTGRES_DBNAME,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )

    curr = conn.cursor()
    curr.execute(DROP_TABLE_QUERY)
    curr.execute(CREATE_TABLE_QUERY)
    conn.commit()
    conn.close()


def record_metrics_postgresql(
    prediction_drift,
    num_drifted_columns,
    share_missing_values,
):
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=5432,
        dbname=POSTGRES_DBNAME,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )

    curr = conn.cursor()
    curr.execute(
        INSERT_RECORD_QUERY,
        (
            datetime.datetime.now(),
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
        ),
    )
    conn.commit()
    conn.close()