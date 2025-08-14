from __future__ import annotations
import os
import pendulum
import pandas as pd

from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.model_selection import train_test_split

DATA_PATH = "/opt/airflow/dags/files/airlines_raw.csv"
PROCESSED_PATH = "/opt/airflow/dags/files/airlines_prepared.csv"


@task
def extract_data() -> str:
    hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
    conn = hook.get_conn()
    df = pd.read_sql("SELECT * FROM airlines_data;", conn)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return DATA_PATH


@task
def select_features(input_path: str) -> str:
    df = pd.read_csv(input_path)

    selected_columns = [
        "airline",
        "departure_time",
        "duration",
        "stops",
        "days_left",
        "source_city",
        "destination_city",
        "class",
        "price"
    ]
    df = df[selected_columns]

    intermediate_path = "/opt/airflow/dags/files/airlines_selected.csv"
    df.to_csv(intermediate_path, index=False)
    return intermediate_path


@task
def clean_data(input_path: str) -> str:
    df = pd.read_csv(input_path)
    df_clean = df.dropna()

    cleaned_path = "/opt/airflow/dags/files/airlines_clean.csv"
    df_clean.to_csv(cleaned_path, index=False)
    return cleaned_path


@task
def encode_categoricals(input_path: str) -> str:
    df = pd.read_csv(input_path)

    categorical_columns = [
        "airline",
        "departure_time",
        "stops",
        "source_city",
        "destination_city",
        "class"
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    df_encoded.to_csv(PROCESSED_PATH, index=False)
    return PROCESSED_PATH

@task
def split_data(input_path: str):
    df = pd.read_csv(input_path)

    X = df.drop(columns=["price"])
    y = df["price"]

    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Second split: 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    os.makedirs("/opt/airflow/dags/files/", exist_ok=True)

    pd.concat([X_train, y_train], axis=1).to_csv("/opt/airflow/dags/files/train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv("/opt/airflow/dags/files/val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("/opt/airflow/dags/files/test.csv", index=False)

    return {
        "train_path": "/opt/airflow/dags/files/train.csv",
        "val_path": "/opt/airflow/dags/files/val.csv",
        "test_path": "/opt/airflow/dags/files/test.csv"
    }

@dag(
    dag_id="airlines_data_prepare_dag",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["ml", "preprocessing", "flights"],
)
def airlines_data_prepare_dag():
    raw = extract_data()
    selected = select_features(raw)
    cleaned = clean_data(selected)
    encoded = encode_categoricals(cleaned)
    split_data(encoded)

airlines_data_prepare_dag()

