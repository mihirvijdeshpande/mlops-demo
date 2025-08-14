from __future__ import annotations
import os
import logging
import pendulum
import requests

from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook


@task
def download_csv():
    url = "https://www.kaggle.com/api/v1/datasets/download/rohitgrewal/airlines-flights-data/airlines_flights_data.csv"
    target_csv = "/opt/airflow/dags/files/airlines.csv"

    os.makedirs(os.path.dirname(target_csv), exist_ok=True)

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download CSV: {response.status_code}")

    with open(target_csv, "wb") as f:
        f.write(response.content)

    logging.info(f"Downloaded CSV to {target_csv}")
    return target_csv

@task
def create_tables():
    create_main_table = """
    CREATE TABLE IF NOT EXISTS airlines_data (
        index INTEGER PRIMARY KEY,
        airline TEXT,
        flight TEXT,
        source_city TEXT,
        departure_time TEXT,
        stops TEXT,
        arrival_time TEXT,
        destination_city TEXT,
        class TEXT,
        duration TEXT,
        days_left INTEGER,
        price INTEGER
    );
    """

    create_temp_table = """
    DROP TABLE IF EXISTS airlines_data_temp;
    CREATE TABLE airlines_data_temp (
        index INTEGER PRIMARY KEY,
        airline TEXT,
        flight TEXT,
        source_city TEXT,
        departure_time TEXT,
        stops TEXT,
        arrival_time TEXT,
        destination_city TEXT,
        class TEXT,
        duration TEXT,
        days_left INTEGER,
        price INTEGER
    );
    """

    hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
    with hook.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(create_main_table)
            cur.execute(create_temp_table)
        conn.commit()
    logging.info("Created airlines_data and airlines_data_temp tables.")


@task
def load_temp_table(csv_path: str):
    hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
    with hook.get_conn() as conn:
        with conn.cursor() as cur, open(csv_path, "r") as file:
            cur.copy_expert("""
                COPY airlines_data_temp FROM STDIN WITH CSV HEADER DELIMITER ',' QUOTE '"';
            """, file)
        conn.commit()
    logging.info("Data loaded into airlines_data_temp.")


@task
def merge_into_main():
    query = """
        INSERT INTO airlines_data
        SELECT * FROM airlines_data_temp
        ON CONFLICT (index) DO UPDATE SET
            airline = EXCLUDED.airline,
            flight = EXCLUDED.flight,
            source_city = EXCLUDED.source_city,
            departure_time = EXCLUDED.departure_time,
            stops = EXCLUDED.stops,
            arrival_time = EXCLUDED.arrival_time,
            destination_city = EXCLUDED.destination_city,
            class = EXCLUDED.class,
            duration = EXCLUDED.duration,
            days_left = EXCLUDED.days_left,
            price = EXCLUDED.price;
    """

    hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
    with hook.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()
    logging.info("Merged data from temp table into airlines_data.")


@dag(
    dag_id="airlines_data_import_dag",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["flights", "postgres", "kaggle"],
)
def airlines_data_import_dag():
    csv_path = download_csv()
    create_tables() >> load_temp_table(csv_path) >> merge_into_main()


airlines_data_import_dag()

