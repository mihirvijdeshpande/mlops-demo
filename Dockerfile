# custom image for installed mlflow
FROM apache/airflow:3.0.3

USER airflow
RUN pip install mlflow

# USER airflow
