from airflow.sdk import dag, task
from datetime import datetime, timedelta

@dag(
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data_ingestion"],
)
def data_ingestion_dag():
    @task
    def ingest_data():
        # Code to ingest data from source (e.g., API, database, etc.)
        print("Ingesting data...")

    ingest_data()   

data_ingestion_dag = data_ingestion_dag()   