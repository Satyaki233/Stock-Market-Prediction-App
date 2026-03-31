from airflow.providers.postgres.hooks.postgres import PostgresHook

POSRGRES_CONN_ID = 'stock_db_conn'

class DBManager:
    def __init__(self, conn_id):
        self.conn_id = conn_id
        self.hook = PostgresHook(postgres_conn_id=self.conn_id)

    def execute_query(self, query):
        with self.hook.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                conn.commit()

    def get_hook(self):
        return self.hook


db_manager = DBManager(conn_id=POSRGRES_CONN_ID)