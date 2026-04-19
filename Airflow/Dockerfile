FROM apache/airflow:3.1.7

USER root

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install uv — fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .

# Install as root so uv can write to the system Python at /usr/python
RUN uv pip install --system -r requirements.txt

# Install mongo provider separately with --no-deps to avoid the
# mlflow packaging<25 vs airflow packaging>=25 conflict
RUN uv pip install --system --no-deps apache-airflow-providers-mongo==5.3.5

USER airflow
