-- Creates the stocks database if it doesn't already exist.
-- This script runs automatically when the postgres container starts for the first time.
SELECT 'CREATE DATABASE stocks_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'stocks_db')\gexec
