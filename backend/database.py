# database.py
import psycopg2
import psycopg2.extras


def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="resume_analysis",   # nama database kamu
        user="postgres",
        password="andreas11"
    )
