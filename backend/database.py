import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="resumatch",
        user="postgres",
        password="asmforeal11"
    )
