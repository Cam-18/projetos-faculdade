import mysql.connector

# Utiliza a biblioteca mysql.connector para conectar ao bd
def get_db_connection():
    return mysql.connector.connect(
        host = "localhost",
        user = "root",
        password = "",
        database = "pacientes"
    )
