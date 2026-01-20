import pandas as pd
import os
import numpy as np
import openpyxl
import sqlite3


sqlite_path = "heart.db"        # name of the sqlite file
table_name = "heart_disease_info"          # name of the table to create


df = pd.read_csv('Heart_Disease_Prediction.csv')
conn = sqlite3.connect(sqlite_path)



df.to_sql(table_name, conn, if_exists="replace", index=False)


cursor = conn.cursor()
cursor.execute(f"SELECT * FROM {table_name} LIMIT 1;")
rows = cursor.fetchall()
print("Sample rows:", rows)

conn.close()
print("Done! Data written to", sqlite_path)