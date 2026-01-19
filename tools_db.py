import pandas as pd
import os
import numpy as np
import openpyxl
import sqlite3
import json
from langchain.tools import tool


# sql db tools
def get_all_tables(db_path: str) -> list:
    """
    Retrieve a list of all table names in the SQLite database.

    This function queries the SQLite master table to get all table names,
    excluding system tables and other database objects.

    Args:
        db_path (str): The file path to the SQLite database file.

    Returns:
        list: A list of strings, where each string is a table name in the database.

    Raises:
        sqlite3.Error: If there's an issue connecting to or querying the database.

    Example:
        >>> tables = get_all_tables("mydatabase.db")
        >>> print(tables)
        ['users', 'orders', 'products']
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    conn.close()
    return tables

#change the db path inside
@tool
def execute_read_query(query: str) -> dict:
    """
    Execute a SELECT query and return the results with column headers.

    This function provides a safe way to execute read-only SELECT queries against
    the SQLite database. It only allows SELECT statements to prevent accidental
    data modification. The results include both the column names and the data rows.

    Args:
        query (str): A valid SELECT SQL query string. Must start with 'SELECT'.
        db_path (str, optional): The file path to the SQLite database file.
                                Defaults to "../output.db".

    Returns:
        dict: A dictionary with two keys:
            - 'headers': list of column names (strings)
            - 'data': list of tuples, where each tuple represents a row

    Raises:
        ValueError: If the query does not start with 'SELECT' (case-insensitive).
        sqlite3.Error: If there's an issue connecting to or querying the database.

    Example:
        >>> result = execute_read_query("SELECT name, age FROM users LIMIT 5")
        >>> print(result['headers'])
        ['name', 'age']
        >>> print(result['data'])
        [('John', 25), ('Jane', 30)]
    """
    db_path = r"C:\\Users\\krish\\OneDrive\\Desktop\\Chat-Bot-using-Streamlit-and-GoogleAI\\Heart_Disease.db"  # Change this to your database path
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed for safety reasons.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(query)
    data = cursor.fetchall()
    headers = [desc[0] for desc in cursor.description] if cursor.description else []

    conn.close()
    return {'headers': headers, 'data': data}


def get_columns(db_path: str, table_name: str) -> list:
    """
    Retrieve a list of column names from a specified SQLite table.

    This function uses the PRAGMA table_info command to get metadata about
    the table's columns, including their names, types, and constraints.

    Args:
        db_path (str): The file path to the SQLite database file.
        table_name (str): The name of the table to get column information for.

    Returns:
        list: A list of strings, where each string is a column name in the table.

    Raises:
        sqlite3.Error: If there's an issue connecting to the database or if
                      the table doesn't exist.

    Example:
        >>> columns = get_columns("users")
        >>> print(columns)
        ['id', 'name', 'email', 'created_at']
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name});")
    cols = [row[1] for row in cursor.fetchall()]  # row[1] = column name

    conn.close()
    return cols


def get_sample_rows(db_path: str, table_name: str, n: int = 2) -> list:
    """
    Retrieve the first n sample rows from a SQLite table.

    This function is useful for quickly inspecting the structure and content
    of a table without retrieving all rows. It returns all columns for the
    specified number of rows.

    Args:
        db_path (str): The file path to the SQLite database file.
        table_name (str): The name of the table to sample rows from.
        n (int, optional): The number of rows to retrieve. Defaults to 2.

    Returns:
        list: A list of tuples, where each tuple represents a row and contains
              all column values for that row.

    Raises:
        sqlite3.Error: If there's an issue connecting to the database or if
                      the table doesn't exist.

    Example:
        >>> rows = get_sample_rows("users", n=3)
        >>> print(len(rows))  # 3
        >>> print(rows[0])    # First row as tuple
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name} LIMIT {n};")
    rows = cursor.fetchall()

    conn.close()
    return rows