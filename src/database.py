import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join("data", "app_data.db")

def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS apps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            publisher TEXT,
            description TEXT,
            permissions TEXT,
            risk_score REAL,
            risk_reasons TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_app(app_data):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM apps WHERE name = ?", (app_data['name'],))
    existing = cursor.fetchone()
    
    if not existing:
        cursor.execute('''
            INSERT INTO apps (name, publisher, description, permissions, risk_score, risk_reasons)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            app_data['name'],
            app_data['publisher'],
            app_data['description'],
            app_data['permissions'],
            app_data['risk_score'],
            app_data['risk_reasons']
        ))
    conn.commit()
    conn.close()

def fetch_all_apps():
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM apps", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df