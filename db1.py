import sqlite3
import bcrypt
from datetime import datetime

# ----------------- Database Setup -----------------
def get_connection():
    conn = sqlite3.connect("users.db", check_same_thread=False)  # allows multi-thread use in Streamlit
    return conn, conn.cursor()

conn, c = get_connection()

# Users table
c.execute('''CREATE TABLE IF NOT EXISTS users 
             (email TEXT PRIMARY KEY, password BLOB, name TEXT, age_group TEXT, language TEXT)''')

# Logs table
c.execute('''CREATE TABLE IF NOT EXISTS logs
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT,
              input_text TEXT,
              task_type TEXT,
              model_used TEXT,
              output_text TEXT,
              timestamp TEXT)''')

conn.commit()

# ----------------- Helper Functions -----------------
def register_user(email, password):
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pw))
    conn.commit()

def login_user(email, password):
    c.execute("SELECT password FROM users WHERE email=?", (email,))
    result = c.fetchone()
    if result and bcrypt.checkpw(password.encode(), result[0]):
        return True
    return False

def save_profile(email, name, age_group, language):
    c.execute("UPDATE users SET name=?, age_group=?, language=? WHERE email=?",
              (name, age_group, language, email))
    conn.commit()

def get_all_users():
    c.execute("SELECT email, name, age_group, language FROM users")
    return c.fetchall()

# ----------------- Logging Functions -----------------
def log_request(email, input_text, task_type, model_used, output_text):
    """Log a summarization or paraphrasing request"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO logs (email, input_text, task_type, model_used, output_text, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (email, input_text, task_type, model_used, output_text, timestamp))
    conn.commit()

def get_logs(email):
    """Fetch logs for a specific user"""
    c.execute("SELECT task_type, model_used, input_text, output_text, timestamp FROM logs WHERE email=? ORDER BY id DESC", (email,))
    return c.fetchall()
