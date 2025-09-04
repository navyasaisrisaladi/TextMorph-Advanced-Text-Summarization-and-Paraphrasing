import sqlite3
import bcrypt

# ----------------- Database Setup -----------------
def get_connection():
    conn = sqlite3.connect("users.db", check_same_thread=False)  # allows multi-thread use in Streamlit
    return conn, conn.cursor()

conn, c = get_connection()

c.execute('''CREATE TABLE IF NOT EXISTS users 
             (email TEXT PRIMARY KEY, password BLOB, name TEXT, age_group TEXT, language TEXT)''')
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
