
# -*- coding: utf-8 -*-
import sqlite3, hashlib, os
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/catalog.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        sha256 TEXT NOT NULL UNIQUE,
        uploaded_at TEXT NOT NULL
    )
    """)
    con.commit()
    con.close()

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def add_file(filename: str, content: bytes) -> str:
    init_db()
    digest = sha256_bytes(content)
    stored_name = f"{digest}_{os.path.basename(filename)}"
    stored_path = Path("data/uploads") / stored_name
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stored_path, "wb") as f:
        f.write(content)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO files(filename, stored_path, sha256, uploaded_at) VALUES (?,?,?,?)",
                (filename, str(stored_path), digest, datetime.utcnow().isoformat()))
    con.commit(); con.close()
    return str(stored_path)

def list_files():
    init_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, filename, stored_path, uploaded_at FROM files ORDER BY uploaded_at DESC")
    rows = cur.fetchall()
    con.close()
    return rows

def get_path_by_id(file_id: int) -> str:
    init_db()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT stored_path FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else None
