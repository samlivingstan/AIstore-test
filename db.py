"""
db.py
Database setup for the grocery store customer credit system, using MySQL.

Design:
- customers: one row per customer. face_encoding is left NULL for now —
  it gets filled in during Phase 2 (face recognition). Stored as BLOB
  (bytes) since a face encoding is just a numeric vector.
- transactions: a ledger. Every purchase-on-credit and every payment is
  a row. A customer's pending balance = sum(purchases) - sum(payments).
  We derive the balance instead of storing it directly so it can never
  drift out of sync with the history.

Connection settings are read from environment variables so you never
have to hardcode credentials in code:
  DB_HOST      (default: localhost)
  DB_PORT      (default: 3306)
  DB_USER      (default: root)
  DB_PASSWORD  (default: "")
  DB_NAME      (default: grocery_credit)

Set them before running, e.g.:
  export DB_USER=root
  export DB_PASSWORD=yourpassword
"""

import os
import mysql.connector
from mysql.connector import errorcode
from datetime import datetime

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "admin")
DB_NAME = os.environ.get("DB_NAME", "grocery_credit")


def get_connection(use_database: bool = True):
    """Open a new MySQL connection. use_database=False is only used
    internally, during first-time database creation."""
    config = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD)
    if use_database:
        config["database"] = DB_NAME
    return mysql.connector.connect(**config)


def init_db():
    """Create the database (if missing) and tables (if missing).
    Safe to run every startup."""

    # Step 1: make sure the database itself exists
    conn = get_connection(use_database=False)
    cur = conn.cursor()
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    conn.commit()
    cur.close()
    conn.close()

    # Step 2: create tables inside that database
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            phone VARCHAR(20),
            face_encoding BLOB,          -- filled in later (Phase 2)
            created_at DATETIME NOT NULL
        ) ENGINE=InnoDB
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id INT NOT NULL,
            type ENUM('purchase', 'payment') NOT NULL,
            amount DECIMAL(10, 2) NOT NULL,
            description TEXT,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            CONSTRAINT chk_amount_positive CHECK (amount > 0)
        ) ENGINE=InnoDB
    """)

    conn.commit()
    cur.close()
    conn.close()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    init_db()
    print(f"Database '{DB_NAME}' initialized on {DB_HOST}:{DB_PORT}")