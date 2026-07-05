"""
store.py
All core business logic: customers, transactions, balances, and face
encoding storage. This file has zero camera/AI code in it — it only
knows how to talk to the database. face_utils.py does the AI work and
hands this file plain bytes/numbers to store.
"""

from db import get_connection, now


# ---------- Customers ----------

def add_customer(name: str, phone: str = None) -> int:
    """Create a new customer. Returns the new customer's id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO customers (name, phone, created_at) VALUES (%s, %s, %s)",
        (name.strip(), phone, now()),
    )
    conn.commit()
    customer_id = cur.lastrowid
    cur.close()
    conn.close()
    return customer_id


def get_customer(customer_id: int):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM customers WHERE id = %s", (customer_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def find_customers_by_name(name: str):
    """Case-insensitive partial match — useful since speech input
    (Phase 3) will produce imperfect text, so exact matching would
    break immediately."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM customers WHERE name LIKE %s ORDER BY name",
        (f"%{name.strip()}%",),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def list_all_customers():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM customers ORDER BY name")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# ---------- Face encoding storage (Phase 2) ----------

def save_face_encoding(customer_id: int, encoding_bytes: bytes):
    """Store a customer's face encoding (raw bytes, produced by
    face_utils.encoding_to_bytes). Overwrites any previous encoding —
    re-enrolling a customer replaces their old face data."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE customers SET face_encoding = %s WHERE id = %s",
        (encoding_bytes, customer_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_enrolled_customers():
    """All customers who have a face encoding on file. This is the pool
    that live camera recognition searches against."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, name, phone, face_encoding FROM customers "
        "WHERE face_encoding IS NOT NULL"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# ---------- Transactions ----------

def add_transaction(customer_id: int, type_: str, amount: float, description: str = ""):
    if type_ not in ("purchase", "payment"):
        raise ValueError("type must be 'purchase' or 'payment'")
    if amount <= 0:
        raise ValueError("amount must be positive")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO transactions (customer_id, type, amount, description, created_at)
           VALUES (%s, %s, %s, %s, %s)""",
        (customer_id, type_, amount, description, now()),
    )
    conn.commit()
    txn_id = cur.lastrowid
    cur.close()
    conn.close()
    return txn_id


def get_transactions(customer_id: int):
    """Transactions for one customer, with their name included via a
    join — handy so callers don't have to separately look up the name."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT t.*, c.name AS customer_name
           FROM transactions t
           JOIN customers c ON c.id = t.customer_id
           WHERE t.customer_id = %s
           ORDER BY t.created_at""",
        (customer_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_all_transactions():
    """Every transaction across every customer, newest first, with the
    customer's name included. This is the store-wide ledger view."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT t.*, c.name AS customer_name
           FROM transactions t
           JOIN customers c ON c.id = t.customer_id
           ORDER BY t.created_at DESC"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# ---------- Balance ----------

def get_balance(customer_id: int) -> float:
    """Pending balance = total purchases - total payments. Never stored,
    always calculated, so it can't drift from the transaction history."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT
             COALESCE(SUM(CASE WHEN type = 'purchase' THEN amount ELSE 0 END), 0) -
             COALESCE(SUM(CASE WHEN type = 'payment'  THEN amount ELSE 0 END), 0)
             AS balance
           FROM transactions WHERE customer_id = %s""",
        (customer_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return round(float(row["balance"]), 2)


def get_all_customers_with_balance():
    """The main list the store owner will look at: everyone plus what
    they currently owe."""
    customers = list_all_customers()
    result = []
    for c in customers:
        c["balance"] = get_balance(c["id"])
        result.append(c)
    return result


def get_customers_with_pending_dues():
    return [c for c in get_all_customers_with_balance() if c["balance"] > 0]