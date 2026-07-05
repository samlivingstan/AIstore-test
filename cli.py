"""
cli.py
A menu-driven command line tool to test store.py without needing any
UI, camera, or microphone. Run this first to add customers and record
transactions before enrolling faces.

Run:  python cli.py
"""

from db import init_db
import store


def print_customer_line(c):
    due = f"Rs.{c['balance']:.2f}" if c["balance"] > 0 else "no dues"
    phone = c["phone"] or "-"
    face = "yes" if c.get("face_encoding") else "no"
    print(f"  [{c['id']}] {c['name']:<20} phone: {phone:<12} balance: {due:<15} face enrolled: {face}")


def menu():
    print("""
--- Grocery Credit System ---
1. Add customer
2. List all customers (with balance)
3. Record a purchase (customer buys on credit)
4. Record a payment
5. View one customer's transaction history
6. List customers with pending dues
7. View all transactions (every customer, with names)
0. Exit
""")


def main():
    init_db()
    while True:
        menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            name = input("Customer name: ").strip()
            phone = input("Phone (optional): ").strip() or None
            cid = store.add_customer(name, phone)
            print(f"Added '{name}' with id {cid}")

        elif choice == "2":
            customers = store.get_all_customers_with_balance()
            if not customers:
                print("No customers yet.")
            for c in customers:
                print_customer_line(c)

        elif choice == "3":
            cid = int(input("Customer id: "))
            amount = float(input("Amount: "))
            desc = input("Description (e.g. items bought): ").strip()
            store.add_transaction(cid, "purchase", amount, desc)
            print(f"Recorded purchase. New balance: Rs.{store.get_balance(cid):.2f}")

        elif choice == "4":
            cid = int(input("Customer id: "))
            amount = float(input("Amount paid: "))
            store.add_transaction(cid, "payment", amount, "payment received")
            print(f"Recorded payment. New balance: Rs.{store.get_balance(cid):.2f}")

        elif choice == "5":
            cid = int(input("Customer id: "))
            customer = store.get_customer(cid)
            if not customer:
                print("No such customer.")
                continue
            print(f"\nHistory for {customer['name']}:")
            for t in store.get_transactions(cid):
                sign = "+" if t["type"] == "purchase" else "-"
                print(f"  {t['created_at']}  {sign}Rs.{t['amount']:.2f}  ({t['type']})  {t['description']}")
            print(f"Current balance: Rs.{store.get_balance(cid):.2f}")

        elif choice == "6":
            pending = store.get_customers_with_pending_dues()
            if not pending:
                print("Nobody owes anything right now.")
            for c in pending:
                print_customer_line(c)

        elif choice == "7":
            transactions = store.get_all_transactions()
            if not transactions:
                print("No transactions recorded yet.")
            for t in transactions:
                sign = "+" if t["type"] == "purchase" else "-"
                print(f"  {t['created_at']}  {t['customer_name']:<20} "
                      f"{sign}Rs.{t['amount']:.2f}  ({t['type']})  {t['description']}")

        elif choice == "0":
            print("Bye.")
            break

        else:
            print("Not a valid option, try again.")


if __name__ == "__main__":
    main()