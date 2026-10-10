
import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "output" / "fraud_predictions.csv"
DB_PATH = BASE_DIR / "output" / "upi_fraud.db"

def create_database():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run fraud_detection.py first to generate predictions."
        )

    df = pd.read_csv(DATA_PATH)

    with sqlite3.connect(DB_PATH) as connection:
        df.to_sql(
            "transactions",
            connection,
            if_exists="replace",
            index=False
        )

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_risk_level
            ON transactions(risk_level)
        """)

        connection.commit()

        total = connection.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]

        high_risk = connection.execute("""
            SELECT COUNT(*)
            FROM transactions
            WHERE risk_level = 'High'
        """).fetchone()[0]

        fraud_cases = connection.execute("""
            SELECT COUNT(*)
            FROM transactions
            WHERE is_fraud = 1
        """).fetchone()[0]

        print("UPI FRAUD DATABASE CREATED")
        print("-" * 35)
        print(f"Database: {DB_PATH}")
        print(f"Transactions stored: {total}")
        print(f"High-risk transactions: {high_risk}")
        print(f"Actual fraud labels in sample: {fraud_cases}")

        print("\nHIGH-RISK TRANSACTIONS")
        query = """
            SELECT transaction_id, amount, risk_score, risk_level
            FROM transactions
            WHERE risk_level = 'High'
            ORDER BY risk_score DESC
        """
        print(pd.read_sql_query(query, connection).to_string(index=False))

if __name__ == "__main__":
    create_database()
