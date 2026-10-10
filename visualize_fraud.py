
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "output" / "fraud_predictions.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 1. Actual transaction label distribution
label_counts = df["is_fraud"].map({
    0: "Legitimate",
    1: "Fraud"
}).value_counts()

plt.figure(figsize=(7, 5))
label_counts.plot(kind="bar")
plt.title("UPI Transaction Label Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transaction_distribution.png")
plt.close()

# 2. Model-assigned risk levels
risk_order = ["Low", "Medium", "High"]
risk_counts = (
    df["risk_level"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
)

plt.figure(figsize=(7, 5))
risk_counts.plot(kind="bar")
plt.title("Transaction Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "risk_level_distribution.png")
plt.close()

# 3. Ten transactions with the highest risk scores
top_risks = df.nlargest(10, "risk_score").sort_values("risk_score")

plt.figure(figsize=(9, 6))
plt.barh(top_risks["transaction_id"], top_risks["risk_score"])
plt.title("Top 10 Transactions by Fraud Risk Score")
plt.xlabel("Risk Score (0–100)")
plt.ylabel("Transaction ID")
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_risk_transactions.png")
plt.close()

print("Visualizations created successfully:")
print(OUTPUT_DIR / "transaction_distribution.png")
print(OUTPUT_DIR / "risk_level_distribution.png")
print(OUTPUT_DIR / "top_risk_transactions.png")
