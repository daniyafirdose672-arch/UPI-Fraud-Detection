
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "transactions.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load transaction data
df = pd.read_csv(DATA_PATH)

print("UPI FRAUD DETECTION ENGINE")
print("-" * 40)
print(f"Total transactions: {len(df)}")

# Validate required columns
required_columns = [
    "transaction_id", "amount", "transaction_hour",
    "location_mismatch", "failed_attempts",
    "account_age_days", "is_fraud"
]

missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if df[required_columns].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

if df["transaction_id"].duplicated().any():
    raise ValueError("Duplicate transaction IDs found.")

# Select model features and target
features = [
    "amount",
    "transaction_hour",
    "location_mismatch",
    "failed_attempts",
    "account_age_days"
]

X = df[features]
y = df["is_fraud"]

if set(y.unique()) != {0, 1}:
    raise ValueError("is_fraud must contain both 0 and 1 labels.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Scale numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), features)
    ]
)

# Build the machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate on test data
predictions = model.predict(X_test)

print("\nMODEL EVALUATION")
print("-" * 40)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Score all transactions
df["fraud_probability"] = model.predict_proba(X)[:, 1]
df["risk_score"] = (df["fraud_probability"] * 100).round(2)

def assign_risk(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"

df["risk_level"] = df["risk_score"].apply(assign_risk)
df["model_prediction"] = model.predict(X)

# Save transaction-level results
results_path = OUTPUT_DIR / "fraud_predictions.csv"
df.to_csv(results_path, index=False)

# Create summary report
summary = pd.DataFrame({
    "Metric": [
        "Total transactions",
        "Actual fraud cases in sample",
        "Model-flagged transactions",
        "High-risk transactions",
        "Medium-risk transactions",
        "Low-risk transactions"
    ],
    "Value": [
        len(df),
        int(df["is_fraud"].sum()),
        int(df["model_prediction"].sum()),
        int((df["risk_level"] == "High").sum()),
        int((df["risk_level"] == "Medium").sum()),
        int((df["risk_level"] == "Low").sum())
    ]
})

summary_path = OUTPUT_DIR / "fraud_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nTRANSACTION RISK SUMMARY")
print(summary.to_string(index=False))
print(f"\nDetailed report saved to: {results_path}")
print(f"Summary report saved to: {summary_path}")
