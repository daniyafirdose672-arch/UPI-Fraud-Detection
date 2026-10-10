# UPI Transaction Fraud Detection & Risk Scoring Engine

## Project Overview

This project demonstrates a machine learning-based approach to identifying potentially fraudulent UPI transactions. It analyzes transaction attributes, estimates fraud probabilities, assigns risk levels, and stores the results in an SQLite database for further investigation.

## Objectives

- Analyze UPI-style transaction data using Python and Pandas.
- Train a Logistic Regression classification model.
- Estimate fraud probabilities and calculate risk scores from 0 to 100.
- Categorize transactions into Low, Medium, and High risk levels.
- Store transaction predictions in an SQLite database.
- Generate summary reports and visualizations.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- SQLite
- SQL
- Matplotlib
- Seaborn

## Project Structure

```text
UPI-Fraud-Detection/
├── data/
│   └── transactions.csv
├── output/
│   ├── fraud_predictions.csv
│   ├── fraud_summary.csv
│   ├── upi_fraud.db
│   ├── transaction_distribution.png
│   ├── risk_level_distribution.png
│   └── top_risk_transactions.png
├── fraud_detection.py
├── database.py
├── visualize_fraud.py
└── README.md
```

## How It Works

1. Loads and validates the transaction dataset.
2. Splits the labeled data into training and testing sets.
3. Trains a Logistic Regression model using scaled features.
4. Evaluates the model using accuracy, precision, recall, F1-score, and a confusion matrix.
5. Generates fraud probabilities, risk scores, and risk categories.
6. Saves predictions and summary reports as CSV files.
7. Stores the results in SQLite and runs SQL queries to retrieve high-risk transactions.
8. Generates charts to visualize transaction labels and risk distributions.

## Risk Classification

- **High Risk:** Score greater than or equal to 70.
- **Medium Risk:** Score greater than or equal to 40 and below 70.
- **Low Risk:** Score below 40.

These thresholds are demonstration settings and would need validation before real-world use.

## Dataset

The project uses a small, synthetic dataset of 20 transactions. Features include transaction amount, transaction hour, location mismatch, failed attempts, and account age.

The `is_fraud` column contains example labels used to train and evaluate the model.

## Results

- Transactions processed: 20
- Example fraud-labeled transactions: 8
- Transactions categorized as High Risk: 7
- Test accuracy in this run: 100%

The test set contains only six records. Therefore, the reported accuracy is not a reliable estimate of performance on real transactions. The synthetic data is small and may contain patterns that make classification easier than it would be in practice.

## Limitations

- The dataset is synthetic and too small for meaningful real-world conclusions.
- Risk scores are model estimates, not proof of fraud.
- The model requires evaluation on a larger, representative dataset before practical deployment.
- This project does not connect to real UPI or banking systems and does not automatically block payments.

## Future Improvements

- Evaluate the model on a larger, realistic dataset.
- Compare Logistic Regression with Random Forest and other classification algorithms.
- Tune the decision threshold using precision-recall analysis.
- Add a dashboard for interactive fraud monitoring.
- Explore class imbalance handling and cost-sensitive evaluation.

## Disclaimer

This is an educational portfolio project. It is not a production fraud detection system and should not be used to make real financial decisions.