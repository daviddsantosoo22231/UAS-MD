"""
inference.py
------------
Kode inferencing untuk model klasifikasi credit score (Poor / Standard / Good).
Memuat model.pkl (berisi pipeline lengkap: cleaning + preprocessing + estimator,
serta label encoder) lalu memprediksi untuk data MENTAH — kolom kotor otomatis
dibersihkan oleh pipeline.

Dipakai oleh app.py (Streamlit) dan untuk pengujian test case (blok __main__).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


class CreditScoreModel:
    """Wrapper inferencing untuk model credit score multi-kelas."""

    def __init__(self, model_path: str = "model.pkl",
                 metadata_path: str = "model_metadata.json"):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.pipeline = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.classes = list(bundle["classes"])

        meta = Path(metadata_path)
        self.metadata = json.loads(meta.read_text()) if meta.exists() else {}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        pred_enc = self.pipeline.predict(df)
        pred = self.label_encoder.inverse_transform(pred_enc)
        proba = self.pipeline.predict_proba(df)
        out = df.copy()
        out["prediction"] = pred
        for i, cls in enumerate(self.classes):
            out[f"prob_{cls}"] = proba[:, i].round(4)
        return out

    def predict_one(self, record: dict) -> dict:
        df = pd.DataFrame([record])
        pred_enc = int(self.pipeline.predict(df)[0])
        pred = self.label_encoder.inverse_transform([pred_enc])[0]
        proba = self.pipeline.predict_proba(df)[0]
        probs = {cls: round(float(proba[i]), 4)
                 for i, cls in enumerate(self.classes)}
        decision = {
            "Good": "SETUJUI",
            "Standard": "SETUJUI dengan syarat / tinjau",
            "Poor": "TOLAK / risiko tinggi",
        }.get(pred, "-")
        return {"prediction": pred, "decision": decision, "probabilities": probs}


# --------------------------------------------------------------------------
# Test case: satu contoh nyata (data mentah) untuk SETIAP kelas.
# Diambil dari dataset & telah diverifikasi diprediksi benar oleh model.
# --------------------------------------------------------------------------
GOOD_EXAMPLE = {"Age": "49", "Occupation": "Entrepreneur", "Annual_Income": "177723.8", "Monthly_Inhand_Salary": 15038.32, "Num_Bank_Accounts": 3, "Num_Credit_Card": 4, "Interest_Rate": 1, "Num_of_Loan": "3", "Type_of_Loan": "Mortgage Loan, Student Loan, and Student Loan", "Delay_from_due_date": -2, "Num_of_Delayed_Payment": "12", "Changed_Credit_Limit": "16.6", "Num_Credit_Inquiries": 2.0, "Credit_Mix": "Good", "Outstanding_Debt": "1292.23", "Credit_Utilization_Ratio": 43.42, "Credit_History_Age": "29 Years and 11 Months", "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 338.79, "Amount_invested_monthly": "221.92", "Payment_Behaviour": "High_spent_Large_value_payments", "Monthly_Balance": "1183.12"}

STANDARD_EXAMPLE = {"Age": "29", "Occupation": "Musician", "Annual_Income": "142081.48", "Monthly_Inhand_Salary": 11771.12, "Num_Bank_Accounts": 6, "Num_Credit_Card": 6, "Interest_Rate": 15, "Num_of_Loan": "3", "Type_of_Loan": "Personal Loan, Not Specified, and Mortgage Loan", "Delay_from_due_date": 29, "Num_of_Delayed_Payment": "20", "Changed_Credit_Limit": "8.21", "Num_Credit_Inquiries": 4.0, "Credit_Mix": "Standard", "Outstanding_Debt": "932.32", "Credit_Utilization_Ratio": 28.41, "Credit_History_Age": "27 Years and 0 Months", "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 180.62, "Amount_invested_monthly": "423.96", "Payment_Behaviour": "Low_spent_Small_value_payments", "Monthly_Balance": "862.53"}

POOR_EXAMPLE = {"Age": "25", "Occupation": "Developer", "Annual_Income": "33119.82", "Monthly_Inhand_Salary": 3024.99, "Num_Bank_Accounts": 10, "Num_Credit_Card": 5, "Interest_Rate": 32, "Num_of_Loan": "5", "Type_of_Loan": "Home Equity Loan, Mortgage Loan, Home Equity Loan, Debt Consolidation Loan, and Not Specified", "Delay_from_due_date": 50, "Num_of_Delayed_Payment": "18", "Changed_Credit_Limit": "15.79", "Num_Credit_Inquiries": 10.0, "Credit_Mix": "Bad", "Outstanding_Debt": "2545.0", "Credit_Utilization_Ratio": 33.81, "Credit_History_Age": "14 Years and 8 Months", "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 111.42, "Amount_invested_monthly": None, "Payment_Behaviour": "Low_spent_Small_value_payments", "Monthly_Balance": "370.61"}

TEST_CASES = {"Good": GOOD_EXAMPLE, "Standard": STANDARD_EXAMPLE, "Poor": POOR_EXAMPLE}


if __name__ == "__main__":
    model = CreditScoreModel()
    print("Model terbaik:", model.metadata.get("best_model"),
          "| Kelas:", model.classes, "\n")
    for expected, rec in TEST_CASES.items():
        res = model.predict_one(rec)
        ok = "OK" if res["prediction"] == expected else "!! beda"
        print(f"=== TEST CASE: harusnya '{expected}'  [{ok}] ===")
        print(json.dumps(res, indent=2), "\n")
