from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Kolom identitas / bocoran / tak berguna -> dibuang
DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN", "Month",
             "Type_of_Loan"]

TARGET = "Credit_Score"

# Kolom yang seharusnya numerik (dibersihkan lalu di-cast ke float)
NUMERIC_RAW = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
]

# Kolom kategorikal + nilai placeholder yang harus dianggap kosong (NaN)
CATEGORICAL_RAW = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount",
                   "Payment_Behaviour"]
PLACEHOLDERS = {"_______", "_", "!@9#%8", "nan", "NaN", "", "NM", "__-333333333333333333333333333__"}

# Batas domain wajar; nilai di luar batas dianggap outlier -> NaN (lalu diimputasi)
CLIP_RULES = {
    "Age": (14, 100),
    "Annual_Income": (0, 500000),
    "Monthly_Balance": (0, 5000),
    "Num_Bank_Accounts": (0, 15),
    "Num_Credit_Card": (0, 15),
    "Interest_Rate": (0, 50),
    "Num_of_Loan": (0, 12),
    "Num_of_Delayed_Payment": (0, 60),
    "Num_Credit_Inquiries": (0, 40),
    "Total_EMI_per_month": (0, 100000),
}

# Fitur final yang dipakai model
NUMERIC_FEATURES = NUMERIC_RAW + ["Credit_History_Months", "Num_Loan_Types"]
CATEGORICAL_FEATURES = CATEGORICAL_RAW


def _to_number(series: pd.Series) -> pd.Series:
    """Buang semua karakter selain digit, titik, dan minus; lalu jadikan float."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    s = s.where(~s.isin(["", "-", "."]), np.nan)
    return pd.to_numeric(s, errors="coerce")


def _credit_history_to_months(series: pd.Series) -> pd.Series:
    """'20 Years and 5 Months' -> 245 (bulan)."""
    def parse(x):
        if not isinstance(x, str):
            return np.nan
        m = re.search(r"(\d+)\s*Years?.*?(\d+)\s*Months?", x)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        return np.nan
    return series.apply(parse)


def _count_loan_types(series: pd.Series) -> pd.Series:
    """Jumlah jenis pinjaman dari 'Type_of_Loan' (dipisah koma)."""
    def count(x):
        if not isinstance(x, str) or x.strip() in PLACEHOLDERS:
            return 0
        parts = [p for p in re.split(r",|and", x) if p.strip()
                 and "Not Specified" not in p]
        return len(parts)
    return series.apply(count)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # fitur turunan sebelum kolom mentah dibuang
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Months"] = _credit_history_to_months(df["Credit_History_Age"])
        df = df.drop(columns=["Credit_History_Age"])
    else:
        df["Credit_History_Months"] = np.nan

    if "Type_of_Loan" in df.columns:
        df["Num_Loan_Types"] = _count_loan_types(df["Type_of_Loan"])
    else:
        df["Num_Loan_Types"] = 0

    # bersihkan kolom numerik
    for col in NUMERIC_RAW:
        if col in df.columns:
            df[col] = _to_number(df[col])
            if col in CLIP_RULES:
                lo, hi = CLIP_RULES[col]
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # bersihkan kolom kategorikal (placeholder -> NaN)
    for col in CATEGORICAL_RAW:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].where(~df[col].isin(PLACEHOLDERS), np.nan)

    # buang kolom identitas / tak berguna
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return df


class RawDataCleaner(BaseEstimator, TransformerMixin):
    """Wrapper sklearn agar clean_dataframe menjadi bagian dari Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cleaned = clean_dataframe(df)
        # pastikan hanya kolom fitur yang diteruskan, urutan konsisten
        cols = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES)
                if c in cleaned.columns]
        return cleaned[cols]


if __name__ == "__main__":
    df = pd.read_csv("data_A.csv")
    print("Sebelum:", df.shape)
    cleaned = clean_dataframe(df)
    print("Sesudah:", cleaned.shape)
    print("\nKolom:", list(cleaned.columns))
    print("\nTipe data:")
    print(cleaned.dtypes)
    print("\nMissing setelah cleaning (akan diimputasi di pipeline):")
    m = cleaned.isna().sum()
    print(m[m > 0])
    print("\nStatistik numerik ringkas:")
    print(cleaned.describe().T[["mean", "min", "max"]].round(2))
