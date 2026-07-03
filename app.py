"""
app.py
------
Aplikasi web deployment (Streamlit) untuk model klasifikasi credit score
(Poor / Standard / Good). Menampilkan form input data nasabah, lalu memberi
prediksi kelas beserta probabilitas tiap kelas.

Jalankan lokal :  streamlit run app.py
Deploy publik  :  push ke GitHub -> https://share.streamlit.io (main file: app.py)
Deploy AWS     :  lihat README (App Runner / EC2).
"""
import os
import gdown
import pandas as pd
import streamlit as st

from inference import CreditScoreModel

st.set_page_config(page_title="Credit Score Classification", page_icon="💳",
                   layout="wide")

OCCUPATIONS = ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
               "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
               "Accountant", "Musician", "Mechanic", "Writer", "Architect"]
CREDIT_MIX = ["Good", "Standard", "Bad"]
PAY_MIN = ["Yes", "No", "NM"]
BEHAVIOUR = ["High_spent_Large_value_payments", "High_spent_Medium_value_payments",
             "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
             "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"]

# preset (dalam ruang widget) untuk tiap kelas — konsisten dgn test case
PRESETS = {
    "Good": dict(age=49, occ="Entrepreneur", income=177723.8, salary=15038.32,
                 banks=3, cards=4, rate=1, loans=3, delay=-2, delayed=12,
                 chg=16.6, inq=2, mix="Good", debt=1292.23, util=43.42,
                 hy=29, hm=11, paymin="No", emi=338.79, invest=221.92,
                 beh="High_spent_Large_value_payments", bal=1183.12, ltypes=3),
    "Standard": dict(age=29, occ="Musician", income=142081.48, salary=11771.12,
                     banks=6, cards=6, rate=15, loans=3, delay=29, delayed=20,
                     chg=8.21, inq=4, mix="Standard", debt=932.32, util=28.41,
                     hy=27, hm=0, paymin="No", emi=180.62, invest=423.96,
                     beh="Low_spent_Small_value_payments", bal=862.53, ltypes=2),
    "Poor": dict(age=25, occ="Developer", income=33119.82, salary=3024.99,
                 banks=10, cards=5, rate=32, loans=5, delay=50, delayed=18,
                 chg=15.79, inq=10, mix="Bad", debt=2545.0, util=33.81,
                 hy=14, hm=8, paymin="Yes", emi=111.42, invest=0.0,
                 beh="Low_spent_Small_value_payments", bal=370.61, ltypes=4),
}


@st.cache_resource
def load_model():
    return CreditScoreModel("model.pkl")


model = load_model()

st.title("💳 Credit Score Classification")
st.caption(f"Model: **{model.metadata.get('best_model', 'n/a')}** · "
           f"F1-macro: **{model.metadata.get('best_f1_macro', 'n/a')}** · "
           f"Kelas: {', '.join(model.classes)}")

if "preset" not in st.session_state:
    st.session_state.preset = PRESETS["Good"]

st.write("Pilih contoh cepat, atau isi manual, lalu klik **Prediksi**.")
b1, b2, b3 = st.columns(3)
if b1.button("🟢 Contoh: GOOD"):
    st.session_state.preset = PRESETS["Good"]
if b2.button("🟡 Contoh: STANDARD"):
    st.session_state.preset = PRESETS["Standard"]
if b3.button("🔴 Contoh: POOR"):
    st.session_state.preset = PRESETS["Poor"]
p = st.session_state.preset

st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Profil**")
    age = st.number_input("Usia", 14, 100, int(p["age"]))
    occ = st.selectbox("Pekerjaan", OCCUPATIONS,
                       index=OCCUPATIONS.index(p["occ"]) if p["occ"] in OCCUPATIONS else 0)
    income = st.number_input("Pendapatan tahunan", 0.0, 500000.0, float(p["income"]))
    salary = st.number_input("Gaji bulanan (in-hand)", 0.0, 20000.0, float(p["salary"]))
    hy = st.number_input("Riwayat kredit (tahun)", 0, 40, int(p["hy"]))
    hm = st.number_input("Riwayat kredit (bulan)", 0, 11, int(p["hm"]))
    ltypes = st.number_input("Jumlah jenis pinjaman", 0, 10, int(p["ltypes"]))
with c2:
    st.markdown("**Akun & Pinjaman**")
    banks = st.number_input("Jumlah rekening bank", 0, 15, int(p["banks"]))
    cards = st.number_input("Jumlah kartu kredit", 0, 15, int(p["cards"]))
    rate = st.number_input("Suku bunga (%)", 0, 50, int(p["rate"]))
    loans = st.number_input("Jumlah pinjaman", 0, 12, int(p["loans"]))
    debt = st.number_input("Outstanding debt", 0.0, 5000.0, float(p["debt"]))
    emi = st.number_input("Total EMI per bulan", 0.0, 50000.0, float(p["emi"]))
    inq = st.number_input("Jumlah credit inquiries", 0, 40, int(p["inq"]))
with c3:
    st.markdown("**Perilaku Pembayaran**")
    delay = st.number_input("Delay dari jatuh tempo (hari)", -10, 100, int(p["delay"]))
    delayed = st.number_input("Jumlah pembayaran terlambat", 0, 60, int(p["delayed"]))
    chg = st.number_input("Perubahan limit kredit", -50.0, 50.0, float(p["chg"]))
    util = st.number_input("Rasio utilisasi kredit (%)", 0.0, 100.0, float(p["util"]))
    invest = st.number_input("Investasi bulanan", 0.0, 10000.0, float(p["invest"]))
    bal = st.number_input("Saldo bulanan", 0.0, 5000.0, float(p["bal"]))
    mix = st.selectbox("Credit mix", CREDIT_MIX,
                       index=CREDIT_MIX.index(p["mix"]) if p["mix"] in CREDIT_MIX else 0)
    paymin = st.selectbox("Bayar minimum?", PAY_MIN,
                          index=PAY_MIN.index(p["paymin"]))
    beh = st.selectbox("Payment behaviour", BEHAVIOUR,
                       index=BEHAVIOUR.index(p["beh"]) if p["beh"] in BEHAVIOUR else 0)

st.divider()

if st.button("🔍 Prediksi Credit Score", type="primary", use_container_width=True):
    record = {
        "Age": age, "Occupation": occ, "Annual_Income": income,
        "Monthly_Inhand_Salary": salary, "Num_Bank_Accounts": banks,
        "Num_Credit_Card": cards, "Interest_Rate": rate, "Num_of_Loan": loans,
        "Type_of_Loan": ", ".join(["Personal Loan"] * int(ltypes)) or "Not Specified",
        "Delay_from_due_date": delay, "Num_of_Delayed_Payment": delayed,
        "Changed_Credit_Limit": chg, "Num_Credit_Inquiries": inq, "Credit_Mix": mix,
        "Outstanding_Debt": debt, "Credit_Utilization_Ratio": util,
        "Credit_History_Age": f"{hy} Years and {hm} Months",
        "Payment_of_Min_Amount": paymin, "Total_EMI_per_month": emi,
        "Amount_invested_monthly": invest, "Payment_Behaviour": beh,
        "Monthly_Balance": bal,
    }
    res = model.predict_one(record)
    pred = res["prediction"]
    color = {"Good": "🟢", "Poor": "🔴", "Standard": "🟡"}.get(pred, "")

    if pred == "Good":
        st.success(f"{color} **{pred}** — Keputusan: **{res['decision']}**")
    elif pred == "Poor":
        st.error(f"{color} **{pred}** — Keputusan: **{res['decision']}**")
    else:
        st.warning(f"{color} **{pred}** — Keputusan: **{res['decision']}**")

    st.markdown("**Probabilitas tiap kelas:**")
    prob_df = pd.DataFrame([res["probabilities"]]).T.rename(columns={0: "Probabilitas"})
    st.bar_chart(prob_df)
    cols = st.columns(len(res["probabilities"]))
    for col, (cls, pr) in zip(cols, res["probabilities"].items()):
        col.metric(cls, f"{pr:.1%}")
