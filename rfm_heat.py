import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# --- Streamlit Page Setup ---
st.set_page_config(page_title="RFM Segment Dashboard", layout="centered")

# --- With Compliments Header ---
st.markdown("""
<div style='text-align: center; font-size: 18px; font-weight: bold; color: green;'>
With compliments<br>
📧 promotions@realanalytics101.co.za &nbsp;&nbsp;&nbsp;&nbsp; 🌐 www.realanalytics101.co.za
</div>
""", unsafe_allow_html=True)

# --- Title and Introduction ---
st.title("🧠 RFM Segment Dashboard")

st.markdown("""
### What is RFM?

RFM stands for:

- <span style='font-weight:bold;'>📅 Recency</span> 
  <span style="cursor: help; color: grey;" title="Recency = Days since last purchase. Lower is better.">ℹ️</span>  
  — Measures how **recently** a customer made a purchase.

- <span style='font-weight:bold;'>🔁 Frequency</span> 
  <span style="cursor: help; color: grey;" title="Frequency = Number of purchases. Higher is better.">ℹ️</span>  
  — Measures how **often** a customer makes purchases.

- <span style='font-weight:bold;'>💰 Monetary</span> 
  <span style="cursor: help; color: grey;" title="Monetary = Total amount spent. Higher is better.">ℹ️</span>  
  — Measures how much money a customer has spent.

Customers are scored from **1 (low)** to **5 (high)** using percentiles for each metric.

This helps you identify:
- 🎯 **Champions** — high-value, active customers
- 🤝 **Loyal Customers** — regular buyers
- ⚠️ **At Risk** — haven’t purchased recently
- 🆕 **New Customers** — recent first-time buyers
""", unsafe_allow_html=True)

# --- Dummy Data Generator ---
@st.cache_data
def generate_dummy_data(n_transactions=1000, n_customers=100):
    np.random.seed(42)
    customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n_customers + 1)]
    invoice_nos = [f"INV{str(i).zfill(6)}" for i in range(1, n_transactions + 1)]

    data = {
        'CustomerID': np.random.choice(customer_ids, n_transactions),
        'InvoiceNo': invoice_nos,
        'InvoiceDate': [datetime(2025, 6, 15) - timedelta(days=np.random.randint(1, 365)) for _ in range(n_transactions)],
        'Amount': np.round(np.random.exponential(scale=500, size=n_transactions), 2)
    }
    return pd.DataFrame(data)

# --- RFM Computation ---
@st.cache_data
def compute_rfm(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    NOW = df['InvoiceDate'].max() + timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (NOW - x.max()).days,
        'InvoiceNo': 'nunique',
        'Amount': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    def segment(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 4:
            return 'Recent Customers'
        elif row['F_Score'] >= 4:
            return 'Loyal Customers'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
            return 'At Risk'
        else:
            return 'Others'

    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm

# --- Excel Export Helper ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='RFM')
    return output.getvalue()

# --- Run Analysis Button ---
if st.button("🚀 Generate & Analyze RFM"):
    df = generate_dummy_data()
    rfm = compute_rfm(df)

    st.subheader("📊 Sample Transactions")
    st.dataframe(df.head())

    st.subheader("📈 RFM Table with Segment Labels")
    st.dataframe(rfm.head())

    # --- Heatmap ---
    st.subheader("🗺️ Recency vs Frequency Heatmap")
    heatmap_data = rfm.groupby(['R_Score', 'F_Score']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.sort_index(ascending=False).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Customer Count'}, ax=ax)
    ax.set_xlabel("Frequency Score (1 = low, 5 = high)")
    ax.set_ylabel("Recency Score (1 = old, 5 = recent)")
    ax.set_title("Customer Count by Recency & Frequency")
    st.pyplot(fig)

    # --- Segment Filter + Export ---
    st.subheader("🎯 Select Segment to Export")
    segments = sorted(rfm['Segment'].unique())
    selected = st.selectbox("Choose a customer segment:", segments)

    filtered = rfm[rfm['Segment'] == selected]
    st.markdown(f"**{len(filtered)} customers** found in segment: **{selected}**")
    st.dataframe(filtered)

    st.download_button(
        label="📥 Download Selected Segment as Excel",
        data=to_excel(filtered),
        file_name=f"{selected.replace(' ', '_').lower()}_rfm.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Click the button above to run the RFM analysis.")
