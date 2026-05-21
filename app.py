import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# ---------- CUSTOM BLACK THEME ----------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #000000;
    color: white;
}

/* Title */
h1, h2, h3 {
    color: #00FFFF;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background-color: #1a1a1a;
    border: 2px solid #00FFFF;
    border-radius: 12px;
    padding: 15px;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: #111111;
    border: 1px solid #00FFFF;
    padding: 15px;
    border-radius: 12px;
}

/* Buttons */
.stButton>button {
    background-color: #00FFFF;
    color: black;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

/* Text */
p, label {
    color: white;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("💳 Credit Card Fraud Detection Using Big Data Analytics")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Credit Card Fraud Dataset (CSV)",
    type=["csv"]
)

# ---------------- PROCESS DATA ----------------
if uploaded_file is not None:

    with st.spinner("Processing Dataset..."):

        # Load CSV using Pandas
        df = pd.read_csv(uploaded_file)

        # Limit rows for performance
        df = df.head(10000)

        # ---------------- DATASET OVERVIEW ----------------
        st.header("1️⃣ Dataset Overview")

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        st.dataframe(df.head())

        # ---------------- DATA TYPES ----------------
        st.header("2️⃣ Column Data Types")

        dtype_df = pd.DataFrame(
            df.dtypes,
            columns=["Data Type"]
        )

        st.dataframe(dtype_df)

        # ---------------- MISSING VALUES ----------------
        st.header("3️⃣ Data Quality Analysis")

        missing = df.isnull().sum()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Missing Values")
            st.dataframe(missing)

        with col2:
            st.subheader("Duplicate Rows")
            st.write(df.duplicated().sum())

        # ---------------- FRAUD DISTRIBUTION ----------------
        st.header("4️⃣ Fraud vs Legitimate Distribution")

        class_counts = df["Class"].value_counts()

        st.dataframe(class_counts)

        fig1, ax1 = plt.subplots()

        class_counts.plot(
            kind="bar",
            ax=ax1
        )

        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        ax1.set_title("Fraud vs Legitimate Transactions")

        st.pyplot(fig1)

        # ---------------- DESCRIPTIVE STATS ----------------
        st.header("5️⃣ Descriptive Statistics")

        st.dataframe(df.describe())

        # ---------------- AMOUNT ANALYSIS ----------------
        st.header("6️⃣ Transaction Amount Analysis")

        avg_amt = df.groupby("Class")["Amount"].mean()

        col3, col4 = st.columns(2)

        with col3:

            fig2, ax2 = plt.subplots()

            avg_amt.plot(
                kind="bar",
                ax=ax2
            )

            ax2.set_title(
                "Average Transaction Amount by Class"
            )

            st.pyplot(fig2)

        with col4:

            fig3, ax3 = plt.subplots()

            df["Amount"].hist(
                bins=50,
                ax=ax3
            )

            ax3.set_title(
                "Overall Amount Distribution"
            )

            st.pyplot(fig3)

        # ---------------- FRAUD AMOUNT ----------------
        st.header("7️⃣ Fraud Amount Distribution")

        fig4, ax4 = plt.subplots()

        df[df["Class"] == 1]["Amount"].hist(
            bins=30,
            ax=ax4
        )

        ax4.set_title(
            "Fraud Transaction Amount Distribution"
        )

        st.pyplot(fig4)

        # ---------------- TOP FRAUDS ----------------
        st.header("8️⃣ High Value Fraud Transactions")

        top_fraud = df[df["Class"] == 1] \
            .sort_values(by="Amount", ascending=False) \
            .head(10)

        st.dataframe(top_fraud)

        # ---------------- CORRELATION ----------------
        st.header("9️⃣ Correlation Heatmap")

        numeric_cols = df.select_dtypes(include=np.number)

        corr = numeric_cols.corr()

        fig5, ax5 = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            ax=ax5
        )

        ax5.set_title("Correlation Heatmap")

        st.pyplot(fig5)

        # ---------------- METRICS ----------------
        st.header("🔟 Key Metrics")

        total = len(df)

        fraud = len(df[df["Class"] == 1])

        fraud_percentage = (fraud / total) * 100

        col5, col6, col7 = st.columns(3)

        with col5:
            st.metric(
                "Total Transactions",
                total
            )

        with col6:
            st.metric(
                "Fraud Transactions",
                fraud
            )

        with col7:
            st.metric(
                "Fraud Percentage",
                f"{fraud_percentage:.4f}%"
            )

        # ---------------- INSIGHTS ----------------
        st.header("💡 Insights & Observations")

        st.markdown("""
        - The dataset is highly imbalanced
        - Fraudulent transactions are very low compared to legitimate transactions
        - High-value fraud transactions create major financial risks
        - Big Data Analytics improves fraud monitoring efficiency
        - Interactive dashboards help in real-time analysis
        """)

        st.success(
            "✅ Credit Card Fraud Analysis Completed Successfully"
        )
