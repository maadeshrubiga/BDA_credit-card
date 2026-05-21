%%writefile app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, mean

# ---------------- SPARK SESSION ----------------
spark = SparkSession.builder \
    .appName("CreditCardFraudBigData") \
    .getOrCreate()


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

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #111111;
}

/* Buttons */
.stButton>button {
    background-color: #00FFFF;
    color: black;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

/* Charts */
.css-1r6slb0 {
    background-color: #111111;
    border-radius: 12px;
    padding: 10px;
}

/* Success Message */
.stSuccess {
    background-color: #003333;
    color: #00FFFF;
}

/* Markdown Text */
p, label {
    color: white;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)
st.title("💳 Credit Card Fraud Detection Using Big Data Analytics (Spark)")

uploaded_file = st.file_uploader("Upload Credit Card Fraud Dataset (CSV)", type="csv")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ---------------- LOAD USING SPARK ----------------
    sdf = spark.read.csv("temp.csv", header=True, inferSchema=True)

    # Convert limited rows to Pandas for visualization
    df = sdf.limit(50000).toPandas()

    # ---------------- DATASET OVERVIEW ----------------
    st.header("1️⃣ Dataset Overview")
    st.write("Shape (approx):", (sdf.count(), len(sdf.columns)))
    st.write("Columns:", sdf.columns)
    st.dataframe(df.head())

    # ---------------- DATA TYPES ----------------
    st.header("2️⃣ Column Data Types")
    dtype_df = pd.DataFrame(sdf.dtypes, columns=["Column", "Data Type"])
    st.dataframe(dtype_df)

    # ---------------- DATA QUALITY ----------------
    st.header("3️⃣ Data Quality Analysis")

    missing = sdf.select([
        count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in sdf.columns
    ]).toPandas().T

    duplicates = sdf.count() - sdf.dropDuplicates().count()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values")
        st.dataframe(missing)

    with col2:
        st.subheader("Duplicate Rows")
        st.write("Duplicates:", duplicates)

    sdf = sdf.dropDuplicates()

    # ---------------- CLASS IMBALANCE ----------------
    st.header("4️⃣ Fraud vs Legitimate Distribution")

    class_counts_spark = sdf.groupBy("Class").count().orderBy("Class")
    class_counts = class_counts_spark.toPandas().set_index("Class")["count"]

    st.dataframe(class_counts)

    fig1 = plt.figure()
    class_counts.plot(kind="bar")
    plt.xlabel("Class (0 = Legit, 1 = Fraud)")
    plt.ylabel("Count")
    plt.title("Fraud vs Legitimate Transactions")
    st.pyplot(fig1)

    # ---------------- DESCRIPTIVE STATS ----------------
    st.header("5️⃣ Descriptive Statistics")
    st.dataframe(sdf.describe().toPandas())

    # ---------------- AMOUNT ANALYSIS ----------------
    st.header("6️⃣ Transaction Amount Analysis")

    avg_amt_spark = sdf.groupBy("Class").agg(mean("Amount").alias("AvgAmount"))
    avg_amt = avg_amt_spark.toPandas().set_index("Class")["AvgAmount"]

    col3, col4 = st.columns(2)

    with col3:
        fig2 = plt.figure()
        avg_amt.plot(kind="bar")
        plt.title("Average Transaction Amount by Class")
        st.pyplot(fig2)

    with col4:
        fig3 = plt.figure()
        df["Amount"].hist(bins=50)
        plt.title("Overall Amount Distribution")
        st.pyplot(fig3)

    # ---------------- FRAUD AMOUNT ----------------
    st.header("7️⃣ Fraud Amount Distribution")

    fig4 = plt.figure()
    df[df["Class"] == 1]["Amount"].hist(bins=30)
    plt.title("Fraud Transaction Amount Distribution")
    st.pyplot(fig4)

    # ---------------- HIGH VALUE FRAUD ----------------
    st.header("8️⃣ High Value Fraud Transactions")

    top_fraud = sdf.filter(col("Class") == 1).orderBy(col("Amount").desc()).limit(10)
    st.dataframe(top_fraud.toPandas())

    # ---------------- CORRELATION ----------------
    st.header("9️⃣ Correlation Analysis (Sample)")

    numeric_cols = df.select_dtypes(include="number").columns[:10]
    corr = df[numeric_cols].corr()

    fig5 = plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    st.pyplot(fig5)

    # ---------------- METRICS ----------------
    st.header("🔟 Key Metrics")

    total = sdf.count()
    fraud = class_counts.get(1, 0)
    fraud_percentage = (fraud / total) * 100

    st.metric("Fraud Percentage", f"{fraud_percentage:.4f}%")
    st.metric("Total Transactions", total)
    st.metric("Fraud Transactions", fraud)

    # ---------------- INSIGHTS ----------------
    st.header("💡 Insights & Observations")
    st.markdown("""
    - Spark processes the full dataset using distributed computing
    - Only samples are converted to Pandas for visualization
    - Dataset is highly imbalanced
    - High-value fraud transactions pose major financial risks
    - Spark enables scalable fraud analytics for big data
    """)

    st.success("✅ Spark-Based Credit Card Fraud Analysis Completed Successfully")
