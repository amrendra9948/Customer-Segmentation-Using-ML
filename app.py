
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ---------------------------------
# 1. Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ---------------------------------
# 2. Load Model and Dataset
# ---------------------------------

MODEL_PATH = "notebook/models/customer_segmentation_model.pkl"
DATA_PATH = r"C:\Users\Amrit Singh\Desktop\Customer_Segmentation_ML\data\Mall_Customers.csv"
OUTPUT_PATH = "notebook/outputs/clustered_customers.csv"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Run the Jupyter Notebook first.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found! Add Mall_Customers.csv to data folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

if os.path.exists(OUTPUT_PATH):
    df = pd.read_csv(OUTPUT_PATH)
else:
    df = pd.read_csv(DATA_PATH)

# Generate clusters if output file is unavailable
if "Cluster" not in df.columns:
    features = df[
        ["Annual Income (k$)", "Spending Score (1-100)"]
    ]
    df["Cluster"] = model.predict(features)

# ---------------------------------
# 3. Main Heading
# ---------------------------------

st.title("🛍️ Customer Segmentation System")

st.write(
    "An interactive Machine Learning application "
    "using K-Means Clustering to group customers "
    "based on annual income and spending behaviour."
)

st.divider()

# ---------------------------------
# 4. Dashboard Metrics
# ---------------------------------

total_customers = len(df)

total_clusters = df["Cluster"].nunique()

silhouette = st.session_state.get("silhouette", None)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    total_customers
)

col2.metric(
    "Customer Segments",
    total_clusters
)

# Recalculate silhouette score using saved model
from sklearn.metrics import silhouette_score

X = df[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]

X_scaled = model.named_steps["scaler"].transform(X)

score = silhouette_score(
    X_scaled,
    model.named_steps["kmeans"].labels_
)

col3.metric(
    "Silhouette Score",
    round(score, 3)
)

st.divider()

# ---------------------------------
# 5. Customer Cluster Visualization
# ---------------------------------

st.subheader("Customer Segmentation Graph")

fig, ax = plt.subplots(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=90,
    ax=ax
)

ax.set_title("K-Means Customer Clusters")

ax.set_xlabel("Annual Income (k$)")

ax.set_ylabel("Spending Score")

st.pyplot(fig)

plt.close(fig)

st.divider()

# ---------------------------------
# 6. Cluster Summary
# ---------------------------------

st.subheader("Customer Segment Analysis")

summary = df.groupby("Cluster").agg({
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean",
    "CustomerID": "count"
}).rename(columns={
    "Annual Income (k$)": "Average Income",
    "Spending Score (1-100)": "Average Spending",
    "CustomerID": "Customer Count"
}).round(2)

st.dataframe(summary, use_container_width=True)

# ---------------------------------
# 7. New Customer Prediction
# ---------------------------------

st.divider()

st.subheader("Predict Customer Segment")

st.write(
    "Enter customer details to identify their "
    "predicted customer group."
)

income = st.slider(
    "Annual Income (in thousand dollars)",
    min_value=15,
    max_value=140,
    value=60
)

spending = st.slider(
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Customer Segment"):

    new_customer = pd.DataFrame(
        [[income, spending]],
        columns=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    prediction = int(model.predict(new_customer)[0])

    st.success(
        f"Predicted Customer Cluster: {prediction}"
    )

    median_income = df["Annual Income (k$)"].median()

    median_spending = df[
        "Spending Score (1-100)"
    ].median()

    if income >= median_income and spending >= median_spending:
        segment = "High Income - High Spending"

    elif income >= median_income and spending < median_spending:
        segment = "High Income - Low Spending"

    elif income < median_income and spending >= median_spending:
        segment = "Low Income - High Spending"

    else:
        segment = "Low Income - Low Spending"

    st.info(
        f"Behavioural description: {segment}"
    )

    st.write(
        "This description is based on dataset medians. "
        "The actual cluster assignment comes from K-Means."
    )

st.divider()

# ---------------------------------
# 8. Show Customer Records
# ---------------------------------

st.subheader("Customer Records")

st.dataframe(
    df,
    use_container_width=True
)

# ---------------------------------
# 9. Download Clustered Dataset
# ---------------------------------

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Segmentation Results",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)

st.success("Customer Segmentation Dashboard Loaded!")
