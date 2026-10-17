
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Medicine Stock-Out Prediction",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Medicine Stock-Out Prediction System")
st.caption("Inventory monitoring and next-week stock-out risk")

# Load files
df = pd.read_csv("dispensing.csv", parse_dates=["date"])
bundle = joblib.load("stockout_model.joblib")
model = bundle["model"]
features = bundle["features"]

# Latest record for each store-medicine combination
df = df.sort_values("date")
latest = df.groupby(
    ["store_id", "drug_id"], as_index=False
).tail(1).copy()

# Store and medicine selection
c1, c2 = st.columns(2)

with c1:
    store = st.selectbox(
        "Select store",
        sorted(df["store_id"].unique())
    )

with c2:
    medicines = sorted(
        df.loc[df["store_id"] == store, "drug_id"].unique()
    )
    drug = st.selectbox("Select medicine", medicines)

record = latest[
    (latest["store_id"] == store) &
    (latest["drug_id"] == drug)
]

if record.empty:
    st.warning("No inventory record found.")
    st.stop()

row = record.iloc[[0]]
probability = model.predict_proba(
    row[features]
)[0][list(model.classes_).index(1)]

# Display current inventory information
a, b, c = st.columns(3)
a.metric("Stock on hand", int(row["stock_on_hand"].iloc[0]))
b.metric("Recorded demand", int(row["demand"].iloc[0]))
c.metric("Predicted stock-out risk", f"{probability:.1%}")

if probability >= 0.5:
    st.error("High predicted risk — review inventory and resupply.")
elif probability >= 0.25:
    st.warning("Moderate predicted risk — monitor stock closely.")
else:
    st.success("Lower predicted risk — continue monitoring.")

st.caption(
    "Risk thresholds are initial demo settings, not validated "
    "operational thresholds. Predictions are based on the latest "
    "available record and require validation."
)

st.subheader("Latest Inventory Records")
st.dataframe(
    latest[[
        "date", "store_id", "drug_id", "stock_on_hand",
        "demand", "units_ordered", "stockout"
    ]].sort_values("date", ascending=False),
    use_container_width=True
)

st.subheader("Inventory Overview")
st.bar_chart(
    latest.groupby("drug_id")["stock_on_hand"].sum()
)

# Display the supplied PNGs
st.subheader("Project Analysis Visualizations")

with st.expander("Model benchmark comparison"):
    st.image(
        "benchmark_model_comparison.png",
        caption="Supplied benchmark graphic — results not independently verified"
    )

with st.expander("Store population and demand"):
    st.image(
        "demographic_demand_coupling.png",
        caption="Supplied population-demand visualization"
    )

with st.expander("Disease severity and medicine demand"):
    st.image(
        "outbreak_demand_alignment.png",
        caption="Supplied demand-alignment visualization"
    )
