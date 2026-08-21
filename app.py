import pandas as pd
import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")
st.title("📦 Milestone 4: Smart Inventory Dashboard & Reporting")


if not os.path.exists("data/forecast_results.csv"):
    st.error("⚠ Run forecasting first! 'data/forecast_results.csv' not found.")
    st.stop()

df = pd.read_csv("data/forecast_results.csv")
st.write("📊 Forecast Summary:", df["forecast_best"].describe())
st.write("📝 Columns in Dataset:", df.columns)


if (df["forecast_best"] < 0).any():
    st.warning("⚠ Negative forecast values detected. They will be clipped to zero.")
    df["forecast_best"] = df["forecast_best"].clip(lower=0)

df["date"] = pd.to_datetime(df["date"])


lead = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
oc = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
hc = st.sidebar.slider("Holding Cost ($/unit)", 1, 20, 2)

service_level_options = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = service_level_options[st.sidebar.selectbox("Service Level", ["90%", "95%", "99%"], 1)]


tab1, tab2, tab3, tab4 = st.tabs(["Forecasts", "Inventory", "Stock Alerts", "Reports"])


with tab1:
    st.subheader("📈 Forecast vs Actuals")

    product = st.selectbox("Select Product", sorted(df["Product Name"].unique()))
    forecast = df[df["Product Name"] == product].copy()
    forecast.rename(columns={"date": "ds", "forecast_best": "yhat"}, inplace=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(forecast["ds"], forecast["yhat"], color="tab:blue", linewidth=2, label="Forecast")

    ax.set_title(f"Forecast Trend for {product}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Forecast Value", fontsize=12)
    ax.legend()


    if len(forecast) > 180:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    elif len(forecast) > 60:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    plan = []
    for p in df["Product Name"].unique():
        d = df[df["Product Name"] == p]
        avg = d["forecast_best"].mean() / 30
        dem = d["forecast_best"].sum()
        std = d["forecast_best"].std()

        if pd.isna(std) or std < 0:
            std = 0.0

        if dem <= 0 or hc <= 0 or oc <= 0:
            eoq = 0.0
        else:
            val_eoq = (2 * dem * oc) / hc
            val_eoq = np.clip(val_eoq, 0, None)
            eoq = np.sqrt(val_eoq)

        ss = z * std * np.sqrt(lead) if lead > 0 else 0
        rop = (avg * lead) + ss

        plan.append({
            "Product": p,
            "AvgDailySales": round(avg,2),
            "TotalDemand": round(dem,2),
            "EOQ": round(eoq,2),
            "SafetyStock": round(ss,2),
            "ReorderPoint": round(rop,2)
        })

    inv = pd.DataFrame(plan)
    st.subheader("🗂️ Inventory Plan")
    st.dataframe(inv)


with tab3:
    inv["CurrentStock"] = np.random.randint(10, 100, len(inv))
    inv["Action"] = np.where(inv["CurrentStock"] < inv["ReorderPoint"], "Reorder 🚨", "OK ✅")
    st.subheader("⚠ Stock Alerts")
    st.dataframe(inv[["Product", "CurrentStock", "ReorderPoint", "Action"]])
    st.bar_chart(inv.set_index("Product")[["CurrentStock", "ReorderPoint"]])

with tab4:
    st.download_button("📥 Download Inventory Report", inv.to_csv(index=False), "inventory_plan.csv")

upl = st.sidebar.file_uploader("Upload New Sales Data", type="csv")
if upl:
    new = pd.read_csv(upl)
    st.sidebar.success("File uploaded ✅")
    st.sidebar.info("Re-run forecasting.py manually to refresh predictions.")
