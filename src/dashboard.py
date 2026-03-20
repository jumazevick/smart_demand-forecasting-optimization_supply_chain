import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow importing pipeline from the same src/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]


def load_output() -> pd.DataFrame:
    output_path = BASE_DIR / "data" / "output.csv"
    if not output_path.exists():
        return pd.DataFrame()
    return pd.read_csv(output_path)


def load_sales() -> pd.DataFrame:
    sales_path = BASE_DIR / "data" / "historical_sales.csv"
    if not sales_path.exists():
        return pd.DataFrame()
    return pd.read_csv(sales_path, parse_dates=["date"])


def main():
    st.title("Supply Chain AI Dashboard")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("Controls")

    if st.sidebar.button("Run Pipeline", type="primary"):
        with st.spinner("Running pipeline…"):
            pl.run_pipeline()
        st.sidebar.success("Pipeline complete!")
        st.rerun()

    df = load_output()

    if df.empty:
        st.info("No results yet. Click **Run Pipeline** in the sidebar to start.")
        st.stop()

    products = sorted(df["product"].unique().tolist())
    stores = sorted(df["store"].unique().tolist())

    selected_products = st.sidebar.multiselect("Filter by product", products, default=products)
    selected_stores = st.sidebar.multiselect("Filter by store", stores, default=stores)

    mask = df["product"].isin(selected_products) & df["store"].isin(selected_stores)
    df = df[mask].copy()

    # ── KPI cards ────────────────────────────────────────────────────────────
    st.subheader("Key Numbers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stores", df["store"].nunique())
    c2.metric("Products", df["product"].nunique())
    c3.metric("Shortages", int((df["shortage"] > 0).sum()))
    c4.metric("Total Transport Cost", f"${df['transport_cost'].sum():,.0f}")

    st.divider()

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predicted Demand by Store")
        by_store = (
            df.groupby("store")["predicted_demand"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(by_store)

    with col_right:
        st.subheader("Predicted Demand by Product")
        by_product = (
            df.groupby("product")["predicted_demand"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(by_product)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Stock vs Predicted Demand (by Product)")
        stock_vs_demand = df.groupby("product")[["available_stock", "predicted_demand"]].first()
        st.bar_chart(stock_vs_demand)

    with col_right2:
        st.subheader("Transport Cost by Warehouse")
        by_wh = (
            df.groupby("warehouse")["transport_cost"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(by_wh)

    # ── Historical sales trend ────────────────────────────────────────────────
    st.divider()
    st.subheader("Historical Sales Trend")

    sales_df = load_sales()
    if not sales_df.empty:
        sales_filtered = sales_df[
            sales_df["product"].isin(selected_products)
            & sales_df["store"].isin(selected_stores)
        ]
        daily = (
            sales_filtered.groupby("date")["sales"]
            .sum()
            .reset_index()
            .set_index("date")
        )
        st.line_chart(daily)
    else:
        st.info("historical_sales.csv not found.")

    # ── Full results table ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Full Forecast & Assignment Table")
    st.dataframe(df, width="stretch")

    # ── Shortages ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Shortage Details")
    shortages = df[df["shortage"] > 0]
    if shortages.empty:
        st.success("No shortages detected. All stores can be fulfilled.")
    else:
        st.warning(f"{len(shortages)} store-product combinations have a shortage.")
        st.dataframe(shortages[["store", "product", "predicted_demand", "available_stock", "shortage"]], width="stretch")

    # ── How it works ─────────────────────────────────────────────────────────
    st.divider()
    with st.expander("How does the pipeline work?"):
        st.markdown(
            """
**Step 1 — Predict**
- Look at 2 years of daily sales for every store + product.
- Calculate the average daily sales, then multiply by 1.1 (10% growth cushion).
- Result: *predicted_demand* per store-product.

**Step 2 — Plan**
- Compare predicted demand against the maximum stock held by any warehouse for that product.
- If demand > stock → flag as *shortage*.

**Step 3 — Deliver**
- For each store-product, pick the closest warehouse (by km) that stocks the product.
- Calculate how much to ship (*ship_qty*) and the transport cost (qty × km).
"""
        )


if __name__ == "__main__":
    main()


