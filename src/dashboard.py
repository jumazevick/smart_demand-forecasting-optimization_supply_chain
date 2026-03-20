import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl


st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]


def load_output() -> pd.DataFrame:
    output_path = BASE_DIR / "data" / "output.csv"
    if not output_path.exists():
        return pd.DataFrame()
    return pd.read_csv(output_path)


def load_store_data() -> pd.DataFrame:
    store_path = BASE_DIR / "data" / "historical_sales.csv"
    if not store_path.exists():
        return pd.DataFrame()
    return pd.read_csv(store_path, parse_dates=["date"])


def load_warehouse_options() -> pd.DataFrame:
    options_path = BASE_DIR / "data" / "warehouse_options.csv"
    if not options_path.exists():
        return pd.DataFrame()
    return pd.read_csv(options_path)


def main():
    st.title("Supply Chain AI Dashboard")
    st.caption(
        "Predict shortages, compare warehouse options, and check whether each store can afford the recommended order."
    )

    st.sidebar.title("Controls")
    if st.sidebar.button("Run Pipeline", type="primary"):
        with st.spinner("Running pipeline..."):
            pl.run_pipeline()
        st.sidebar.success("Pipeline complete.")
        st.rerun()

    output_df = load_output()
    store_df = load_store_data()
    options_df = load_warehouse_options()

    if output_df.empty or store_df.empty:
        st.info("No results yet. Click Run Pipeline in the sidebar.")
        st.stop()

    stores = sorted(output_df["store"].unique().tolist())
    products = sorted(output_df["product"].unique().tolist())

    selected_stores = st.sidebar.multiselect("Filter stores", stores, default=stores)
    selected_products = st.sidebar.multiselect("Filter products", products, default=products)

    filtered_output_df = output_df[
        output_df["store"].isin(selected_stores)
        & output_df["product"].isin(selected_products)
    ].copy()

    focus_store = st.sidebar.selectbox("Warehouse comparison store", selected_stores or stores, index=0)
    focus_product = st.sidebar.selectbox(
        "Warehouse comparison product",
        selected_products or products,
        index=0,
    )

    st.subheader("Key Numbers")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Restock Needs", int((filtered_output_df["shortage"] > 0).sum()))
    col2.metric("Affordable Orders", int(filtered_output_df["can_afford_order"].fillna(False).sum()))
    col3.metric("Total Recommended Qty", f"{filtered_output_df['ship_qty'].sum():,.0f}")
    col4.metric("Total Delivery Cost", f"${filtered_output_df['delivery_cost'].sum():,.0f}")

    st.divider()
    st.subheader("Decision Logic")
    st.markdown(
        """
Based on historical store data, the app predicts likely inventory shortages and then scores warehouse choices using five inputs:

- Cost: cheapest valid delivery option.
- Distance: shorter routes reduce transit time and fuel burn.
- Delivery reliability: warehouses with stronger fulfillment history score higher.
- Stock availability: warehouses must have enough inventory for the recommended order.
- Operational affordability: the store's projected cash must cover the order cost.

Goal: recommend the warehouse that balances cost, speed, and reliability while keeping the store operationally solvent.
"""
    )

    st.divider()
    st.subheader("Time Series: Stock vs Predicted Demand")
    focus_row = filtered_output_df[
        (filtered_output_df["store"] == focus_store)
        & (filtered_output_df["product"] == focus_product)
    ]

    if focus_row.empty:
        st.info("Select at least one matching store and product to view the time series.")
    else:
        predicted_demand = float(focus_row.iloc[0]["predicted_demand"])
        focus_history_df = store_df[
            (store_df["store"] == focus_store) & (store_df["product"] == focus_product)
        ].copy()
        focus_history_df["predicted_demand"] = predicted_demand
        focus_history_df["shortage_gap"] = (
            focus_history_df["predicted_demand"] - focus_history_df["stock"]
        ).clip(lower=0)

        left_chart, right_chart = st.columns(2)
        with left_chart:
            st.caption(f"{focus_store} • {focus_product}")
            stock_chart_df = focus_history_df.set_index("date")[["stock", "predicted_demand"]]
            st.line_chart(stock_chart_df)
        with right_chart:
            st.caption("Shortage periods highlighted")
            shortage_chart_df = focus_history_df.set_index("date")[["shortage_gap"]]
            st.area_chart(shortage_chart_df)

    st.divider()
    st.subheader("Warehouse Comparison Dashboard")
    if options_df.empty:
        st.info("Run the pipeline to generate warehouse option comparisons.")
    else:
        comparison_df = options_df[
            (options_df["store"] == focus_store) & (options_df["product"] == focus_product)
        ].copy()
        comparison_df = comparison_df.sort_values(["decision_score", "delivery_cost", "distance_km"])

        if comparison_df.empty:
            st.info("No warehouse options available for the current selection.")
        else:
            compare_left, compare_right = st.columns(2)
            with compare_left:
                st.caption("Delivery cost by warehouse")
                st.bar_chart(
                    comparison_df.set_index("warehouse")[["delivery_cost"]].sort_values(
                        by="delivery_cost", ascending=False
                    )
                )
            with compare_right:
                st.caption("Distance and stock availability")
                st.bar_chart(
                    comparison_df.set_index("warehouse")[["distance_km", "stock"]]
                )

            st.dataframe(
                comparison_df[
                    [
                        "warehouse",
                        "delivery_cost",
                        "distance_km",
                        "stock",
                        "reliability_score",
                        "recommended_order_qty",
                        "can_fulfill",
                        "can_afford_order",
                        "decision_score",
                    ]
                ],
                width="stretch",
            )

    st.divider()
    st.subheader("Profit Projection and Affordability")
    store_daily_df = (
        store_df[store_df["store"].isin(selected_stores)]
        .groupby(["store", "date"], as_index=False)[["profit", "operating_cost", "cash_available"]]
        .sum()
    )
    if store_daily_df.empty:
        st.info("No store finance data available.")
    else:
        focus_finance_df = store_daily_df[store_daily_df["store"] == focus_store].copy()
        threshold_value = 0.0
        if not focus_row.empty:
            threshold_value = float(focus_row.iloc[0]["delivery_cost"])
        focus_finance_df["order_cost_threshold"] = threshold_value
        finance_chart_df = focus_finance_df.set_index("date")[["cash_available", "order_cost_threshold"]]
        st.caption(f"Projected affordability for store {focus_store}")
        st.line_chart(finance_chart_df)
        st.dataframe(
            focus_finance_df.tail(15)[["date", "profit", "operating_cost", "cash_available"]],
            width="stretch",
        )

    st.divider()
    st.subheader("Recommended Orders")
    st.dataframe(filtered_output_df, width="stretch")

    shortages_df = filtered_output_df[filtered_output_df["shortage"] > 0].copy()
    st.divider()
    st.subheader("Shortage Details")
    if shortages_df.empty:
        st.success("No shortages detected for the current filters.")
    else:
        st.dataframe(
            shortages_df[
                [
                    "store",
                    "product",
                    "current_stock",
                    "predicted_demand",
                    "shortage",
                    "warehouse",
                    "delivery_cost",
                    "can_afford_order",
                    "status",
                ]
            ],
            width="stretch",
        )


if __name__ == "__main__":
    main()


