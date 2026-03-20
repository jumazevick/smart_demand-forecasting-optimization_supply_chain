import sys
from pathlib import Path

import altair as alt
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


def build_daily_manager_table(
    output_df: pd.DataFrame,
    store_df: pd.DataFrame,
    selected_stores: list[str],
    selected_products: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build a daily store-product planning table for managers."""
    recommendation_cols = [
        "store",
        "product",
        "predicted_demand",
        "warehouse",
        "distance_km",
        "min_order_qty",
        "delivery_cost",
        "ship_qty",
        "projected_cash_available",
    ]
    rec_df = output_df[recommendation_cols].copy()

    base_df = store_df[
        store_df["store"].isin(selected_stores)
        & store_df["product"].isin(selected_products)
        & (store_df["date"] >= start_date)
        & (store_df["date"] <= end_date)
    ][["date", "store", "product", "stock"]].copy()

    daily_cash_df = (
        store_df[store_df["store"].isin(selected_stores)]
        .groupby(["store", "date"], as_index=False)["cash_available"]
        .sum()
        .rename(columns={"cash_available": "projected_cash_available_daily"})
    )

    table_df = (
        base_df.merge(rec_df, on=["store", "product"], how="left")
        .merge(daily_cash_df, on=["store", "date"], how="left")
    )

    table_df["current_stock"] = table_df["stock"].fillna(0)
    table_df["shortage"] = (table_df["predicted_demand"] - table_df["current_stock"]).clip(lower=0).round(2)
    table_df["recommended_order_qty"] = table_df["shortage"]
    active_mask = table_df["recommended_order_qty"] > 0
    table_df.loc[active_mask, "recommended_order_qty"] = table_df.loc[
        active_mask, ["recommended_order_qty", "min_order_qty"]
    ].max(axis=1)

    unit_delivery_cost = (table_df["delivery_cost"] / table_df["ship_qty"].replace(0, pd.NA)).fillna(0)
    table_df["delivery_cost_daily"] = (table_df["recommended_order_qty"] * unit_delivery_cost).round(2)

    # Ordering is recommended only if there is shortage and enough cash on that day.
    table_df["recommended_period"] = (
        (table_df["shortage"] > 0)
        & (table_df["projected_cash_available_daily"] >= table_df["delivery_cost_daily"])
    )
    table_df["planned_order_date"] = table_df["date"].where(table_df["recommended_period"])

    lead_days = (table_df["distance_km"] / 120).round().clip(lower=1).fillna(1).astype(int)
    table_df["expected_delivery_date"] = table_df["planned_order_date"] + pd.to_timedelta(lead_days, unit="D")

    return table_df


def render_timeline_chart(timeline_df: pd.DataFrame, show_recommended: bool):
    """Show stock, predicted demand, and cash flow lines, with optional recommendation highlights."""
    plot_df = timeline_df.sort_values("date").copy()
    plot_df["next_date"] = plot_df["date"] + pd.Timedelta(days=1)

    base = alt.Chart(plot_df)
    stock_line = base.mark_line(color="#16a34a", strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("current_stock:Q", title="Units"),
    )
    demand_line = base.mark_line(color="#dc2626", strokeDash=[6, 4], strokeWidth=2).encode(
        x="date:T",
        y="predicted_demand:Q",
    )
    cash_line = base.mark_line(color="#2563eb", strokeWidth=2).encode(
        x="date:T",
        y=alt.Y("projected_cash_available_daily:Q", title="Cash"),
    )

    layers = [stock_line, demand_line, cash_line]
    if show_recommended:
        highlights = plot_df[plot_df["recommended_period"]].copy()
        if not highlights.empty:
            bands = alt.Chart(highlights).mark_rect(color="#f59e0b", opacity=0.18).encode(
                x="date:T",
                x2="next_date:T",
            )
            layers.insert(0, bands)

    st.altair_chart(alt.layer(*layers).resolve_scale(y="independent").properties(height=340), use_container_width=True)


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
    min_date = store_df["date"].min().date()
    max_date = store_df["date"].max().date()

    selected_stores = st.sidebar.multiselect("Filter stores", stores, default=stores)
    selected_products = st.sidebar.multiselect("Filter products", products, default=products)
    date_range = st.sidebar.date_input(
        "Filter date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) != 2:
        st.warning("Please select start and end dates.")
        st.stop()
    selected_start_date = pd.to_datetime(date_range[0])
    selected_end_date = pd.to_datetime(date_range[1])

    filtered_output_df = output_df[
        output_df["store"].isin(selected_stores)
        & output_df["product"].isin(selected_products)
    ].copy()

    daily_table_df = build_daily_manager_table(
        output_df,
        store_df,
        selected_stores,
        selected_products,
        selected_start_date,
        selected_end_date,
    )

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
    st.subheader("Timeline: Stock, Predicted Demand, and Projected Cash")
    show_recommended_periods = st.button("Show Recommended Order Periods")

    focus_timeline_df = daily_table_df[
        (daily_table_df["store"] == focus_store) & (daily_table_df["product"] == focus_product)
    ].copy()
    if focus_timeline_df.empty:
        st.info("No timeline data for current filters.")
    else:
        render_timeline_chart(focus_timeline_df, show_recommended_periods)
        if show_recommended_periods:
            recommended_days = int(focus_timeline_df["recommended_period"].sum())
            st.caption(f"Recommended ordering periods found on {recommended_days} day(s).")

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

            with st.expander("📊 Warehouse Comparison Table - Column Reference"):
                st.markdown("""
- **warehouse**: Warehouse code (W1–W5)
- **delivery_cost**: Total cost to deliver recommended order quantity
- **distance_km**: Distance from warehouse to store in kilometers
- **stock**: Units available in that warehouse for this product
- **reliability_score**: Warehouse's historical fulfillment reliability (0–1, higher is better)
- **recommended_order_qty**: Quantity needed to fill the shortage
- **can_fulfill**: ✓ if warehouse has enough stock; ✗ otherwise
- **can_afford_order**: ✓ if store's cash covers cost; ✗ otherwise
- **decision_score**: Combined ranking (cost, distance, reliability weighted). Lower score = better choice.
                """)

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
        focus_row = filtered_output_df[
            (filtered_output_df["store"] == focus_store)
            & (filtered_output_df["product"] == focus_product)
        ]
        threshold_value = 0.0
        if not focus_row.empty:
            threshold_value = float(focus_row.iloc[0]["delivery_cost"])
        focus_finance_df["order_cost_threshold"] = threshold_value
        finance_chart_df = focus_finance_df.set_index("date")[["cash_available", "order_cost_threshold"]]
        st.caption(f"Projected affordability for store {focus_store}")
        st.line_chart(finance_chart_df)
        
        with st.expander("💰 Finance Metrics Explained"):
            st.markdown("""
- **date**: Daily snapshot date
- **profit**: Daily net profit (revenue minus operating costs and COGS)
- **operating_cost**: Daily fixed costs (rent, utilities, staff, etc.)
- **cash_available**: Projected liquid cash reserves on that day (used to assess order affordability)
- **order_cost_threshold**: The delivery cost for the recommended order (amber line on chart); if cash falls below this, order becomes unaffordable
            """)
        
        st.dataframe(
            focus_finance_df.tail(15)[["date", "profit", "operating_cost", "cash_available"]],
            width="stretch",
        )

    st.divider()
    st.subheader("Recommended Orders")
    st.caption(
        "Daily planning table for each store-product pair: stock, forecast, shortage, suggested order, selected warehouse, cost, cash, and delivery dates."
    )

    # Column legend for manager table
    with st.expander("📋 Column Descriptions for Recommended Orders Table", expanded=True):
        col_legend_cols = st.columns(2)
        with col_legend_cols[0]:
            st.markdown("""
**Identity Columns:**
- **store**: Store location code (A, B, C, etc.)
- **product**: Product SKU or name
- **date**: Date of the snapshot (daily record)

**Stock & Demand:**
- **current_stock**: Units currently in inventory at the store
- **predicted_demand**: Forecasted units needed for the next cycle (based on historical average × 1.1 growth)
- **shortage**: Shortfall amount = max(0, predicted_demand - current_stock). Zero if stock exceeds forecast.
            """)
        
        with col_legend_cols[1]:
            st.markdown("""
**Order Planning:**
- **recommended_order_qty**: Quantity recommended to order (shortfall or minimum order qty, whichever is larger)
- **warehouse**: Selected warehouse code (W1, W2, W3, W4, W5) based on cost, distance, reliability, and stock
- **delivery_cost**: Estimated total delivery cost for the recommended order (in dollars)

**Financial Viability:**
- **projected_cash_available**: Store's projected cash reserves on that date (sum across all products)
- **planned_order_date**: Date to place the order (only if shortage exists AND store has enough cash)
- **expected_delivery_date**: Estimated arrival date (calculated from distance: ~1 day per 120 km)
            """)

    manager_table_cols = [
        "store",
        "product",
        "date",
        "current_stock",
        "predicted_demand",
        "shortage",
        "recommended_order_qty",
        "warehouse",
        "delivery_cost_daily",
        "projected_cash_available_daily",
        "planned_order_date",
        "expected_delivery_date",
    ]
    manager_table = daily_table_df[manager_table_cols].rename(
        columns={
            "delivery_cost_daily": "delivery_cost",
            "projected_cash_available_daily": "projected_cash_available",
        }
    )
    st.dataframe(manager_table.sort_values(["store", "product", "date"]), width="stretch")

    with st.expander("🏭 Additional Warehouse Selection Criteria"):
        st.markdown(
            """
**How warehouses are scored and ranked:**
- **Cost**: Delivery cost for the recommended quantity (40% weight in decision)
- **Distance**: Geographic distance in kilometers; drives transit time and fuel (25% weight)
- **Reliability**: Historical fulfillment success rate (20% weight)
- **Stock Availability**: Warehouse must have sufficient inventory for the order
- **Affordability**: Store's projected cash must cover the delivery cost

The algorithm ranks all 5 warehouses on each factor and selects the best-scoring option that can fulfill and is affordable.
"""
        )

    shortages_df = filtered_output_df[filtered_output_df["shortage"] > 0].copy()
    st.divider()
    st.subheader("Shortage Details")
    if shortages_df.empty:
        st.success("No shortages detected for the current filters.")
    else:
        with st.expander("⚠️ Shortage Details Table - Column Reference"):
            st.markdown("""
- **store**: Store location code
- **product**: Product SKU or name
- **current_stock**: Units in inventory now
- **predicted_demand**: Forecasted units needed
- **shortage**: Units missing (predicted_demand - current_stock)
- **warehouse**: Best-selected warehouse to fulfill this shortage
- **delivery_cost**: Cost to deliver from selected warehouse
- **can_afford_order**: ✓ if store has sufficient cash; ✗ if order exceeds available funds (affordability alert)
- **status**: Order status or note (e.g., "recommended", "affordable", "financially constrained")
            """)
        
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


