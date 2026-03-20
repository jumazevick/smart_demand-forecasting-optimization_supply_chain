from pathlib import Path

import pandas as pd


DELIVERY_COLS = [
    "base_handling_fee",
    "cost_per_km",
    "weight_kg_per_unit",
    "min_order_qty",
    "express_surcharge_pct",
]

WAREHOUSE_RELIABILITY = {
    "W1": 0.96,
    "W2": 0.93,
    "W3": 0.95,
    "W4": 0.91,
    "W5": 0.97,
}


def load_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = base_dir / "data"
    sales_df = pd.read_csv(data_dir / "historical_sales.csv", parse_dates=["date"])
    warehouses_df = pd.read_csv(data_dir / "warehouses.csv")
    distances_df = pd.read_csv(data_dir / "distances.csv")
    return sales_df, warehouses_df, distances_df


def predict_demand(sales_df: pd.DataFrame, growth_factor: float = 1.1) -> pd.DataFrame:
    """Predict next-day demand per store-product using historical average with growth."""
    forecast_df = (
        sales_df.groupby(["store", "product"], as_index=False)["units_sold"]
        .mean()
        .rename(columns={"units_sold": "avg_historical_units_sold"})
    )
    forecast_df["predicted_demand"] = (
        forecast_df["avg_historical_units_sold"] * growth_factor
    ).round(2)
    return forecast_df


def build_store_snapshot(sales_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the latest store stock snapshot plus 30-day finance averages."""
    latest_date = sales_df["date"].max()
    latest_rows = sales_df[sales_df["date"] == latest_date].copy()

    current_stock_df = latest_rows[["store", "product", "stock"]].rename(
        columns={"stock": "current_stock"}
    )

    daily_store_df = (
        sales_df.groupby(["store", "date"], as_index=False)[
            ["revenue", "operating_cost", "profit", "cash_available"]
        ]
        .sum()
        .sort_values(["store", "date"])
    )

    trailing_30_df = daily_store_df.groupby("store", group_keys=False).tail(30)
    finance_df = trailing_30_df.groupby("store", as_index=False).agg(
        avg_daily_revenue=("revenue", "mean"),
        avg_daily_operating_cost=("operating_cost", "mean"),
        avg_daily_profit=("profit", "mean"),
    )

    latest_funds_df = daily_store_df.groupby("store", as_index=False).last()[
        ["store", "cash_available"]
    ].rename(columns={"cash_available": "cash_available_today"})

    finance_df = finance_df.merge(latest_funds_df, on="store", how="left")
    finance_df["projected_cash_available"] = (
        finance_df["cash_available_today"] + finance_df["avg_daily_profit"] * 7
    ).round(2)
    return current_stock_df, finance_df


def plan_inventory(
    forecast_df: pd.DataFrame,
    current_stock_df: pd.DataFrame,
    finance_df: pd.DataFrame,
    warehouses_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare predicted demand against current in-store stock and warehouse capacity."""
    max_stock_df = warehouses_df.groupby("product", as_index=False)["stock"].max().rename(
        columns={"stock": "warehouse_available_stock"}
    )

    plan_df = (
        forecast_df.merge(current_stock_df, on=["store", "product"], how="left")
        .merge(finance_df, on="store", how="left")
        .merge(max_stock_df, on="product", how="left")
    )

    plan_df["current_stock"] = plan_df["current_stock"].fillna(0)
    plan_df["shortage"] = (
        plan_df["predicted_demand"] - plan_df["current_stock"]
    ).clip(lower=0).round(2)
    plan_df["status"] = "ok"
    plan_df.loc[plan_df["shortage"] > 0, "status"] = "needs_restock"
    return plan_df


def build_warehouse_options(
    plan_df: pd.DataFrame,
    warehouses_df: pd.DataFrame,
    distances_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every warehouse option for each store-product shortage."""
    warehouse_metrics_df = warehouses_df.copy()
    warehouse_metrics_df["reliability_score"] = warehouse_metrics_df["warehouse"].map(
        WAREHOUSE_RELIABILITY
    )

    options_df = (
        plan_df[
            [
                "store",
                "product",
                "predicted_demand",
                "current_stock",
                "shortage",
                "projected_cash_available",
            ]
        ]
        .merge(distances_df, on="store", how="left")
        .merge(warehouse_metrics_df, on=["warehouse", "product"], how="left")
    )

    options_df["recommended_order_qty"] = options_df[["shortage", "min_order_qty"]].max(axis=1)
    options_df.loc[options_df["shortage"] <= 0, "recommended_order_qty"] = 0

    options_df["is_urgent"] = (
        options_df["shortage"] / options_df["predicted_demand"].replace(0, 1)
    ) > 0.35

    options_df["base_plus_variable_cost"] = (
        options_df["base_handling_fee"].where(options_df["recommended_order_qty"] > 0, 0)
        + (
            options_df["weight_kg_per_unit"]
            * options_df["recommended_order_qty"]
            * options_df["cost_per_km"]
            * options_df["distance_km"]
        )
    )

    options_df["express_multiplier"] = 1.0
    urgent_mask = options_df["is_urgent"] & (options_df["recommended_order_qty"] > 0)
    options_df.loc[urgent_mask, "express_multiplier"] = (
        1 + options_df.loc[urgent_mask, "express_surcharge_pct"] / 100
    )
    options_df["delivery_cost"] = (
        options_df["base_plus_variable_cost"] * options_df["express_multiplier"]
    ).round(2)

    options_df["can_fulfill"] = options_df["stock"] >= options_df["recommended_order_qty"]
    options_df["can_afford_order"] = (
        options_df["projected_cash_available"] >= options_df["delivery_cost"]
    )
    options_df["stock_gap"] = (
        options_df["recommended_order_qty"] - options_df["stock"]
    ).clip(lower=0).round(2)

    group_cols = ["store", "product"]
    options_df["cost_rank"] = options_df.groupby(group_cols)["delivery_cost"].rank(
        method="dense", ascending=True
    )
    options_df["distance_rank"] = options_df.groupby(group_cols)["distance_km"].rank(
        method="dense", ascending=True
    )
    options_df["reliability_rank"] = options_df.groupby(group_cols)["reliability_score"].rank(
        method="dense", ascending=False
    )
    options_df["stock_penalty"] = (
        (~options_df["can_fulfill"]).astype(int) * 10
        + options_df["stock_gap"] / options_df["recommended_order_qty"].replace(0, 1)
    )
    options_df["decision_score"] = (
        0.4 * options_df["cost_rank"]
        + 0.25 * options_df["distance_rank"]
        + 0.2 * options_df["reliability_rank"]
        + 0.15 * options_df["stock_penalty"]
    ).round(3)
    return options_df


def assign_best_warehouse(plan_df: pd.DataFrame, options_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best warehouse using cost, distance, reliability, stock, and affordability."""
    best_option_df = (
        options_df.sort_values(
            [
                "store",
                "product",
                "can_fulfill",
                "can_afford_order",
                "decision_score",
                "distance_km",
            ],
            ascending=[True, True, False, False, True, True],
        )
        .groupby(["store", "product"], as_index=False)
        .first()
    )

    output_df = plan_df.merge(
        best_option_df[
            [
                "store",
                "product",
                "warehouse",
                "distance_km",
                "stock",
                "reliability_score",
                *DELIVERY_COLS,
                "recommended_order_qty",
                "delivery_cost",
                "can_fulfill",
                "can_afford_order",
                "decision_score",
            ]
        ].rename(columns={"stock": "selected_warehouse_stock"}),
        on=["store", "product"],
        how="left",
    )

    output_df["ship_qty"] = output_df["recommended_order_qty"].round(2)
    output_df.loc[output_df["shortage"] <= 0, "ship_qty"] = 0
    output_df.loc[
        (output_df["shortage"] > 0) & (~output_df["can_fulfill"]),
        "status",
    ] = "warehouse_stock_risk"
    output_df.loc[
        (output_df["shortage"] > 0)
        & (output_df["can_fulfill"])
        & (~output_df["can_afford_order"]),
        "status",
    ] = "unaffordable"
    return output_df


def run_pipeline() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    sales_df, warehouses_df, distances_df = load_data(base_dir)

    forecast_df = predict_demand(sales_df, growth_factor=1.1)
    current_stock_df, finance_df = build_store_snapshot(sales_df)
    plan_df = plan_inventory(forecast_df, current_stock_df, finance_df, warehouses_df)
    options_df = build_warehouse_options(plan_df, warehouses_df, distances_df)
    output_df = assign_best_warehouse(plan_df, options_df)

    data_dir = base_dir / "data"
    output_path = data_dir / "output.csv"
    options_path = data_dir / "warehouse_options.csv"
    output_df.to_csv(output_path, index=False)
    options_df.to_csv(options_path, index=False)

    print("Pipeline complete!")
    print(f"Saved: {output_path}")
    print(f"Saved: {options_path}")
    print(f"\nProcessed {len(output_df)} store-product combinations")
    print(f"From {len(sales_df)} historical store-product records")
    print(f"\nRestock needs identified: {len(output_df[output_df['shortage'] > 0])}")
    print(f"Affordable orders: {int(output_df['can_afford_order'].fillna(False).sum())}")
    print(f"Total delivery cost: ${output_df['delivery_cost'].sum():,.2f}")
    print("\nOutput preview:")
    print(output_df.head(10))
    return output_df


if __name__ == "__main__":
    run_pipeline()

